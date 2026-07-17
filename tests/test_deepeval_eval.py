"""
DeepEval-based evaluation for the RAG pipeline (Lesson 5, framework version).

Lives in tests/ per DeepEval's own convention (their quickstart puts test files
in a `tests/` folder), but every test here is marked `llm_eval` and excluded
from the default `pytest` run via pytest.ini's `addopts = -m "not llm_eval"`.
That's deliberate: these tests make real, paid, non-deterministic OpenAI calls
(unlike everything else in tests/, which is fast and mocked) — mixing them into
the default gate would make `pytest tests/ -v` slow, flaky, and expensive every
time it runs.

Run the whole file with:
    deepeval test run tests/test_deepeval_eval.py -m llm_eval

test_deepeval_setup_works and test_deepeval_refuses_unanswerable don't call
assert_test(golden=...) and also run fine under plain
`pytest tests/test_deepeval_eval.py -m llm_eval -v -s` — but
test_deepeval_rag_pipeline needs the `deepeval test run` CLI specifically:
assert_test(golden=...) reads the just-executed @observe trace off a
ContextVar that DeepEval's own Observer clears the instant the outer traced
function returns. DeepEval's pytest plugin is what keeps that trace alive
across the rest of the test body (by wrapping the whole test in its own
outer span) — but it only does that when the DEEPEVAL env var is set, which
only `deepeval test run` sets. Plain pytest silently fails with
"No active trace found for this test."

DATASET FORMAT
----------------
evals/datasets/golden_dataset.json uses DeepEval's native Golden schema
(input, expected_output, additional_metadata) and is loaded through
EvaluationDataset.add_goldens_from_json_file rather than a hand-rolled
json.load() — this is DeepEval's own dataset abstraction, so the same file
could later be pushed to / pulled from Confident AI's cloud without a rewrite.

WHY REFERENCE-FREE METRICS, NOT GEval-AGAINST-expected_output
------------------------------------------------------------------
An earlier version of this file graded every case with GEval compared against
a written reference answer in expected_output. That produced false failures:
GEval treats expected_output as a real answer to match, and penalizes
deviation from its exact wording, structure, or level of detail — even when
the actual answer was factually correct. Per DeepEval's own RAG evaluation
guide (docs/getting-started-rag), the recommended RAG metrics — Faithfulness,
AnswerRelevancy, ContextualRelevancy — are reference-free by design: they
check groundedness and topical relevance directly against what the pipeline
actually retrieved/produced, not against a hand-written "correct" answer. That
sidesteps the phrasing-brittleness problem entirely instead of trying to word
a criteria string carefully enough to work around it. expected_output is kept
in the dataset for readability/documentation, but the RAG-quality tests below
don't grade against it.

COMPONENT-LEVEL EVALUATION (retriever vs. generator)
---------------------------------------------------------
A single end-to-end score can't tell you whether a bad answer came from bad
retrieval or bad generation — this is the same "per-stage metrics" principle
the original hand-rolled run_eval.py was built around. DeepEval's idiomatic
way to get that separation, when you own the pipeline code (we do), is
component-level tracing: @observe() wraps a function as a traced span,
update_current_span() attaches an LLMTestCase to that span, and metrics
declared on @observe(metrics=[...]) score that span specifically when
assert_test(golden=golden) runs. ContextualRelevancyMetric lives on the
retriever span (did we retrieve relevant chunks?), AnswerRelevancyMetric on
the generator span (does the answer address the question?), and
FaithfulnessMetric at the trace level (is the final answer grounded in what
was actually retrieved?).
"""

import os
from pathlib import Path

# FaithfulnessMetric's default async_mode=True path fires 2+ judge sub-calls
# concurrently via asyncio.gather inside a reused/nested event loop. Reproduced
# standalone (outside pytest/tracing entirely) that this concurrent path hangs
# and times out on this machine/environment, while the exact same calls made
# sequentially, or made concurrently via a raw AsyncOpenAI client (bypassing
# DeepEval's own async/event-loop handling), both succeed in seconds. Metrics
# below are constructed with async_mode=False to force DeepEval's fully
# sequential code path (still real, still judged — just no asyncio.gather).
# Kept as a safety margin: even sequential judge calls for FaithfulnessMetric
# (truths -> claims -> verdicts -> reason, 4 real LLM calls) can legitimately
# take longer than DeepEval's default 88.5s per-attempt cap on slower content.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "150")

import chromadb
import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.tracing import observe, update_current_span, update_current_trace
from openai import OpenAI

from src.config import settings
from src.generation.generator import generate_answer
from src.retrieval.fusion import Reranker, hybrid_retrieve

DATASET_PATH = Path(__file__).parent.parent / "evals" / "datasets" / "golden_dataset.json"

pytestmark = pytest.mark.llm_eval

# Built once at module scope (not a fixture): @observe(metrics=[...]) below needs
# a real metric instance at *decoration* time (module import), before any pytest
# fixture would exist. GPTModel construction is cheap (no model download, unlike
# Reranker), so this is safe to build eagerly.
_JUDGE = GPTModel(model=settings.judgement_model, api_key=settings.openai_api_key)


def _load_goldens():
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(str(DATASET_PATH))
    return dataset.goldens


_ALL_GOLDENS = _load_goldens()
ANSWERABLE_GOLDENS = [g for g in _ALL_GOLDENS if g.additional_metadata["should_answer"]]
UNANSWERABLE_GOLDENS = [g for g in _ALL_GOLDENS if not g.additional_metadata["should_answer"]]


def test_deepeval_setup_works():
    """No RAG pipeline involved at all — this only proves DeepEval + the OpenAI
    key + GEval are wired correctly, isolated from any of our own code. If this
    fails, the problem is environment/setup, not the RAG system."""
    judge = GPTModel(model=settings.judgement_model, api_key=settings.openai_api_key)
    correctness = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=0.5,
    )
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris is the capital of France.",
    )
    assert_test(test_case, [correctness])


# Module-scoped fixtures: Reranker() loads a CrossEncoder model from disk on
# construction, and the Chroma/OpenAI/judge clients are cheap to share, expensive
# to rebuild per parametrized case.
@pytest.fixture(scope="module")
def collection():
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(name=settings.chroma_collection_name)


@pytest.fixture(scope="module")
def reranker() -> Reranker:
    return Reranker()


@pytest.fixture(scope="module")
def openai_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


@observe(type="retriever", metrics=[ContextualRelevancyMetric(model=_JUDGE, async_mode=False)])
def _traced_retrieve(query: str, collection, reranker: Reranker):
    """The retriever span. ContextualRelevancyMetric scores whether the retrieved
    chunks are actually relevant to the query — this is retrieval quality, isolated
    from anything the generator does with those chunks afterward."""
    retrieved = hybrid_retrieve(query, collection, reranker, k=settings.reranker_top_k)
    update_current_span(
        test_case=LLMTestCase(input=query, retrieval_context=[r.text for r in retrieved])
    )
    return retrieved


@observe(type="llm", metrics=[AnswerRelevancyMetric(model=_JUDGE, async_mode=False)])
def _traced_generate(query: str, retrieved: list, openai_client: OpenAI):
    """The generator span. AnswerRelevancyMetric scores whether the answer actually
    addresses the question — independent of whether the right chunks were retrieved,
    which is _traced_retrieve's job to catch, not this span's."""
    gen_result = generate_answer(query, retrieved, openai_client)
    update_current_span(test_case=LLMTestCase(input=query, actual_output=gen_result.answer))
    return gen_result


@observe()
def _traced_rag_pipeline(query: str, collection, reranker: Reranker, openai_client: OpenAI) -> str:
    """The outer trace, tying both spans together. update_current_trace's
    retrieval_context (not just the generator's actual_output) is what lets the
    trace-level FaithfulnessMetric — attached in the test below, not here — check
    whether the final answer is actually grounded in what was retrieved."""
    retrieved = _traced_retrieve(query, collection, reranker)
    gen_result = _traced_generate(query, retrieved, openai_client)
    update_current_trace(
        input=query,
        output=gen_result.answer,
        retrieval_context=[r.text for r in retrieved],
    )
    return gen_result.answer


@pytest.mark.parametrize("golden", ANSWERABLE_GOLDENS, ids=[g.name for g in ANSWERABLE_GOLDENS])
def test_deepeval_rag_pipeline(golden, collection, reranker, openai_client):
    """Runs every answerable golden case through the traced pipeline. assert_test(golden=golden)
    scores the just-executed trace: ContextualRelevancyMetric and AnswerRelevancyMetric fire
    automatically because they're attached to their respective spans via @observe(metrics=[...]),
    and FaithfulnessMetric fires here at the trace level. All three are reference-free — none
    of them read golden.expected_output — so a factually-correct-but-differently-phrased answer
    can't fail on wording the way the earlier GEval-vs-reference version did."""
    _traced_rag_pipeline(golden.input, collection, reranker, openai_client)
    assert_test(golden=golden, metrics=[FaithfulnessMetric(model=_JUDGE, async_mode=False)])


@pytest.mark.parametrize("golden", UNANSWERABLE_GOLDENS, ids=[g.name for g in UNANSWERABLE_GOLDENS])
def test_deepeval_refuses_unanswerable(golden, collection, reranker, openai_client):
    """No GEval metric here on purpose — DeepEval's RAG/correctness metrics all assume
    the system SHOULD produce an answer to compare against a reference; they have no
    "correctly refused" mode. A confident wrong answer here is worse than a wrong answer
    to a real question, so this is a hard assert on has_answer, not a graded score."""
    retrieved = hybrid_retrieve(golden.input, collection, reranker, k=settings.reranker_top_k)
    gen_result = generate_answer(golden.input, retrieved, openai_client)

    assert gen_result.has_answer is False, (
        f"expected the system to refuse (no answer in context), "
        f"but it answered: {gen_result.answer!r}"
    )
