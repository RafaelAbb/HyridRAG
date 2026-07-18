"""
Minimal DeepEval smoke test for the RAG pipeline, no component-level tracing.

Unlike tests/test_deepeval_eval.py (which uses @observe spans + assert_test(golden=...)
and therefore needs the `deepeval test run` CLI specifically), this file builds a plain
LLMTestCase directly from the pipeline's own output and calls assert_test(test_case, [...]),
so it runs fine under plain pytest too:

    pytest tests/llm/rag_llm_eval.py -m llm_eval -v -s
    deepeval test run tests/llm/rag_llm_eval.py -m llm_eval

One golden case for now — add more strings to QUESTIONS below when ready.
"""

import os

# FaithfulnessMetric makes several sequential judge calls; DeepEval's default
# per-attempt timeout has been observed to be too tight for that in this
# environment. See tests/test_deepeval_eval.py for the full investigation.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "150")

import chromadb
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase
from openai import OpenAI

from src.config import settings
from src.generation.generator import generate_answer
from src.retrieval.fusion import Reranker, hybrid_retrieve

pytestmark = pytest.mark.llm_eval

_JUDGE = GPTModel(model=settings.judgement_model, api_key=settings.openai_api_key)

QUESTIONS = [
    "What paper introduced the Transformer architecture?",
    # add more questions here later
]


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


@pytest.mark.parametrize("question", QUESTIONS)
def test_rag_pipeline(question, collection, reranker, openai_client):
    retrieved = hybrid_retrieve(question, collection, reranker, k=settings.reranker_top_k)
    gen_result = generate_answer(question, retrieved, openai_client)

    test_case = LLMTestCase(
        input=question,
        actual_output=gen_result.answer,
        retrieval_context=[r.text for r in retrieved],
    )
    assert_test(
        test_case,
        [
            FaithfulnessMetric(model=_JUDGE, async_mode=False),
            AnswerRelevancyMetric(model=_JUDGE, async_mode=False),
        ],
    )
