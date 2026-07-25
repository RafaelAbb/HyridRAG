"""Stage 0 smoke test — proves the eval plumbing works end to end on one question:

    ingest corpus_sample/ -> hybrid_retrieve -> generate_answer -> score with RAGAS Faithfulness

Run from the repo root with:
    python -m evals.ragas.stage0_smoke

Nothing here is the "real" eval yet — the 5-page sample corpus, a single question
pulled from the golden set, one metric. The only goal is confirming every piece of
the chain talks to the next one correctly, cheaply, before trusting the full
Stage 2 run in run_eval.py.
"""

import asyncio
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

# Must be imported before anything from `ragas` — see _compat.py for why.
from evals.ragas import _compat  # noqa: F401
from evals.ragas._shared import ingest, load_golden_dataset
from ragas.llms.base import llm_factory
from ragas.metrics.collections import Faithfulness

from src.config import settings
from src.generation.generator import generate_answer
from src.retrieval.fusion import Reranker, hybrid_retrieve

CORPUS_DIR = Path(__file__).parent / "corpus_sample"
CHROMA_PATH = Path(__file__).parent / "chroma_stage0"
COLLECTION_NAME = "rag_docs_ragas_stage0"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


async def score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    # ragas metrics call an LLM judge, so scoring is async — llm_factory wraps a
    # plain AsyncOpenAI client into the shape ragas' metrics expect.
    llm = llm_factory(settings.judgement_model, client=AsyncOpenAI(api_key=settings.openai_api_key))
    metric = Faithfulness(llm=llm)

    result = await metric.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )
    return result.value


def pick_answerable_question(golden: list[dict]) -> dict:
    # golden_dataset.json was generated from the FULL corpus/ (36 pages), but
    # this smoke test only ingests the 5-page corpus_sample/ — most golden
    # questions are about pages that aren't even in this small collection.
    # Picking one at random would mostly test "does the pipeline correctly
    # say I don't know", not the happy path. Pick the first golden question
    # whose reference_contexts actually appear in one of the sample files.
    sample_texts = [p.read_text(encoding="utf-8") for p in CORPUS_DIR.glob("*.md")]
    for item in golden:
        if any(ctx[:200] in text for ctx in item["reference_contexts"] for text in sample_texts):
            return item
    return golden[0]


def main() -> None:
    collection = ingest(CORPUS_DIR, CHROMA_PATH, COLLECTION_NAME)
    golden = load_golden_dataset(GOLDEN_DATASET_PATH)
    question_item = pick_answerable_question(golden)
    question = question_item["user_input"]

    openai_client = OpenAI(api_key=settings.openai_api_key)
    reranker = Reranker()

    retrieved = hybrid_retrieve(question, collection, reranker, k=settings.reranker_top_k)
    gen_result = generate_answer(question, retrieved, openai_client)

    print(f"Question: {question}")
    print(f"Answer:   {gen_result.answer}")
    print(f"Retrieved {len(retrieved)} chunks")

    contexts = [r.text for r in retrieved]
    faithfulness = asyncio.run(score_faithfulness(question, gen_result.answer, contexts))

    print(f"Faithfulness score: {faithfulness:.2f}")


if __name__ == "__main__":
    main()
