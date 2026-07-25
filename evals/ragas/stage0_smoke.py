"""Stage 0 smoke test — proves the eval plumbing works end to end on one question:

    ingest corpus_sample/ -> hybrid_retrieve -> generate_answer -> score with RAGAS Faithfulness

Run from the repo root with:
    python -m evals.ragas.stage0_smoke

Nothing here is the "real" eval yet — no golden dataset, one hand-picked question, one
metric. The only goal is confirming every piece of the chain talks to the next one
correctly before Stage 1 scales it up.
"""

import asyncio
from pathlib import Path

import chromadb
from openai import AsyncOpenAI, OpenAI

# Must be imported before anything from `ragas` — see _compat.py for why.
from evals.ragas import _compat  # noqa: F401
from ragas.llms.base import llm_factory
from ragas.metrics.collections import Faithfulness

from src.config import settings
from src.generation.generator import generate_answer
from src.ingestion.chuncker import chunk_documents
from src.ingestion.embedder import Embedder
from src.ingestion.loader import load_directory
from src.retrieval.fusion import Reranker, hybrid_retrieve

CORPUS_DIR = Path(__file__).parent / "corpus_sample"
CHROMA_PATH = Path(__file__).parent / "chroma_stage0"
COLLECTION_NAME = "rag_docs_ragas_stage0"

QUESTION = "who is harry poter?"


def ingest() -> chromadb.api.models.Collection.Collection:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Embedder.embed() upserts by a stable chunk id, so re-running this on an
    # already-populated collection is a no-op, not a duplicate.
    if collection.count() == 0:
        raw_docs = load_directory(str(CORPUS_DIR))
        chunks = chunk_documents(raw_docs)
        openai_client = OpenAI(api_key=settings.openai_api_key)
        Embedder(
            chroma_client=client,
            openai_client=openai_client,
            collection_name=COLLECTION_NAME,
        ).embed(chunks)

    return collection


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


def main() -> None:
    collection = ingest()

    openai_client = OpenAI(api_key=settings.openai_api_key)
    reranker = Reranker()

    retrieved = hybrid_retrieve(QUESTION, collection, reranker, k=settings.reranker_top_k)
    gen_result = generate_answer(QUESTION, retrieved, openai_client)
    
    print(f"Question: {QUESTION}")
    print(f"Answer:   {gen_result.answer}")
    print(f"Retrieved {len(retrieved)} chunks")

    contexts = [r.text for r in retrieved]
    faithfulness = asyncio.run(score_faithfulness(QUESTION, gen_result.answer, contexts))

    print(f"Faithfulness score: {faithfulness:.2f}")


if __name__ == "__main__":
    main()
