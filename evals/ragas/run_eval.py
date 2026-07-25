"""Stage 2 — the real eval harness.

For every question in golden_dataset.json (human-curated in Stage 1): run it
through the actual RAG pipeline (hybrid_retrieve -> generate_answer, unchanged
from src/), then score the result with RAGAS's four core RAG metrics:

    Faithfulness       - does the answer only claim things the retrieved context supports?
    Answer Relevancy   - does the answer actually address the question?
    Context Precision  - of what was retrieved, how much was actually useful?
    Context Recall     - did retrieval find everything needed to answer correctly?

Context Precision/Recall grade retrieval against golden_dataset.json's
reference_contexts, so the eval Chroma collection here is built from the SAME
evals/ragas/corpus/ files the golden set was generated from - ingesting a
different corpus would make those two scores meaningless.

Sequential for now: one question at a time (retrieve -> generate -> score x4),
not concurrent. Simple to debug; fine at ~35 questions.

Run from the repo root with:
    python -m evals.ragas.run_eval
"""

import asyncio
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from openai import AsyncOpenAI, OpenAI

# Must be imported before anything from `ragas` — see _compat.py for why.
from evals.ragas import _compat  # noqa: F401
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms.base import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from src.config import settings
from src.generation.generator import generate_answer
from src.ingestion.chuncker import chunk_documents
from src.ingestion.embedder import Embedder
from src.ingestion.loader import load_directory
from src.retrieval.fusion import Reranker, hybrid_retrieve

CORPUS_DIR = Path(__file__).parent / "corpus"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
CHROMA_PATH = Path(__file__).parent / "chroma_eval"
COLLECTION_NAME = "rag_docs_ragas_eval"
RESULTS_DIR = Path(__file__).parent.parent / "results"

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


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


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


async def score_one(
    metrics: dict,
    question: str,
    answer: str,
    contexts: list[str],
    reference: str,
) -> dict[str, float]:
    faithfulness_result = await metrics["faithfulness"].ascore(
        user_input=question, response=answer, retrieved_contexts=contexts
    )
    answer_relevancy_result = await metrics["answer_relevancy"].ascore(
        user_input=question, response=answer
    )
    context_precision_result = await metrics["context_precision"].ascore(
        user_input=question, reference=reference, retrieved_contexts=contexts
    )
    context_recall_result = await metrics["context_recall"].ascore(
        user_input=question, retrieved_contexts=contexts, reference=reference
    )

    return {
        "faithfulness": faithfulness_result.value,
        "answer_relevancy": answer_relevancy_result.value,
        "context_precision": context_precision_result.value,
        "context_recall": context_recall_result.value,
    }


def build_metrics() -> dict:
    # One shared LLM/embeddings client for every question — not recreated
    # per-question, same "batch, don't call per-item" principle as Embedder.
    llm = llm_factory(settings.judgement_model, client=AsyncOpenAI(api_key=settings.openai_api_key))
    embeddings = RagasOpenAIEmbeddings(
        client=AsyncOpenAI(api_key=settings.openai_api_key), model=settings.embedding_model
    )
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def run_all(
    golden: list[dict], collection, reranker, openai_client, metrics: dict, incremental_path: Path
) -> list[dict]:
    results = []

    # Append one JSON line per question as soon as it's scored, and flush
    # immediately — a crash (rate limit, quota, network) partway through a
    # 35-question paid run loses at most the in-flight question, not
    # everything scored before it.
    with open(incremental_path, "a", encoding="utf-8") as incremental_file:
        for i, item in enumerate(golden, start=1):
            question = item["user_input"]
            reference = item["reference"]

            retrieved = hybrid_retrieve(question, collection, reranker, k=settings.reranker_top_k)
            gen_result = generate_answer(question, retrieved, openai_client)
            contexts = [r.text for r in retrieved]

            scores = await score_one(metrics, question, gen_result.answer, contexts, reference)

            print(f"[{i}/{len(golden)}] {question}")
            print(f"    {' | '.join(f'{k}={v:.2f}' for k, v in scores.items())}")

            result = {
                "question": question,
                "reference": reference,
                "answer": gen_result.answer,
                "retrieved_contexts": contexts,
                "scores": scores,
            }
            results.append(result)

            incremental_file.write(json.dumps(result) + "\n")
            incremental_file.flush()

    return results


def summarize(results: list[dict]) -> dict[str, float]:
    return {
        name: statistics.mean(r["scores"][name] for r in results)
        for name in METRIC_NAMES
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    incremental_path = RESULTS_DIR / f"{timestamp}.jsonl"

    collection = ingest()
    golden = load_golden_dataset()
    print(f"Loaded {len(golden)} golden questions")
    print(f"Saving each question's result to {incremental_path} as it completes\n")

    reranker = Reranker()
    openai_client = OpenAI(api_key=settings.openai_api_key)
    metrics = build_metrics()

    results = asyncio.run(run_all(golden, collection, reranker, openai_client, metrics, incremental_path))

    averages = summarize(results)

    print("\n=== Summary (mean across all questions) ===")
    for name, value in averages.items():
        print(f"{name:20s} {value:.3f}")

    output_path = RESULTS_DIR / f"{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "averages": averages, "results": results}, f, indent=2)

    print(f"\nWrote full results to {output_path}")


if __name__ == "__main__":
    main()
