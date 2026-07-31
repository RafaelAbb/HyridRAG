"""Run ONE golden question through the real pipeline and print everything —
retrieved chunks side by side with what the golden set expected — for manually
diagnosing a specific low-scoring question instead of re-running all 35.

Run from the repo root with:
    python -m evals.ragas.inspect_question <index>
    python -m evals.ragas.inspect_question 6          # 0-based index into golden_dataset.json
    python -m evals.ragas.inspect_question "APIRouter" # or a substring of the question text
"""

import sys
from pathlib import Path

# Windows console default codepage (e.g. cp1255) can't encode emoji that show
# up in the FastAPI docs corpus (e.g. the VS Code extension blurb) — force
# stdout to utf-8 so printing a retrieved chunk never crashes the script.
sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI

from evals.ragas._shared import ingest, load_golden_dataset
from src.config import settings
from src.generation.generator import generate_answer
from src.retrieval.fusion import Reranker, hybrid_retrieve

CORPUS_DIR = Path(__file__).parent / "corpus"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
CHROMA_PATH = Path(__file__).parent / "chroma_eval"
COLLECTION_NAME = "rag_docs_ragas_eval"


def find_question(golden: list[dict], selector: str) -> dict:
    if selector.isdigit():
        return golden[int(selector)]
    matches = [item for item in golden if selector.lower() in item["user_input"].lower()]
    if not matches:
        raise SystemExit(f"No golden question matches {selector!r}")
    if len(matches) > 1:
        for i, m in enumerate(matches):
            print(f"  [{i}] {m['user_input']}")
        raise SystemExit(f"{len(matches)} questions match {selector!r} — be more specific")
    return matches[0]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m evals.ragas.inspect_question <index-or-substring>")

    golden = load_golden_dataset(GOLDEN_DATASET_PATH)
    item = find_question(golden, sys.argv[1])
    question = item["user_input"]

    collection = ingest(CORPUS_DIR, CHROMA_PATH, COLLECTION_NAME)
    reranker = Reranker()
    openai_client = OpenAI(api_key=settings.openai_api_key)

    retrieved = hybrid_retrieve(question, collection, reranker, k=settings.reranker_top_k)
    gen_result = generate_answer(question, retrieved, openai_client)

    print(f"QUESTION: {question}\n")
    print(f"GENERATED ANSWER:\n{gen_result.answer}\n")

    print(f"RETRIEVED ({len(retrieved)} chunks):")
    for i, r in enumerate(retrieved, start=1):
        source = r.metadata.source.source_name if r.metadata and r.metadata.source else "?"
        print(f"  [{i}] score={r.score:.3f} source={source}")
        print(f"      {r.text[:200]!r}")

    print(f"\nGOLDEN REFERENCE ANSWER:\n{item['reference']}\n")

    print(f"GOLDEN REFERENCE_CONTEXTS ({len(item['reference_contexts'])}):")
    for i, ctx in enumerate(item["reference_contexts"], start=1):
        print(f"  [{i}] {ctx[:200]!r}")


if __name__ == "__main__":
    main()
