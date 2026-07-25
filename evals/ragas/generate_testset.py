"""Stage 1 — bootstrap candidate golden-set questions from the FastAPI docs corpus
using RAGAS's TestsetGenerator.

This produces CANDIDATES, not the final golden set. Output goes to
candidate_testset.json for human review — read every row, fix wording/expected
answers that are off, discard anything bad, and save the survivors as
golden_dataset.json. That curated file is the permanent, fixed dataset every
later eval run scores against.

Run from the repo root with:
    python -m evals.ragas.generate_testset

Costs real OpenAI API calls (LLM + embeddings) to build the knowledge graph
across ~40 documents — this is the slow, one-time step, not something you
re-run on every code change.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Must be imported before anything from `ragas` — see _compat.py for why.
from evals.ragas import _compat  # noqa: F401
from ragas.testset import TestsetGenerator

from src.config import settings

CORPUS_DIR = Path(__file__).parent / "corpus"
OUTPUT_PATH = Path(__file__).parent / "candidate_testset.json"

TESTSET_SIZE = 35  # candidates to generate; curation will narrow this to ~20-30


def load_corpus_as_documents() -> list[Document]:
    documents = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append(Document(page_content=text, metadata={"source": path.name}))
    return documents


def main() -> None:
    docs = load_corpus_as_documents()
    print(f"Loaded {len(docs)} documents from {CORPUS_DIR}")

    generator = TestsetGenerator.from_langchain(
        llm=ChatOpenAI(model=settings.judgement_model, api_key=settings.openai_api_key),
        embedding_model=OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key),
    )

    # raise_exceptions=False: a handful of thin/heading-light docs fail the
    # knowledge-graph HeadlineSplitter step (known ragas rough edge — the
    # HeadlinesExtractor sometimes comes back empty for short pages). Without
    # this flag, one bad node aborts the entire run; with it, that node is
    # skipped and the rest of the corpus still produces a testset.
    testset = generator.generate_with_langchain_docs(
        docs, testset_size=TESTSET_SIZE, raise_exceptions=False
    )

    df = testset.to_pandas()
    df.to_json(OUTPUT_PATH, orient="records", indent=2, force_ascii=False)

    print(f"Wrote {len(df)} candidate questions to {OUTPUT_PATH}")
    print("Next: review every row by hand, edit/discard as needed, save survivors as golden_dataset.json")


if __name__ == "__main__":
    main()
