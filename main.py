import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.config import settings, ChunkingStrategy
from src.ingestion import load_file, load_directory, chunk_documents, Embedder
from src.retrieval.fusion import Reranker, hybrid_retrieve
from src.generation.generator import generate_answer


def prompt_chunking_strategy() -> ChunkingStrategy:
    options = list(ChunkingStrategy)
    labels = "/".join(s.value for s in options)
    raw = input(f"Chunking strategy [{labels}] (default: {settings.default_chunk_strategy.value}): ").strip().lower()
    if not raw:
        return settings.default_chunk_strategy

    try:
        return ChunkingStrategy(raw)
    except ValueError:
        print(f"Unknown strategy '{raw}', using default: {settings.default_chunk_strategy.value}")
        return settings.default_chunk_strategy


def insert_document(embedder: Embedder) -> None:
    path = input("Path to a file or a folder: ").strip().strip('"')
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return

    raw_documents = load_directory(path) if os.path.isdir(path) else load_file(path)
    if not raw_documents:
        print("No loadable documents found.")
        return

    strategy = prompt_chunking_strategy()
    chunks = chunk_documents(raw_documents, strategy=strategy)
    embedder.embed(chunks)
    print(f"Ingested {len(raw_documents)} document(s) -> {len(chunks)} chunks ({strategy.value}).")


def view_index(embedder: Embedder) -> None:
    results = embedder.collection.get(include=["documents", "metadatas"])
    count = len(results["ids"])
    print(f"{count} chunk(s) in collection '{settings.chroma_collection_name}':")
    for doc, meta in zip(results["documents"], results["metadatas"]):
        line = f"  {meta} -> {doc[:80]!r}"
        print(line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))


def ask_question(embedder: Embedder, reranker: Reranker) -> None:
    query = input("Question: ").strip()
    if not query:
        return

    retrieved = hybrid_retrieve(query, embedder.collection, reranker, k=settings.reranker_top_k)
    if not retrieved:
        print("No chunks retrieved — is anything ingested yet?")
        return

    result = generate_answer(query, retrieved)
    print(f"\nAnswer: {result.answer}")
    print(f"Has answer: {result.has_answer} | Confidence: {result.confidence:.2f}")
    print("Sources used:")
    for i, r in enumerate(retrieved, start=1):
        source = r.metadata.source.source_name if r.metadata and r.metadata.source else "unknown"
        print(f"  [{i}] {source} (score={r.score:.3f})")


MENU = """
1) Insert document(s)
2) View index
3) Ask a question
4) Exit
"""


def main() -> None:
    embedder = Embedder()
    reranker = Reranker()

    while True:
        print(MENU)
        choice = input("Choice: ").strip()

        if choice == "1":
            insert_document(embedder)
        elif choice == "2":
            view_index(embedder)
        elif choice == "3":
            ask_question(embedder, reranker)
        elif choice == "4":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
