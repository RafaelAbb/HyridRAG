"""Shared helpers for the RAGAS eval scripts (stage0_smoke.py, run_eval.py).

Both scripts ingest a corpus into their own Chroma collection and load
questions from a golden-set JSON file — this is that common logic, factored
out so a fix here (e.g. batch size handling) doesn't need to be made twice.
"""

import json
from pathlib import Path

import chromadb
from openai import OpenAI

from src.config import settings
from src.ingestion.chuncker import chunk_documents
from src.ingestion.embedder import Embedder
from src.ingestion.loader import load_directory


def ingest(
    corpus_dir: Path, chroma_path: Path, collection_name: str
) -> chromadb.api.models.Collection.Collection:
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(name=collection_name)

    # Embedder.embed() upserts by a stable chunk id, so re-running this on an
    # already-populated collection is a no-op, not a duplicate.
    if collection.count() == 0:
        raw_docs = load_directory(str(corpus_dir))
        chunks = chunk_documents(raw_docs)
        openai_client = OpenAI(api_key=settings.openai_api_key)
        Embedder(
            chroma_client=client,
            openai_client=openai_client,
            collection_name=collection_name,
        ).embed(chunks)

    return collection


def load_golden_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
