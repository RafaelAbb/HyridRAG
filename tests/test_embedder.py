import uuid
from unittest.mock import MagicMock
import chromadb
import pytest

from src.ingestion.base import Chunk, ChunkingStrategy, DocumentMetadata, Source
from src.ingestion.embedder import Embedder


# ── helpers ────────────────────────────────────────────────────────────────────

def make_chunk(source="doc.txt", chunk_id=0, strategy=ChunkingStrategy.FIXED, content="hello world"):
    metadata = DocumentMetadata(source=Source(source_name=source))
    return Chunk(content=content, metadata=metadata, chunk_id=chunk_id, chunk_strategy=strategy)


def make_embedder(n_dims=8):
    """Embedder wired to in-memory ChromaDB and a fake OpenAI client.
    Each call gets a unique collection name to prevent cross-test state leakage."""
    fake_openai = MagicMock()
    fake_openai.embeddings.create.side_effect = lambda model, input: _fake_response(input, n_dims)
    return Embedder(
        chroma_client=chromadb.EphemeralClient(),
        openai_client=fake_openai,
        collection_name=f"test_{uuid.uuid4().hex}",
    )


def _fake_response(texts, n_dims):
    """Build an OpenAI-shaped response with zero vectors."""
    response = MagicMock()
    response.data = [MagicMock(embedding=[0.0] * n_dims) for _ in texts]
    return response


# ── _generate_id ───────────────────────────────────────────────────────────────

class TestGenerateId:

    def test_stable_across_calls(self):
        embedder = make_embedder()
        chunk = make_chunk()
        assert embedder.generate_id(chunk) == embedder.generate_id(chunk)

    def test_includes_source_chunk_id_and_strategy(self):
        embedder = make_embedder()
        chunk = make_chunk(source="file.txt", chunk_id=3, strategy=ChunkingStrategy.RECURSIVE)
        id_ = embedder.generate_id(chunk)
        assert "file.txt" in id_
        assert "3" in id_
        assert "RECURSIVE" in id_

    def test_different_strategy_different_id(self):
        embedder = make_embedder()
        fixed = make_chunk(strategy=ChunkingStrategy.FIXED)
        semantic = make_chunk(strategy=ChunkingStrategy.SEMANTIC)
        assert embedder.generate_id(fixed) != embedder.generate_id(semantic)

    def test_different_chunk_id_different_id(self):
        embedder = make_embedder()
        c0 = make_chunk(chunk_id=0)
        c1 = make_chunk(chunk_id=1)
        assert embedder.generate_id(c0) != embedder.generate_id(c1)

    def test_none_metadata_returns_string(self):
        embedder = make_embedder()
        chunk = Chunk(content="text", metadata=None, chunk_id=0, chunk_strategy=None)
        result = embedder.generate_id(chunk)
        assert isinstance(result, str)
        assert "unknown" in result


# ── embed ──────────────────────────────────────────────────────────────────────

class TestEmbed:

    def test_chunks_stored_in_collection(self):
        embedder = make_embedder()
        chunks = [make_chunk(chunk_id=i, content=f"sentence {i}") for i in range(3)]
        embedder.embed(chunks)
        assert embedder.collection.count() == 3

    def test_single_openai_call_for_one_batch(self):
        embedder = make_embedder()
        chunks = [make_chunk(chunk_id=i) for i in range(5)]
        embedder.embed(chunks)
        embedder.openai.embeddings.create.assert_called_once()

    def test_multiple_openai_calls_when_batched(self):
        embedder = make_embedder()
        chunks = [make_chunk(chunk_id=i, content=f"text {i}") for i in range(10)]
        # force batch_size=3 → ceil(10/3) = 4 calls
        embedder.embed(chunks, batch_size=3)
        assert embedder.openai.embeddings.create.call_count == 4

    def test_idempotent_same_chunks_twice(self):
        embedder = make_embedder()
        chunks = [make_chunk(chunk_id=i, content=f"sentence {i}") for i in range(3)]
        embedder.embed(chunks)
        embedder.embed(chunks)
        assert embedder.collection.count() == 3

    def test_empty_chunks_no_error(self):
        embedder = make_embedder()
        embedder.embed([])
        assert embedder.collection.count() == 0

    def test_metadata_stored_in_collection(self):
        embedder = make_embedder()
        chunk = make_chunk(source="myfile.txt", chunk_id=0, strategy=ChunkingStrategy.FIXED)
        embedder.embed([chunk])
        results = embedder.collection.get(include=["metadatas"])
        meta = results["metadatas"][0]
        assert meta["source"] == "myfile.txt"
        assert meta["strategy"] == "FIXED"
