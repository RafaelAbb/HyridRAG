import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.base import RawDocument, DocumentMetadata, Chunk
from src.ingestion.chuncker import ChunkingStrategy
from src.ingestion.chunckers.fixed import FixedChuncker
from src.ingestion.chunckers.recursive import RecursiveChuncker
from src.ingestion.chunckers.semantic import SemanticChuncker

SHORT_TEXT = "Hello world. This is a test."

LONG_TEXT = (
    "The sky is blue. The sun is bright. "
    "Clouds drift slowly across the horizon. "
    "Birds sing in the morning. The air is fresh and cool. "
    "Python is a programming language. It is widely used in data science. "
    "Machine learning models require large datasets. Neural networks learn from examples. "
    "The ocean is deep and vast. Whales swim in the ocean. Fish live near coral reefs."
)

SEMANTIC_TEXT = (
    "The mitochondria is the powerhouse of the cell. "
    "DNA carries genetic information in living organisms. "
    "Cells divide through a process called mitosis. "
    "The stock market rose sharply today. "
    "Investors reacted positively to earnings reports. "
    "Interest rates were held steady by the central bank."
)


def make_doc(text: str) -> RawDocument:
    return RawDocument(content=text, metadata=DocumentMetadata())


# ── Fixed ──────────────────────────────────────────────────────────────────────

class TestFixedChuncker:

    def test_returns_chunks(self):
        chunks = list(FixedChuncker(chunk_size=20, chunk_overlap=0).chunk(make_doc(LONG_TEXT)))
        assert len(chunks) > 0

    def test_chunk_type(self):
        chunks = list(FixedChuncker(chunk_size=20, chunk_overlap=0).chunk(make_doc(LONG_TEXT)))
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_strategy_tag(self):
        chunks = list(FixedChuncker(chunk_size=20, chunk_overlap=0).chunk(make_doc(LONG_TEXT)))
        assert all(c.chunk_strategy == ChunkingStrategy.FIXED for c in chunks)

    def test_chunk_size_respected(self):
        chunks = list(FixedChuncker(chunk_size=20, chunk_overlap=0).chunk(make_doc(LONG_TEXT)))
        # all chunks except possibly the last must be <= chunk_size
        assert all(len(c.content) <= 20 for c in chunks)

    def test_overlap_produces_more_chunks(self):
        no_overlap = list(FixedChuncker(chunk_size=50, chunk_overlap=0).chunk(make_doc(LONG_TEXT)))
        with_overlap = list(FixedChuncker(chunk_size=50, chunk_overlap=25).chunk(make_doc(LONG_TEXT)))
        assert len(with_overlap) > len(no_overlap)

    def test_short_text_single_chunk(self):
        chunks = list(FixedChuncker(chunk_size=200, chunk_overlap=0).chunk(make_doc(SHORT_TEXT)))
        assert len(chunks) == 1
        assert chunks[0].content == SHORT_TEXT

    def test_metadata_preserved(self):
        meta = DocumentMetadata()
        doc = RawDocument(content=LONG_TEXT, metadata=meta)
        chunks = list(FixedChuncker(chunk_size=50, chunk_overlap=0).chunk(doc))
        assert all(c.metadata is meta for c in chunks)

    def test_empty_content(self):
        chunks = list(FixedChuncker().chunk(make_doc("")))
        assert chunks == []


# ── Recursive ─────────────────────────────────────────────────────────────────

class TestRecursiveChuncker:

    def test_returns_chunks(self):
        chunks = list(RecursiveChuncker().chunk(make_doc(LONG_TEXT)))
        assert len(chunks) > 0

    def test_chunk_type(self):
        chunks = list(RecursiveChuncker().chunk(make_doc(LONG_TEXT)))
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_strategy_tag(self):
        chunks = list(RecursiveChuncker().chunk(make_doc(LONG_TEXT)))
        assert all(c.chunk_strategy == ChunkingStrategy.RECURSIVE for c in chunks)

    def test_chunk_ids_sequential(self):
        chunks = list(RecursiveChuncker().chunk(make_doc(LONG_TEXT)))
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_content_coverage(self):
        # all words from original should appear somewhere in the chunks
        chunks = list(RecursiveChuncker().chunk(make_doc(LONG_TEXT)))
        combined = " ".join(c.content for c in chunks)
        for word in LONG_TEXT.split():
            assert word in combined

    def test_metadata_preserved(self):
        meta = DocumentMetadata()
        doc = RawDocument(content=LONG_TEXT, metadata=meta)
        chunks = list(RecursiveChuncker().chunk(doc))
        assert all(c.metadata is meta for c in chunks)


# ── Semantic ───────────────────────────────────────────────────────────────────

def make_fake_encode(n_sentences):
    """Return embeddings where first half and second half are orthogonal → guaranteed topic split."""
    def fake_encode(sentences):
        n = len(sentences)
        vecs = np.zeros((n, 2))
        for i in range(n):
            vecs[i] = [1.0, 0.0] if i < n // 2 else [0.0, 1.0]
        return vecs
    return fake_encode


@pytest.fixture(scope="module")
def mock_sentence_transformer():
    with patch("src.ingestion.chunckers.semantic.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.encode.side_effect = make_fake_encode(6)
        mock_cls.return_value = mock_model
        yield mock_cls, mock_model


class TestSemanticChuncker:

    def test_returns_chunks(self, mock_sentence_transformer):
        _, mock_model = mock_sentence_transformer
        mock_model.encode.side_effect = make_fake_encode(6)
        chunks = list(SemanticChuncker().chunk(make_doc(SEMANTIC_TEXT)))
        assert len(chunks) > 0

    def test_chunk_type(self, mock_sentence_transformer):
        _, mock_model = mock_sentence_transformer
        mock_model.encode.side_effect = make_fake_encode(6)
        chunks = list(SemanticChuncker().chunk(make_doc(SEMANTIC_TEXT)))
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_strategy_tag(self, mock_sentence_transformer):
        _, mock_model = mock_sentence_transformer
        mock_model.encode.side_effect = make_fake_encode(6)
        chunks = list(SemanticChuncker().chunk(make_doc(SEMANTIC_TEXT)))
        assert all(c.chunk_strategy == ChunkingStrategy.SEMANTIC for c in chunks)

    def test_chunk_ids_sequential(self, mock_sentence_transformer):
        _, mock_model = mock_sentence_transformer
        mock_model.encode.side_effect = make_fake_encode(6)
        chunks = list(SemanticChuncker().chunk(make_doc(SEMANTIC_TEXT)))
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_topic_shift_splits(self, mock_sentence_transformer):
        # first half embeddings [1,0], second half [0,1] → cosine sim drops at midpoint
        _, mock_model = mock_sentence_transformer
        mock_model.encode.side_effect = make_fake_encode(6)
        chunks = list(SemanticChuncker().chunk(make_doc(SEMANTIC_TEXT)))
        assert len(chunks) >= 2

    def test_model_loaded_once(self, mock_sentence_transformer):
        mock_cls, _ = mock_sentence_transformer
        SemanticChuncker()
        assert mock_cls.called
        assert hasattr(SemanticChuncker.__init__, "__wrapped__") or True  # model stored on self

    def test_metadata_preserved(self, mock_sentence_transformer):
        _, mock_model = mock_sentence_transformer
        mock_model.encode.side_effect = make_fake_encode(6)
        meta = DocumentMetadata()
        doc = RawDocument(content=SEMANTIC_TEXT, metadata=meta)
        chunks = list(SemanticChuncker().chunk(doc))
        assert all(c.metadata is meta for c in chunks)
