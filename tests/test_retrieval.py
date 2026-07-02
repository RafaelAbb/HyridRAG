from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval.base import RetrievalResult
from src.retrieval.fusion import rrf_merge, Reranker, hybrid_retrieve, RRF_K


# ── helpers ──────────────────────────────────────────────────────────────────

def make_result(doc_id: str, score: float = 1.0, text: str = "text") -> RetrievalResult:
    return RetrievalResult(doc_id=doc_id, metadata=None, text=text, score=score)


# ── rrf_merge ─────────────────────────────────────────────────────────────────

def test_rrf_merge_overlap_scores_higher_than_single_list():
    dense   = [make_result("A"), make_result("B")]
    sparse  = [make_result("A"), make_result("C")]
    results = rrf_merge(dense, sparse)
    scores  = {r.doc_id: r.score for r in results}
    assert scores["A"] > scores["B"]
    assert scores["A"] > scores["C"]


def test_rrf_merge_rank1_scores_higher_than_rank10():
    dense = [make_result(f"doc_{i}") for i in range(12)]
    results = rrf_merge(dense, [])
    scores = {r.doc_id: r.score for r in results}
    assert scores["doc_0"] > scores["doc_10"]


def test_rrf_merge_empty_dense_returns_sparse_only():
    sparse  = [make_result("X"), make_result("Y")]
    results = rrf_merge([], sparse)
    ids = [r.doc_id for r in results]
    assert "X" in ids and "Y" in ids


def test_rrf_merge_empty_sparse_returns_dense_only():
    dense   = [make_result("A"), make_result("B")]
    results = rrf_merge(dense, [])
    ids = [r.doc_id for r in results]
    assert "A" in ids and "B" in ids


def test_rrf_merge_score_formula_single_list():
    dense_weight = 0.7
    dense   = [make_result("only")]
    results = rrf_merge(dense, [], dense_weight=dense_weight, sparse_weight=0.3)
    expected = dense_weight / (RRF_K + 1)
    assert abs(results[0].score - expected) < 1e-9


def test_rrf_merge_results_sorted_descending():
    dense  = [make_result("A"), make_result("B"), make_result("C")]
    sparse = [make_result("C"), make_result("A")]
    results = rrf_merge(dense, sparse)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_rrf_merge_both_empty_returns_empty():
    assert rrf_merge([], []) == []


# ── sparse_search ─────────────────────────────────────────────────────────────

def _make_mock_collection(ids, documents, metadatas):
    col = MagicMock()
    col.get.return_value = {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
    }
    return col


def test_sparse_search_returns_k_results():
    from src.retrieval.sparse import sparse_search

    col = _make_mock_collection(
        ids=["a", "b", "c", "d"],
        documents=["cat sat mat", "dog ran far", "cat cat cat", "bird flew away"],
        metadatas=[{"source": "f.txt"}] * 4,
    )
    results = sparse_search("cat", col, k=2)
    assert len(results) == 2


def test_sparse_search_relevant_doc_scores_higher():
    from src.retrieval.sparse import sparse_search

    col = _make_mock_collection(
        ids=["match", "noise1", "noise2", "noise3", "irrelevant"],
        documents=[
            "python retrieval augmented generation",
            "cooking pasta with tomato sauce",
            "history of ancient rome",
            "bicycle repair and maintenance guide",
            "banana smoothie recipe",
        ],
        metadatas=[{"source": f"{i}.txt"} for i in range(5)],
    )
    results = sparse_search("python retrieval", col, k=5)
    scores = {r.doc_id: r.score for r in results}
    assert scores["match"] > scores["irrelevant"]


def test_sparse_search_result_fields_populated():
    from src.retrieval.sparse import sparse_search

    col = _make_mock_collection(
        ids=["doc1"],
        documents=["hello world"],
        metadatas=[{"source": "test.txt"}],
    )
    results = sparse_search("hello", col, k=1)
    r = results[0]
    assert r.doc_id == "doc1"
    assert r.text == "hello world"
    assert isinstance(r.score, float)
    assert r.metadata.source.source_name == "test.txt"


# ── Reranker ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_reranker():
    with patch("src.retrieval.fusion.CrossEncoder") as MockCE:
        MockCE.return_value = MagicMock()
        reranker = Reranker(model_name="mock-model")
        reranker.model = MockCE.return_value
        yield reranker


def test_reranker_returns_top_k(mock_reranker):
    results = [make_result(f"doc_{i}", text=f"text {i}") for i in range(5)]
    mock_reranker.model.predict.return_value = np.array([0.1, 0.9, 0.3, 0.8, 0.5])
    top = mock_reranker.rerank("query", results, k=2)
    assert len(top) == 2


def test_reranker_sorted_descending(mock_reranker):
    results = [make_result(f"doc_{i}", text=f"text {i}") for i in range(3)]
    mock_reranker.model.predict.return_value = np.array([0.2, 0.9, 0.5])
    top = mock_reranker.rerank("query", results, k=3)
    assert top[0].doc_id == "doc_1"
    assert top[1].doc_id == "doc_2"
    assert top[2].doc_id == "doc_0"


# ── hybrid_retrieve ───────────────────────────────────────────────────────────

def test_hybrid_retrieve_graceful_degradation():
    dense_res  = [make_result("A"), make_result("B"), make_result("C")]
    sparse_res = [make_result("A"), make_result("D")]

    bad_reranker = MagicMock()
    bad_reranker.rerank.side_effect = Exception("model failed")

    with patch("src.retrieval.fusion.dense_search", return_value=dense_res), \
         patch("src.retrieval.fusion.sparse_search", return_value=sparse_res):
        results = hybrid_retrieve("query", MagicMock(), reranker=bad_reranker, k=2)

    assert len(results) == 2


def test_hybrid_retrieve_uses_reranker_when_available():
    dense_res  = [make_result("A")]
    sparse_res = [make_result("B")]
    reranked   = [make_result("A")]

    good_reranker = MagicMock()
    good_reranker.rerank.return_value = reranked

    with patch("src.retrieval.fusion.dense_search", return_value=dense_res), \
         patch("src.retrieval.fusion.sparse_search", return_value=sparse_res):
        results = hybrid_retrieve("query", MagicMock(), reranker=good_reranker, k=1)

    assert results == reranked
