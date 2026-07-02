import logging
from chromadb.api.models.Collection import Collection
from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.base import RetrievalResult
from src.retrieval.dense import dense_search
from src.retrieval.sparse import sparse_search


K_MULTIPLIER = 4
RRF_K = 60


def rrf_merge(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    dense_weight: float = settings.rrf_dense_weight,
    sparse_weight: float = settings.rrf_sparse_weight,
) -> list[RetrievalResult]:

    id_dict = {}
    for result in dense_results:
        id_dict[result.doc_id] = RetrievalResult(doc_id=result.doc_id, metadata=result.metadata, text=result.text, score=0.0)

    for result in sparse_results:
        if result.doc_id not in id_dict:
            id_dict[result.doc_id] = RetrievalResult(doc_id=result.doc_id, metadata=result.metadata, text=result.text, score=0.0)
    
    
    for rank, result in enumerate(dense_results, start=1):
        id_dict[result.doc_id].score += dense_weight / (RRF_K + rank)
    
    
    for rank, result in enumerate(sparse_results, start=1):
        id_dict[result.doc_id].score += sparse_weight / (RRF_K + rank)
        
    sorted_results = sorted(id_dict.values(), key=lambda x: x.score, reverse=True)
    return sorted_results 


class Reranker:

    def __init__(self, model_name: str = settings.reranker_model):
        self.model = CrossEncoder(model_name)
        

    def rerank(self, query: str, results: list[RetrievalResult], k: int = settings.reranker_top_k) -> list[RetrievalResult]:
        
        pairs = [(query, result.text) for result in results]
        scores = self.model.predict(pairs)
        sorted_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [result for result, score in sorted_results[:k]]


def hybrid_retrieve(query: str,
                    collection: Collection,
                    reranker: Reranker = None,
                    k: int = settings.reranker_top_k
                    ) -> list[RetrievalResult]:

    dense_results = dense_search(query, collection, k * K_MULTIPLIER)
    sparse_results = sparse_search(query, collection, k * K_MULTIPLIER)
    merged_results = rrf_merge(dense_results, sparse_results)
    try:
        reranker = reranker or Reranker()
        reranked_results = reranker.rerank(query, merged_results, k)
        return reranked_results
    except Exception as e:
        logging.warning(f"Error occurred while reranking: {e}")
        return merged_results[:k]
