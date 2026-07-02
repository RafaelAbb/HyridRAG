from chromadb.api.models.Collection import Collection
import numpy as np
from rank_bm25 import BM25Okapi

from src.ingestion.base import DocumentMetadata, Source
from src.retrieval.base import RetrievalResult


def sparse_search(query: str, collection: Collection, k: int) -> list[RetrievalResult]:

    result = collection.get(include=["documents", "metadatas"])

    all_ids       = result["ids"]
    all_texts     = result["documents"]
    all_metadatas = result["metadatas"]

    tokenized_corpus = [doc.lower().split() for doc in all_texts]
    tokenized_query  = query.lower().split()

    bm25   = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:k]

    return [
        RetrievalResult(
            doc_id=all_ids[i],
            metadata=DocumentMetadata(
                source=Source(source_name=all_metadatas[i]["source"])
            ) if all_metadatas[i] else None,
            text=all_texts[i],
            score=float(scores[i]),
        )
        for i in top_indices
    ]


    