from chromadb.api.models.Collection import Collection
import numpy as np
from rank_bm25 import BM25Okapi

from src.ingestion.base import DocumentMetadata, Source
from src.retrieval.base import RetrievalResult

# Common English stopwords carry near-zero signal but appear disproportionately
# often in short chunks — left unfiltered they let BM25's length normalization
# rank a short, off-topic chunk ("what"/"are" a few times in 10 tokens) above
# a long, genuinely relevant one. Small fixed list, no extra dependency.
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "by", "from", "as",
    "and", "or", "but", "if", "so", "that", "this", "these", "those",
    "it", "its", "do", "does", "did", "can", "could", "will", "would",
    "what", "which", "who", "whom", "how", "when", "where", "why",
})


def _tokenize(text: str) -> list[str]:
    return [tok for tok in text.lower().split() if tok not in STOPWORDS]


def sparse_search(query: str, collection: Collection, k: int) -> list[RetrievalResult]:

    result = collection.get(include=["documents", "metadatas"])

    all_ids       = result["ids"]
    all_texts     = result["documents"]
    all_metadatas = result["metadatas"]

    if not all_texts:
        return []

    tokenized_corpus = [_tokenize(doc) for doc in all_texts]
    tokenized_query  = _tokenize(query)

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


    