from openai import OpenAI
from chromadb.api.models.Collection import Collection

from src.config import settings
from src.ingestion.base import DocumentMetadata, Source
from src.retrieval.base import RetrievalResult

_openai = OpenAI(api_key=settings.openai_api_key)


def dense_search(query: str, collection: Collection, k: int) -> list[RetrievalResult]:

    response = _openai.embeddings.create(
        model=settings.embedding_model,
        input=[query],
    )
    query_vector = response.data[0].embedding

    collection_results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
    )

    return [
        RetrievalResult(
            doc_id=doc_id,
            metadata=DocumentMetadata(
                source=Source(source_name=metadata["source"])
            ) if metadata else None,
            text=text,
            score=1 - distance,
        )
        for doc_id, metadata, text, distance in zip(
            collection_results['ids'][0],
            collection_results['metadatas'][0],
            collection_results['documents'][0],
            collection_results['distances'][0]
        )
    ]