from typing import List

import chromadb
from openai import OpenAI

from src.config import settings
from src.ingestion.base import Chunk


class Embedder:
    def __init__(self, chroma_client=None, openai_client=None, collection_name=None):
        self.openai = openai_client or OpenAI(api_key=settings.openai_api_key)

        client = chroma_client or chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        self.collection = client.get_or_create_collection(
            name=collection_name or settings.chroma_collection_name
        )


    def generate_id(self, chunk: Chunk) -> str:

        source_name = chunk.metadata.source.source_name if chunk.metadata and chunk.metadata.source else "unknown"
        strategy = chunk.chunk_strategy.name if chunk.chunk_strategy else "unknown"
        src = chunk.metadata.source if chunk.metadata else None
        page = src.page_number if src and src.page_number is not None else "unknown"
        section = src.section if src and src.section is not None else "unknown"
        return f"{source_name}_{page}_{section}_{chunk.chunk_id}_{strategy}"


    def embed(self, chunks: List[Chunk], batch_size: int = None) -> None:
        # Process in batches — never one HTTP call per chunk.
        batch_size = batch_size or settings.embedding_batch_size
        
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            ids = [self.generate_id(c) for c in batch]
            texts = [c.content for c in batch]
            metadatas = [
                {
                    "source": c.metadata.source.source_name if c.metadata and c.metadata.source else "",
                    "strategy": c.chunk_strategy.name if c.chunk_strategy else "",
                    "chunk_id": c.chunk_id or 0,
                }
                for c in batch
            ]

            # One API call for the whole batch
            response = self.openai.embeddings.create(
                model=settings.embedding_model,
                input=texts,
            )
            vectors = [item.embedding for item in response.data]

            # upsert = insert if new, overwrite if ID exists → idempotent
            self.collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )
