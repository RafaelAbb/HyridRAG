from typing import Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.base import ChunckerInterface, Chunk, ChunkingStrategy

CHUNK_SIZE = 100
CHUNK_OVERLAP = 0

class RecursiveChuncker(ChunckerInterface):
    
    
    def chunk(self, raw_document, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> Iterator[Chunk]:
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_text(raw_document.content)
        for i, text in enumerate(texts):
            chunk_metadata = raw_document.metadata
            chunk_id  = i
            chunk_strategy = ChunkingStrategy.RECURSIVE
            yield Chunk(content=text, metadata=chunk_metadata, chunk_id=chunk_id, chunk_strategy=chunk_strategy)