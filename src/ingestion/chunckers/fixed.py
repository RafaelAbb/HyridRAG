from typing import List

from src.ingestion.base import ChunckerInterface, Chunk, RawDocument
from src.ingestion.chuncker import ChunkingStrategy

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

class FixedChuncker(ChunckerInterface):
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, raw_document: RawDocument) -> List[Chunk]:
        metadata = raw_document.metadata
        
        for i in range(0, len(raw_document.content), self.chunk_size - self.chunk_overlap):
            chunk_content = raw_document.content[i:i + self.chunk_size]
            chunk_metadata = metadata
            chunk_id  = i
            chunk_strategy = ChunkingStrategy.FIXED
            yield Chunk(content=chunk_content, metadata=chunk_metadata, chunk_id=chunk_id, chunk_strategy=chunk_strategy)