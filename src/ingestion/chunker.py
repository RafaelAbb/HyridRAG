from src.config import settings

from src.ingestion.base import ChunkerInterface, Chunk, RawDocument, ChunkingStrategy
from src.ingestion.chunkers.fixed import FixedChunker
from src.ingestion.chunkers.recursive import RecursiveChunker
from src.ingestion.chunkers.semantic import SemanticChunker


def get_chunker(strategy: ChunkingStrategy) -> ChunkerInterface:
   
    match strategy:
        case ChunkingStrategy.FIXED: return FixedChunker()
        case ChunkingStrategy.RECURSIVE: return RecursiveChunker()
        case ChunkingStrategy.SEMANTIC: return SemanticChunker()
 

def chunk_document(docs :RawDocument, strategy :ChunkingStrategy = settings.default_chunk_strategy) -> list[Chunk]:

    chunker = get_chunker(strategy)
    return chunker.chunk(docs)
    

def chunk_documents(docs : list[RawDocument], strategy :ChunkingStrategy = settings.default_chunk_strategy) -> list[Chunk]:
    chunks = []
    chunker = get_chunker(strategy)
    for doc in docs:
        chunks.extend(chunker.chunk(doc))
    return chunks