from typing import List

from src.ingestion.base import ChunckerInterface, RawDocument, ChunkingStrategy
from src.ingestion.chunckers.fixed import FixedChuncker
from src.ingestion.chunckers.recursive import RecursiveChuncker
from src.ingestion.chunckers.semantic import SemanticChuncker


def get_chuncker(strategy: ChunkingStrategy) -> ChunckerInterface:
   
    match strategy:
        case ChunkingStrategy.FIXED: return FixedChuncker()
        case ChunkingStrategy.RECURSIVE: return RecursiveChuncker()
        case ChunkingStrategy.SEMANTIC: return SemanticChuncker()
 

def chunk_document(docs :RawDocument, strategy :ChunkingStrategy):

    chuncker = get_chuncker(strategy)
    return chuncker.chunk(docs)
    

def chunk_documents(docs : List[RawDocument], strategy :ChunkingStrategy):
    chunks = []
    chuncker = get_chuncker(strategy)
    for doc in docs:
        chunks.extend(chuncker.chunk(doc))
    return chunks