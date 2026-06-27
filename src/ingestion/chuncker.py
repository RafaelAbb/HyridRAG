from enum import Enum
from typing import List

from src.ingestion.base import ChunckerInterface, RawDocument
from src.ingestion.chunckers.fixed import FixedChuncker
from src.ingestion.chunckers.recursive import RecursiveChuncker
from src.ingestion.chunckers.semantic import SemanticChuncker


class ChunkingStrategy(Enum):
    FIXED = 1
    RECURSIVE = 2
    SEMANTIC = 3


def GetChuncker(strategy: ChunkingStrategy) -> ChunckerInterface:
   
    match strategy:
        case ChunkingStrategy.FIXED: return FixedChuncker()
        case ChunkingStrategy.RECURSIVE: return RecursiveChuncker()
        case ChunkingStrategy.SEMANTIC: return SemanticChuncker()
 

def chunk_document(docs = RawDocument, strategy=ChunkingStrategy):

    chuncker = GetChuncker(strategy)
    return chuncker.chunk(docs)
    
    
    
    


def chunk_documents(docs = List[RawDocument], strategy=ChunkingStrategy):
    chunks = []
    for doc in docs:
        chuncker = GetChuncker(strategy)
        chunks.extend(chuncker.chunk(doc))
    return chunks