from dataclasses import dataclass, field
from enum import Enum

from src.retrieval.base import RetrievalResult


@dataclass
class GenerationResult:
    """
    A class to represent a generation result.
    """
    answer: str  # The generated answer
    claim_source_pairs: list[tuple[str, str]]  # List of tuples containing claim and its source
    has_answer: bool = True
    confidence: float = 0.0
    list_of_references: list[RetrievalResult] = field(default_factory=list)  # List of RetrievalResult objects
    


@dataclass
class CitationVerification:
    claim: str
    doc_id: str
    is_supported: bool
    
    
class JudgeEnum(Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INSUFFICIENT_INFO = "insufficient_info"