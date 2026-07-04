from dataclasses import dataclass

from src.ingestion.base import DocumentMetadata


@dataclass
class RetrievalResult:
    doc_id: str
    metadata: DocumentMetadata
    text: str
    score: float
    
    __str__ = lambda self: f"RetrievalResult(doc_id={self.doc_id}, metadata={self.metadata}, text={self.text}, score={self.score})"