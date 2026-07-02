from dataclasses import dataclass

from src.ingestion.base import DocumentMetadata


@dataclass
class RetrievalResult:
    doc_id: str
    metadata: DocumentMetadata
    text: str
    score: float