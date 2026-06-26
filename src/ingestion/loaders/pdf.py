import pymupdf

from src.ingestion.base import FileLoaderInterface, RawDocument, DocumentMetadata, Source
from src.ingestion.utils import get_mod_time


class PDFLoader(FileLoaderInterface):
    '''Class model for loading PDF files.'''

    def load(self, file_path: str) -> list:  # TODO: Handle images in pdfs
        doc = pymupdf.open(file_path)
        mod_time = get_mod_time(file_path)
        raw_documents = []

        for i, page in enumerate(doc, start=1):
            content = page.get_text().strip()
            document_metadata = DocumentMetadata(source=Source(source_name=file_path, page_number=i), mod_time=mod_time)
            raw_documents.append(RawDocument(content=content, metadata=document_metadata))

        return raw_documents
