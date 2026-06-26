import re

from src.ingestion.base import FileLoaderInterface, RawDocument, DocumentMetadata, Source
from src.ingestion.utils import get_mod_time


class HTMLLoader(FileLoaderInterface):
    '''Class model for loading HTML files.'''

    def load(self, file_path: str) -> list:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            mod_time = get_mod_time(file_path)

            text = re.sub(r'<[^>]+>', ' ', content)
            text = re.sub(r'\s+', ' ', text).strip()

            document_metadata = DocumentMetadata(
                source=Source(source_name=file_path),
                mod_time=mod_time
            )
            return [RawDocument(content=text, metadata=document_metadata)]
