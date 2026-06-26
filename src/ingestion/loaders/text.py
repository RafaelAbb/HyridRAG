from src.ingestion.base import FileLoaderInterface, RawDocument, DocumentMetadata, Source
from src.ingestion.utils import get_mod_time


class TextLoader(FileLoaderInterface):
    '''Class model for loading Text files.'''

    def load(self, file_path: str) -> list:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            mod_time = get_mod_time(file_path)
            document_metadata = DocumentMetadata(source=Source(source_name=file_path), mod_time=mod_time)
            return [RawDocument(content=content, metadata=document_metadata)]
