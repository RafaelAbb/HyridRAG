import re

from src.ingestion.base import FileLoaderInterface, RawDocument, DocumentMetadata, Source
from src.ingestion.utils import get_mod_time


class MarkdownLoader(FileLoaderInterface):
    '''Class model for loading Markdown files.'''

    def load(self, file_path: str) -> list:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            mod_time = get_mod_time(file_path)

            parts = re.split(r'(?m)(^#{1,3} )', content)
            raw_documents = []

            for i in range(1, len(parts) - 1, 2):
                level = parts[i].strip()
                body = parts[i + 1]
                heading_line = body.splitlines()[0].strip()
                full_heading = f"{level} {heading_line}"
                text = body.strip()

                if text:
                    section_metadata = DocumentMetadata(
                        source=Source(source_name=file_path, section=full_heading),
                        mod_time=mod_time
                    )
                    raw_documents.append(RawDocument(content=text, metadata=section_metadata))

            return raw_documents
