from abc import ABC, abstractmethod
from datetime import datetime


class FileLoaderInterface(ABC):
    '''interface for file loaders.'''

    @abstractmethod
    def load(self, file_path: str) -> list:
        pass


class RawDocument:
    '''Class model for a raw document.'''
    def __init__(self, content: str, metadata: 'DocumentMetadata' = None):
        self.content = content
        self.metadata = metadata


class DocumentMetadata:
    '''Class model for metadata of a raw document.'''
    def __init__(self, source: 'Source' = None, mod_time: datetime = None):
        self.source = source
        self.mod_time = mod_time


class Source:
    '''Class model for a source of raw documents.'''
    def __init__(self, source_name: str, author: str = None, description: str = None, page_count: int = None, section: str = None):
        self.source_name = source_name
        self.description = description
        self.page_count = page_count
        self.section = section
        self.author = author
