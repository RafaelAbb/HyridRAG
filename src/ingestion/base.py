from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterator, List


class ChunkingStrategy(Enum):
    FIXED = 1
    RECURSIVE = 2
    SEMANTIC = 3


class FileLoaderInterface(ABC):
    '''interface for file loaders.'''

    @abstractmethod
    def load(self, file_path: str) -> list:
        pass


class ChunckerInterface(ABC):
    '''interface for file chunkers.'''

    @abstractmethod
    def chunk(self, raw_document: 'RawDocument') -> 'Iterator[Chunk]':
        pass

@dataclass
class RawDocument:
    '''Class model for a raw document.'''
    content: str
    metadata: 'DocumentMetadata' = None


@dataclass
class DocumentMetadata:
    '''Class model for metadata of a raw document.'''
    source: 'Source' = None
    mod_time: datetime = None


@dataclass
class Source:
    '''Class model for a source of raw documents.'''
    source_name: str
    author: str = None
    description: str = None
    page_number: int = None
    section: str = None

@dataclass   
class Chunk:
    '''Class model for a chunk of a raw document.'''
    content: str
    metadata: 'DocumentMetadata' = None
    chunk_id: int = None
    chunk_strategy: ChunkingStrategy = None
        