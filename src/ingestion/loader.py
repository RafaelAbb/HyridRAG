import os
from enum import Enum

from src.ingestion.loaders.pdf import PDFLoader
from src.ingestion.loaders.markdown import MarkdownLoader
from src.ingestion.loaders.html import HTMLLoader
from src.ingestion.loaders.text import TextLoader


class FileExtension(Enum):
    HTML = 1
    PDF = 2
    TXT = 3
    MD = 4


def load_file(file_name: str) -> list:
    _, extension = os.path.splitext(file_name)
    extension_name = extension.lower().lstrip('.').upper()

    try:
        matched_enum = FileExtension[extension_name]
    except KeyError:
        supported = ", ".join([e.name.lower() for e in FileExtension])
        raise ValueError(f"Unsupported extension: .{extension_name.lower()}. Supported: {supported}")

    match matched_enum:
        case FileExtension.HTML: loader = HTMLLoader()
        case FileExtension.PDF:  loader = PDFLoader()
        case FileExtension.TXT:  loader = TextLoader()
        case FileExtension.MD:   loader = MarkdownLoader()

    return loader.load(file_name)


def load_directory(directory_path: str) -> list:
    raw_documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                raw_documents.extend(load_file(file_path))
            except ValueError as e:
                print(f"Skipping {file_path}: {e}")
    return raw_documents
