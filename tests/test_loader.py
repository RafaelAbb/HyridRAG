import pathlib
import pytest

from src.ingestion.base import RawDocument, DocumentMetadata
from src.ingestion.loader import load_file, load_directory

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"

TXT_PATH  = str(FIXTURES_DIR / "sample.txt")
MD_PATH   = str(FIXTURES_DIR / "sample.md")
HTML_PATH = str(FIXTURES_DIR / "sample.html")
PDF_PATH  = str(FIXTURES_DIR / "sample.pdf")


# ── TextLoader ─────────────────────────────────────────────────────────────────

class TestTextLoader:

    def test_returns_list(self):
        result = load_file(TXT_PATH)
        assert isinstance(result, list)

    def test_single_document(self):
        result = load_file(TXT_PATH)
        assert len(result) == 1

    def test_document_type(self):
        result = load_file(TXT_PATH)
        assert isinstance(result[0], RawDocument)

    def test_content_not_empty(self):
        result = load_file(TXT_PATH)
        assert result[0].content.strip() != ""

    def test_content_matches_file(self):
        result = load_file(TXT_PATH)
        assert "Hello world" in result[0].content

    def test_source_name_set(self):
        result = load_file(TXT_PATH)
        assert result[0].metadata.source.source_name == TXT_PATH

    def test_mod_time_set(self):
        result = load_file(TXT_PATH)
        assert result[0].metadata.mod_time is not None


# ── MarkdownLoader ─────────────────────────────────────────────────────────────

class TestMarkdownLoader:

    def test_returns_list(self):
        result = load_file(MD_PATH)
        assert isinstance(result, list)

    def test_splits_by_heading(self):
        # sample.md has 3 headings → 3 documents
        result = load_file(MD_PATH)
        assert len(result) == 3

    def test_document_type(self):
        result = load_file(MD_PATH)
        assert all(isinstance(d, RawDocument) for d in result)

    def test_section_metadata_set(self):
        result = load_file(MD_PATH)
        assert all(d.metadata.source.section is not None for d in result)

    def test_section_contains_heading_text(self):
        result = load_file(MD_PATH)
        sections = [d.metadata.source.section for d in result]
        assert any("Introduction" in s for s in sections)
        assert any("Methods" in s for s in sections)
        assert any("Results" in s for s in sections)

    def test_content_not_empty(self):
        result = load_file(MD_PATH)
        assert all(d.content.strip() != "" for d in result)

    def test_source_name_set(self):
        result = load_file(MD_PATH)
        assert all(d.metadata.source.source_name == MD_PATH for d in result)

    def test_mod_time_set(self):
        result = load_file(MD_PATH)
        assert all(d.metadata.mod_time is not None for d in result)


# ── HTMLLoader ─────────────────────────────────────────────────────────────────

class TestHTMLLoader:

    def test_returns_list(self):
        result = load_file(HTML_PATH)
        assert isinstance(result, list)

    def test_single_document(self):
        result = load_file(HTML_PATH)
        assert len(result) == 1

    def test_tags_stripped(self):
        result = load_file(HTML_PATH)
        assert "<" not in result[0].content
        assert ">" not in result[0].content

    def test_text_content_preserved(self):
        result = load_file(HTML_PATH)
        assert "Hello World" in result[0].content
        assert "sample paragraph" in result[0].content

    def test_source_name_set(self):
        result = load_file(HTML_PATH)
        assert result[0].metadata.source.source_name == HTML_PATH

    def test_mod_time_set(self):
        result = load_file(HTML_PATH)
        assert result[0].metadata.mod_time is not None


# ── PDFLoader ──────────────────────────────────────────────────────────────────

class TestPDFLoader:

    def test_returns_list(self):
        result = load_file(PDF_PATH)
        assert isinstance(result, list)

    def test_one_document_per_page(self):
        # sample.pdf has 2 pages
        result = load_file(PDF_PATH)
        assert len(result) == 2

    def test_document_type(self):
        result = load_file(PDF_PATH)
        assert all(isinstance(d, RawDocument) for d in result)

    def test_content_not_empty(self):
        result = load_file(PDF_PATH)
        assert all(d.content.strip() != "" for d in result)

    def test_source_name_set(self):
        result = load_file(PDF_PATH)
        assert all(d.metadata.source.source_name == PDF_PATH for d in result)

    def test_mod_time_set(self):
        result = load_file(PDF_PATH)
        assert all(d.metadata.mod_time is not None for d in result)

    def test_page_number_stored(self):
        result = load_file(PDF_PATH)
        pages = [d.metadata.source.page_number for d in result]
        assert pages == [1, 2]


# ── load_file dispatch ─────────────────────────────────────────────────────────

class TestLoadFile:

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported extension"):
            load_file("document.xyz")

    def test_txt_routed_correctly(self):
        result = load_file(TXT_PATH)
        assert len(result) == 1

    def test_md_routed_correctly(self):
        result = load_file(MD_PATH)
        assert len(result) > 0

    def test_html_routed_correctly(self):
        result = load_file(HTML_PATH)
        assert len(result) == 1

    def test_pdf_routed_correctly(self):
        result = load_file(PDF_PATH)
        assert len(result) == 2


# ── load_directory ─────────────────────────────────────────────────────────────

class TestLoadDirectory:

    def test_loads_all_supported_files(self):
        result = load_directory(str(FIXTURES_DIR))
        assert len(result) > 0

    def test_returns_list_of_raw_documents(self):
        result = load_directory(str(FIXTURES_DIR))
        assert all(isinstance(d, RawDocument) for d in result)

    def test_skips_unsupported_files(self, tmp_path):
        (tmp_path / "valid.txt").write_text("hello")
        (tmp_path / "invalid.csv").write_text("a,b,c")
        result = load_directory(str(tmp_path))
        assert len(result) == 1
        assert "hello" in result[0].content

    def test_empty_directory(self, tmp_path):
        result = load_directory(str(tmp_path))
        assert result == []
