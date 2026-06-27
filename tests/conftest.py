import pathlib
import pytest
import pymupdf

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"
PDF_PATH = FIXTURES_DIR / "sample.pdf"


@pytest.fixture(scope="session", autouse=True)
def create_sample_pdf():
    """Generate a two-page PDF fixture once per test session."""
    doc = pymupdf.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "Page one content.\nThis is the first page of the sample PDF.")

    page2 = doc.new_page()
    page2.insert_text((72, 72), "Page two content.\nThis is the second page of the sample PDF.")

    doc.save(str(PDF_PATH))
    doc.close()

    yield
