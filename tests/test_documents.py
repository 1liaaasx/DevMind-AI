from __future__ import annotations

import pytest

from devmind_rag.documents import DocumentError, clean_text, extract_document, split_pages
from devmind_rag.models import PageText


def test_clean_text_normalizes_spacing_and_paragraphs() -> None:
    raw = "  Première   ligne  \r\n\r\n\r\nDeuxième\tligne  "
    assert clean_text(raw) == "Première ligne\n\nDeuxième ligne"


def test_extract_markdown_keeps_filename_and_text() -> None:
    doc_hash, pages = extract_document("guide.md", "# Guide\n\nBonjour".encode())
    assert len(doc_hash) == 64
    assert pages == [PageText(text="# Guide\n\nBonjour")]


def test_extract_pdf_preserves_page_numbers(monkeypatch) -> None:
    class FakePdfPage:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakePdfReader:
        pages = [FakePdfPage("Page une"), FakePdfPage("Page deux")]

    monkeypatch.setattr("devmind_rag.documents.PdfReader", lambda _: FakePdfReader())

    _, pages = extract_document("guide.pdf", b"%PDF-factice")

    assert pages == [PageText(text="Page une", page=1), PageText(text="Page deux", page=2)]


def test_empty_document_is_rejected() -> None:
    with pytest.raises(DocumentError, match="vide"):
        extract_document("vide.txt", b"")


def test_invalid_pdf_is_rejected_without_leaking_parser_details() -> None:
    with pytest.raises(DocumentError, match="Impossible de lire"):
        extract_document("corrompu.pdf", b"ceci n'est pas un PDF")


def test_unsupported_document_is_rejected() -> None:
    with pytest.raises(DocumentError, match="Format"):
        extract_document("image.png", b"content")


def test_split_pages_preserves_page_and_overlap() -> None:
    text = " ".join(f"mot{i}" for i in range(180))
    chunks = split_pages(
        [PageText(text=text, page=3)],
        "cours.pdf",
        "abc",
        chunk_size=180,
        overlap=30,
    )
    assert len(chunks) > 1
    assert all(chunk.page == 3 for chunk in chunks)
    assert chunks[0].filename == "cours.pdf"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
