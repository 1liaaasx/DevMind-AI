"""Document extraction, cleaning and chunking."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from .models import PageText, TextChunk


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class DocumentError(ValueError):
    """Raised when an uploaded document cannot provide usable text."""


def document_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph breaks."""

    normalized = unicodedata.normalize("NFKC", text).replace("\x00", "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentError("Le fichier texte utilise un encodage non pris en charge.")


def extract_document(filename: str, data: bytes) -> tuple[str, list[PageText]]:
    """Extract clean page-level text from PDF, TXT or Markdown bytes."""

    safe_name = Path(filename).name
    extension = Path(safe_name).suffix.casefold()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentError("Format non pris en charge. Utilisez PDF, TXT ou Markdown.")
    if not data:
        raise DocumentError("Le document est vide.")

    pages: list[PageText] = []
    try:
        if extension == ".pdf":
            reader = PdfReader(BytesIO(data))
            for number, page in enumerate(reader.pages, start=1):
                text = clean_text(page.extract_text() or "")
                if text:
                    pages.append(PageText(text=text, page=number))
        else:
            text = clean_text(_decode_text(data))
            if text:
                pages.append(PageText(text=text))
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError(f"Impossible de lire « {safe_name} ».") from exc

    if not pages:
        if extension == ".pdf":
            raise DocumentError(
                "Aucun texte exploitable n’a été trouvé. Les PDF scannés nécessitent une étape OCR."
            )
        raise DocumentError("Le document ne contient aucun texte exploitable.")

    return document_hash(data), pages


def _best_break(text: str, start: int, target: int, minimum: int) -> int:
    """Prefer a paragraph or sentence boundary near the target chunk size."""

    window = text[start:target]
    floor = min(len(window), minimum)
    candidates = [window.rfind("\n\n", floor), window.rfind(". ", floor), window.rfind("\n", floor)]
    best = max(candidates)
    if best < 0:
        return target
    separator_width = 2 if window[best : best + 2] in {"\n\n", ". "} else 1
    return start + best + separator_width


def split_pages(
    pages: list[PageText],
    filename: str,
    doc_hash: str,
    *,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[TextChunk]:
    """Split page text into overlapping, boundary-aware character chunks."""

    if chunk_size < 100:
        raise ValueError("chunk_size doit être supérieur ou égal à 100.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap doit être positif et inférieur à chunk_size.")

    chunks: list[TextChunk] = []
    chunk_index = 0
    safe_name = Path(filename).name
    for page in pages:
        start = 0
        while start < len(page.text):
            target = min(start + chunk_size, len(page.text))
            end = target
            if target < len(page.text):
                end = _best_break(page.text, start, target, int(chunk_size * 0.6))
            passage = page.text[start:end].strip()
            if passage:
                chunks.append(
                    TextChunk(
                        text=passage,
                        filename=safe_name,
                        document_hash=doc_hash,
                        chunk_index=chunk_index,
                        page=page.page,
                    )
                )
                chunk_index += 1
            if end >= len(page.text):
                break
            start = max(start + 1, end - overlap)
    return chunks
