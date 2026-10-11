"""Small data objects shared by the RAG components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    """Clean text extracted from one logical page or text document."""

    text: str
    page: int | None = None


@dataclass(frozen=True)
class TextChunk:
    """A passage ready to be embedded and stored."""

    text: str
    filename: str
    document_hash: str
    chunk_index: int
    page: int | None = None
    document_name: str | None = None
    document_url: str | None = None
    source_url: str | None = None
    source_type: str = "document"


@dataclass(frozen=True)
class RetrievedChunk:
    """A passage returned by semantic search."""

    text: str
    filename: str
    document_hash: str
    chunk_index: int
    page: int | None = None
    distance: float | None = None
    document_name: str | None = None
    source_url: str | None = None
    source_type: str = "document"

    def as_source(self) -> dict[str, str | int | None]:
        return {
            "name": self.document_name or self.filename,
            "path": self.filename,
            "page": self.page,
            "text": self.text,
            "kind": self.filename.rsplit(".", 1)[-1].upper(),
            "url": self.source_url,
            "source_type": self.source_type,
        }


@dataclass(frozen=True)
class DocumentInfo:
    """User-facing summary of one indexed document."""

    filename: str
    document_hash: str
    chunk_count: int
    page_count: int | None = None
    source_type: str = "document"
    source_url: str | None = None


@dataclass(frozen=True)
class IndexResult:
    """Outcome of one upload attempt."""

    document: DocumentInfo
    already_indexed: bool


@dataclass(frozen=True)
class AnswerResult:
    """Grounded answer plus the exact passages used to generate it."""

    answer: str
    sources: list[RetrievedChunk]
