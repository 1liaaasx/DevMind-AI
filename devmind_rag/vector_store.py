"""Thin ChromaDB layer for isolated DevMind AI session collections."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import chromadb

from .models import DocumentInfo, RetrievedChunk, TextChunk


class ChromaDocumentStore:
    """Store document chunks and metadata in one explicitly named collection."""

    def __init__(self, persist_directory: Path | str, collection_name: str = "devmind_documents") -> None:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def has_document(self, doc_hash: str) -> bool:
        result = self.collection.get(
            where={"document_hash": doc_hash},
            limit=1,
            include=["metadatas"],
        )
        return bool(result["ids"])

    def add_chunks(self, chunks: Sequence[TextChunk], embeddings: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Le nombre de passages et de vecteurs doit être identique.")

        self.collection.add(
            ids=[f"{chunk.document_hash}:{chunk.chunk_index}" for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
            metadatas=[
                {
                    "filename": chunk.filename,
                    "document_name": chunk.document_name or chunk.filename,
                    "document_url": chunk.document_url or "",
                    "document_hash": chunk.document_hash,
                    "chunk_index": chunk.chunk_index,
                    "page": chunk.page or 0,
                    "source_type": chunk.source_type,
                    "source_url": chunk.source_url or "",
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 4,
        document_hash: str | None = None,
    ) -> list[RetrievedChunk]:
        if self.collection.count() == 0:
            return []
        query_filter = {"document_hash": document_hash} if document_hash else None
        result = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(limit, self.collection.count()),
            where=query_filter,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [
            RetrievedChunk(
                text=text,
                filename=str(metadata["filename"]),
                document_hash=str(metadata["document_hash"]),
                chunk_index=int(metadata["chunk_index"]),
                page=int(metadata["page"]) or None,
                distance=float(distance),
                document_name=str(metadata.get("document_name") or metadata["filename"]),
                source_url=str(metadata.get("source_url") or "") or None,
                source_type=str(metadata.get("source_type") or "document"),
            )
            for text, metadata, distance in zip(documents, metadatas, distances, strict=True)
        ]

    def get_document_chunks(self, doc_hash: str) -> list[RetrievedChunk]:
        """Return every stored chunk for one document or repository.

        This is intentionally scoped by ``document_hash``.  The RAG service uses
        it to supplement semantic search with repository-level evidence (README,
        manifests and entry points) without ever mixing another source.
        """

        result = self.collection.get(
            where={"document_hash": doc_hash},
            include=["documents", "metadatas"],
        )
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            RetrievedChunk(
                text=str(text),
                filename=str(metadata["filename"]),
                document_hash=str(metadata["document_hash"]),
                chunk_index=int(metadata["chunk_index"]),
                page=int(metadata.get("page", 0)) or None,
                distance=None,
                document_name=str(metadata.get("document_name") or metadata["filename"]),
                source_url=str(metadata.get("source_url") or "") or None,
                source_type=str(metadata.get("source_type") or "document"),
            )
            for text, metadata in zip(documents, metadatas, strict=True)
        ]

    def list_documents(self, *, source_type: str | None = None) -> list[DocumentInfo]:
        result = self.collection.get(include=["metadatas"])
        grouped: dict[str, list[dict]] = defaultdict(list)
        for metadata in result.get("metadatas") or []:
            metadata_source_type = str(metadata.get("source_type") or "document")
            if source_type and metadata_source_type != source_type:
                continue
            grouped[str(metadata["document_hash"])].append(metadata)

        documents = []
        for doc_hash, entries in grouped.items():
            pages = {int(entry["page"]) for entry in entries if int(entry.get("page", 0)) > 0}
            documents.append(
                DocumentInfo(
                    filename=str(entries[0].get("document_name") or entries[0]["filename"]),
                    document_hash=doc_hash,
                    chunk_count=len(entries),
                    page_count=max(pages) if pages else None,
                    source_type=str(entries[0].get("source_type") or "document"),
                    source_url=str(entries[0].get("document_url") or "") or None,
                )
            )
        return sorted(documents, key=lambda document: document.filename.casefold())

    def document_hash_for_source_url(self, source_url: str) -> str | None:
        """Return the indexed hash for a repository URL, if any."""

        result = self.collection.get(
            where={"document_url": source_url},
            limit=1,
            include=["metadatas"],
        )
        metadatas = result.get("metadatas") or []
        if not metadatas:
            return None
        return str(metadatas[0]["document_hash"])

    def delete_document(self, doc_hash: str) -> None:
        self.collection.delete(where={"document_hash": doc_hash})

    def clear(self) -> None:
        """Remove all chunks from this collection without touching other sessions."""

        ids = self.collection.get(include=[]).get("ids") or []
        if ids:
            self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()
