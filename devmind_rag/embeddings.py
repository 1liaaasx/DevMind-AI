"""Local sentence-transformer embeddings with lazy model loading."""

from __future__ import annotations

import logging
from threading import Lock, Thread
from typing import Protocol, Sequence


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOGGER = logging.getLogger("devmind.embeddings")
_MODEL_CACHE: dict[str, object] = {}
_MODEL_CACHE_LOCK = Lock()


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedding:
    """A lightweight multilingual local embedding provider."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._warmup_lock = Lock()
        self._warmup_thread: Thread | None = None

    @property
    def is_ready(self) -> bool:
        """Return whether the local model has already completed its first load."""

        return self._model is not None or self.model_name in _MODEL_CACHE

    @property
    def model(self):
        if self._model is None:
            # Reuse one model across storage directories and Streamlit sessions.
            # The global lock prevents concurrent cold starts from loading several
            # copies and making the first interaction appear frozen.
            with _MODEL_CACHE_LOCK:
                cached = _MODEL_CACHE.get(self.model_name)
                if cached is None:
                    from sentence_transformers import SentenceTransformer

                    cached = SentenceTransformer(self.model_name)
                    _MODEL_CACHE[self.model_name] = cached
                self._model = cached
        return self._model

    def warm_up_async(self) -> None:
        """Start the expensive model load without delaying initial UI rendering."""

        if self.is_ready:
            return
        with self._warmup_lock:
            if self._warmup_thread is not None and self._warmup_thread.is_alive():
                return

            def load() -> None:
                try:
                    _ = self.model
                except Exception:
                    # A real query will retry and expose its normal user-facing
                    # error; the background failure is only useful in server logs.
                    LOGGER.exception("Background embedding model warm-up failed")

            self._warmup_thread = Thread(
                target=load,
                name="devmind-embedding-warmup",
                daemon=True,
            )
            self._warmup_thread.start()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()
