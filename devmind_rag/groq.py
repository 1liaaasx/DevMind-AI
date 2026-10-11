"""Groq text-generation adapter kept separate for easy testing."""

from __future__ import annotations

from collections import deque
from threading import Lock
from time import monotonic

from groq import APIConnectionError, APIStatusError, AuthenticationError, Groq, RateLimitError


DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_REQUEST_TIMEOUT_SECONDS = 25.0
GROQ_MAX_RETRIES = 1
GROQ_MAX_COMPLETION_TOKENS = 1800
GROQ_SESSION_REQUESTS_PER_MINUTE = 8
GROQ_PROCESS_REQUESTS_PER_MINUTE = 30


class GroqGenerationError(RuntimeError):
    """A safe, user-facing Groq generation failure."""


class SlidingWindowRateLimiter:
    """Small thread-safe limiter that keeps public demo usage bounded."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: deque[float] = deque()
        self._lock = Lock()

    def acquire(self) -> bool:
        now = monotonic()
        with self._lock:
            while self._requests and now - self._requests[0] >= self.window_seconds:
                self._requests.popleft()
            if len(self._requests) >= self.max_requests:
                return False
            self._requests.append(now)
            return True


_PROCESS_RATE_LIMITER = SlidingWindowRateLimiter(GROQ_PROCESS_REQUESTS_PER_MINUTE)


class GroqGenerator:
    """Generate text with Groq's chat-completions API."""

    def __init__(self, api_key: str, model: str = DEFAULT_GROQ_MODEL) -> None:
        # Keep the UI bounded while allowing one SDK-managed exponential-backoff
        # retry for transient connection, 408, 409, 429 and 5xx failures.
        self.client = Groq(
            api_key=api_key,
            timeout=GROQ_REQUEST_TIMEOUT_SECONDS,
            max_retries=GROQ_MAX_RETRIES,
        )
        self.model = model
        self._session_rate_limiter = SlidingWindowRateLimiter(
            GROQ_SESSION_REQUESTS_PER_MINUTE
        )

    def generate(self, prompt: str) -> str:
        if not self._session_rate_limiter.acquire() or not _PROCESS_RATE_LIMITER.acquire():
            raise GroqGenerationError(
                "La démonstration reçoit trop de demandes. Patientez une minute, puis réessayez."
            )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                reasoning_effort="low",
                max_completion_tokens=GROQ_MAX_COMPLETION_TOKENS,
            )
        except AuthenticationError as exc:
            raise GroqGenerationError(
                "La clé Groq a été refusée. Vérifiez GROQ_API_KEY, puis réessayez."
            ) from exc
        except RateLimitError as exc:
            raise GroqGenerationError(
                "La limite de requêtes Groq est atteinte. Patientez quelques instants, puis réessayez."
            ) from exc
        except APIConnectionError as exc:
            raise GroqGenerationError(
                "Groq est momentanément inaccessible. Vérifiez votre connexion, puis réessayez."
            ) from exc
        except APIStatusError as exc:
            raise GroqGenerationError(
                f"Groq a refusé la requête (erreur {exc.status_code}). Réessayez plus tard."
            ) from exc
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise GroqGenerationError("Groq n’a renvoyé aucune réponse. Réessayez.")
        if getattr(response.choices[0], "finish_reason", None) == "length":
            text += (
                "\n\n> La réponse a atteint la limite de longueur. "
                "Posez une question plus ciblée pour obtenir la suite."
            )
        return text
