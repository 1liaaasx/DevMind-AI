from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from groq import APIConnectionError, AuthenticationError, RateLimitError

import devmind_rag.groq as groq_module


class FakeCompletions:
    def __init__(self, content: str | None = "Réponse Groq", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeGroqClient:
    def __init__(self, content: str | None = "Réponse Groq", error: Exception | None = None) -> None:
        self.completions = FakeCompletions(content, error)
        self.chat = SimpleNamespace(completions=self.completions)


def test_groq_generator_uses_requested_model_and_prompt(monkeypatch) -> None:
    client = FakeGroqClient()
    client_options = {}

    def fake_groq(**kwargs):
        client_options.update(kwargs)
        return client

    monkeypatch.setattr(groq_module, "Groq", fake_groq)

    generator = groq_module.GroqGenerator("secret-test")
    result = generator.generate("Explique ce code")

    assert result == "Réponse Groq"
    assert client.completions.kwargs["model"] == "openai/gpt-oss-120b"
    assert client.completions.kwargs["messages"] == [
        {"role": "user", "content": "Explique ce code"}
    ]
    assert client_options == {
        "api_key": "secret-test",
        "timeout": groq_module.GROQ_REQUEST_TIMEOUT_SECONDS,
        "max_retries": groq_module.GROQ_MAX_RETRIES,
    }
    assert client.completions.kwargs["reasoning_effort"] == "low"
    assert (
        client.completions.kwargs["max_completion_tokens"]
        == groq_module.GROQ_MAX_COMPLETION_TOKENS
    )


def test_groq_length_limit_is_reported_to_the_user(monkeypatch) -> None:
    class LengthLimitedCompletions(FakeCompletions):
        def create(self, **kwargs):
            message = SimpleNamespace(content="Réponse partielle")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="length")]
            )

    client = FakeGroqClient()
    client.completions = LengthLimitedCompletions()
    client.chat = SimpleNamespace(completions=client.completions)
    monkeypatch.setattr(groq_module, "Groq", lambda **kwargs: client)

    result = groq_module.GroqGenerator("secret-test").generate("Question")

    assert "limite de longueur" in result


def test_groq_generator_rejects_empty_response(monkeypatch) -> None:
    client = FakeGroqClient(content="")
    monkeypatch.setattr(groq_module, "Groq", lambda **kwargs: client)

    generator = groq_module.GroqGenerator("secret-test")

    with pytest.raises(groq_module.GroqGenerationError, match="aucune réponse"):
        generator.generate("Question")


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (
            AuthenticationError(
                "invalid key",
                response=httpx.Response(
                    401, request=httpx.Request("POST", "https://api.groq.com")
                ),
                body=None,
            ),
            "GROQ_API_KEY",
        ),
        (
            RateLimitError(
                "rate limit",
                response=httpx.Response(
                    429, request=httpx.Request("POST", "https://api.groq.com")
                ),
                body=None,
            ),
            "limite de requêtes",
        ),
        (
            APIConnectionError(
                request=httpx.Request("POST", "https://api.groq.com")
            ),
            "momentanément inaccessible",
        ),
    ],
)
def test_groq_failures_are_translated_to_safe_messages(
    monkeypatch, error: Exception, expected_message: str
) -> None:
    client = FakeGroqClient(error=error)
    monkeypatch.setattr(groq_module, "Groq", lambda **kwargs: client)

    generator = groq_module.GroqGenerator("secret-test")

    with pytest.raises(groq_module.GroqGenerationError, match=expected_message):
        generator.generate("Question")

