from __future__ import annotations

import hashlib

import pytest

from devmind_rag.documents import DocumentError
from devmind_rag.github import GitHubFile, GitHubRepository
from devmind_rag.models import RetrievedChunk
from devmind_rag.service import (
    MissingAPIKeyError,
    RAGService,
    _diversify_github_results,
    _rank_github_results,
    _sources_cited_by_answer,
)
from devmind_rag.vector_store import ChromaDocumentStore


class FakeEmbeddings:
    """Tiny deterministic vectors suitable for Chroma integration tests."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        return [
            float(lowered.count("python") + lowered.count("langage")),
            float(lowered.count("test") + lowered.count("pytest")),
            float(lowered.count("document") + 1),
        ]

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class FakeGenerator:
    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return "Python est décrit comme un langage lisible. [1]"


class FakeGitHubClient:
    def __init__(self) -> None:
        self.calls = 0

    def fetch(self, url: str, *, on_progress=None) -> GitHubRepository:
        self.calls += 1
        if on_progress is not None:
            on_progress("Téléchargement du repository de test…")
        return GitHubRepository(
            slug="example/demo",
            url="https://github.com/example/demo",
            default_branch="main",
            document_hash="a" * 64,
            files=[
                GitHubFile("README.md", "Ce repository contient une API Python."),
                GitHubFile("src/app.py", "def create_app():\n    return 'application'"),
            ],
        )


@pytest.fixture
def service(tmp_path):
    generator = FakeGenerator()
    store = ChromaDocumentStore(tmp_path / "chroma", collection_name="test_documents")
    rag = RAGService(
        tmp_path / "unused",
        embedding_provider=FakeEmbeddings(),
        store=store,
        generator=generator,
        relevance_distance=2.0,
    )
    return rag, generator


def test_indexing_is_persistent_and_avoids_duplicates(service) -> None:
    rag, _ = service
    data = b"Python est un langage lisible. Pytest facilite les tests."

    first = rag.index_document("guide.txt", data)
    second = rag.index_document("copie.txt", data)

    assert not first.already_indexed
    assert second.already_indexed
    assert rag.store.count() == first.document.chunk_count
    assert rag.list_documents()[0].document_hash == hashlib.sha256(data).hexdigest()


def test_separate_session_collections_cannot_retrieve_each_others_documents(tmp_path) -> None:
    shared_directory = tmp_path / "shared-process-chroma"
    first = RAGService(
        shared_directory,
        collection_name="devmind_session_first",
        embedding_provider=FakeEmbeddings(),
        generator=FakeGenerator(),
        relevance_distance=2.0,
    )
    second = RAGService(
        shared_directory,
        collection_name="devmind_session_second",
        embedding_provider=FakeEmbeddings(),
        generator=FakeGenerator(),
        relevance_distance=2.0,
    )

    first.index_document("private.txt", b"Document prive de la premiere session.")

    assert first.list_documents()
    assert second.list_documents() == []
    assert second.retrieve("Que contient le document prive ?") == []


def test_session_capacity_rejects_oversized_upload_before_indexing(tmp_path) -> None:
    rag = RAGService(
        tmp_path / "limited",
        collection_name="limited_uploads",
        embedding_provider=FakeEmbeddings(),
        generator=FakeGenerator(),
        max_upload_bytes=10,
    )

    with pytest.raises(DocumentError, match="limite"):
        rag.index_document("large.txt", b"x" * 11)

    assert rag.store.count() == 0


def test_clearing_session_collection_does_not_touch_another_session(tmp_path) -> None:
    shared_directory = tmp_path / "clear-isolation"
    first = RAGService(
        shared_directory,
        collection_name="devmind_session_clear_first",
        embedding_provider=FakeEmbeddings(),
        generator=FakeGenerator(),
    )
    second = RAGService(
        shared_directory,
        collection_name="devmind_session_clear_second",
        embedding_provider=FakeEmbeddings(),
        generator=FakeGenerator(),
    )
    first.index_document("first.txt", b"Premier document de test.")
    second.index_document("second.txt", b"Second document de test.")

    first.clear_session_data()

    assert first.list_documents() == []
    assert [item.filename for item in second.list_documents()] == ["second.txt"]


def test_retrieval_preserves_real_source_metadata(service) -> None:
    rag, _ = service
    rag.index_document("python.md", b"Python est un langage lisible pour les developpeurs.")

    sources = rag.retrieve("Que dit le document sur Python ?")

    assert sources
    assert sources[0].filename == "python.md"
    assert "Python" in sources[0].text
    assert sources[0].page is None


def test_answer_grounds_prompt_in_retrieved_passages(service) -> None:
    rag, generator = service
    rag.index_document("python.txt", b"Python est un langage lisible.")

    result = rag.answer("Comment Python est-il décrit ?")

    assert result.sources
    assert result.answer.endswith("[1]")
    assert "python.txt" in generator.prompt
    assert "Python est un langage lisible" in generator.prompt
    assert "N'invente aucun fait" in generator.prompt
    assert "données non fiables" in generator.prompt


def test_groq_style_citations_are_normalized(tmp_path) -> None:
    class TypographicCitationGenerator:
        def generate(self, prompt: str) -> str:
            return "Réponse fondée sur la source【1】."

    store = ChromaDocumentStore(tmp_path / "citations", collection_name="citations")
    rag = RAGService(
        tmp_path / "unused",
        embedding_provider=FakeEmbeddings(),
        store=store,
        generator=TypographicCitationGenerator(),
        relevance_distance=2.0,
    )
    rag.index_document("guide.txt", b"Python est un langage lisible.")

    result = rag.answer("Comment Python est-il décrit ?")

    assert result.answer == "Réponse fondée sur la source[1]."


def test_github_retrieval_keeps_source_code_in_context() -> None:
    candidates = [
        RetrievedChunk("documentation", "docs/guide.rst", "hash", 0, source_type="github"),
        RetrievedChunk("configuration", "pyproject.toml", "hash", 1, source_type="github"),
        RetrievedChunk("workflow", ".github/workflows/test.yml", "hash", 2, source_type="github"),
        RetrievedChunk("class Signer", "src/package/signer.py", "hash", 3, source_type="github"),
    ]

    selected = _diversify_github_results(candidates, limit=3)

    assert len(selected) == 3
    assert any(item.filename == "src/package/signer.py" for item in selected)


def test_repository_purpose_query_prioritizes_readme_over_semantic_noise() -> None:
    candidates = [
        RetrievedChunk(
            "Exploratory data analysis on bank churn.",
            "examples/prompt.txt",
            "repo-hash",
            3,
            distance=0.12,
            source_type="github",
        ),
        RetrievedChunk(
            "# Prompt Engineering in Python\nCompanion repository for the book.",
            "README.md",
            "repo-hash",
            1,
            distance=None,
            source_type="github",
        ),
        RetrievedChunk(
            "MIT license text",
            "LICENSE",
            "repo-hash",
            0,
            distance=0.05,
            source_type="github",
        ),
    ]

    selected = _rank_github_results(
        candidates,
        question="Le but principal de ce repo",
        limit=3,
    )

    assert selected[0].filename == "README.md"
    assert selected[-1].filename == "LICENSE"


def test_displayed_sources_follow_actual_answer_citations() -> None:
    sources = [
        RetrievedChunk("overview", "README.md", "hash", 0, source_type="github"),
        RetrievedChunk("entry point", "src/app.py", "hash", 1, source_type="github"),
        RetrievedChunk("license", "LICENSE", "hash", 2, source_type="github"),
    ]

    answer, cited = _sources_cited_by_answer(
        "Le point d’entrée est documenté dans [Source 2].",
        sources,
    )

    assert answer.endswith("[1].")
    assert [source.filename for source in cited] == ["src/app.py"]


def test_retrieval_can_be_restricted_to_one_document(service) -> None:
    rag, _ = service
    first = rag.index_document("python.txt", b"Python est un langage lisible.")
    rag.index_document("tests.txt", b"Pytest automatise les tests du projet.")

    sources = rag.retrieve(
        "Que contient ce document ?",
        document_hash=first.document.document_hash,
    )

    assert sources
    assert {source.filename for source in sources} == {"python.txt"}


def test_missing_information_does_not_call_generator(tmp_path) -> None:
    generator = FakeGenerator()
    store = ChromaDocumentStore(tmp_path / "empty", collection_name="empty_documents")
    rag = RAGService(
        tmp_path / "unused",
        embedding_provider=FakeEmbeddings(),
        store=store,
        generator=generator,
    )

    result = rag.answer("Question inconnue")

    assert not result.sources
    assert "pas trouvé" in result.answer
    assert generator.prompt == ""


def test_empty_question_returns_no_sources_without_generation(service) -> None:
    rag, generator = service
    rag.index_document("guide.txt", b"Python est un langage lisible.")

    result = rag.answer("   ")

    assert result.sources == []
    assert "pas trouvé" in result.answer
    assert generator.prompt == ""


def test_missing_api_key_is_explicit(tmp_path) -> None:
    store = ChromaDocumentStore(tmp_path / "nokey", collection_name="no_key_documents")
    rag = RAGService(
        tmp_path / "unused",
        embedding_provider=FakeEmbeddings(),
        store=store,
        generator=None,
        relevance_distance=2.0,
    )
    rag.index_document("guide.txt", b"Python est un langage lisible.")

    with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY"):
        rag.answer("Parle-moi de Python")


def test_github_repository_reuses_rag_pipeline(tmp_path) -> None:
    store = ChromaDocumentStore(tmp_path / "github", collection_name="github_documents")
    github_client = FakeGitHubClient()
    rag = RAGService(
        tmp_path / "unused",
        embedding_provider=FakeEmbeddings(),
        store=store,
        generator=FakeGenerator(),
        github_client=github_client,
        relevance_distance=2.0,
    )

    indexed = rag.index_github_repository("https://github.com/example/demo")
    sources = rag.retrieve(
        "Où est créée l'application Python ?",
        document_hash=indexed.document.document_hash,
    )

    assert not indexed.already_indexed
    assert indexed.document.source_type == "github"
    assert sources
    assert {source.source_type for source in sources} == {"github"}
    assert any(source.filename == "src/app.py" for source in sources)
    assert all(source.source_url and "/blob/main/" in source.source_url for source in sources)
    assert rag.list_documents(source_type="github")[0].filename == "example/demo"
    assert rag.list_documents(source_type="document") == []

    repeated = rag.index_github_repository("https://github.com/example/demo")
    assert repeated.already_indexed
    assert github_client.calls == 1


def test_github_indexing_reports_progress(service) -> None:
    rag, _ = service
    rag.github_client = FakeGitHubClient()
    updates: list[str] = []

    result = rag.index_github_repository_with_progress(
        "https://github.com/example/demo",
        on_progress=updates.append,
    )

    assert not result.already_indexed
    assert any("Connexion" in update for update in updates)
    assert any("embeddings" in update for update in updates)
    assert updates[-1] == "Enregistrement du repository…"


def test_github_indexing_failure_does_not_leave_partial_content(tmp_path) -> None:
    class FailingEmbeddings(FakeEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("embedding failed")

    store = ChromaDocumentStore(tmp_path / "failure", collection_name="failed_github")
    rag = RAGService(
        tmp_path / "unused",
        embedding_provider=FailingEmbeddings(),
        store=store,
        generator=FakeGenerator(),
        github_client=FakeGitHubClient(),
    )

    with pytest.raises(RuntimeError, match="embedding failed"):
        rag.index_github_repository("https://github.com/example/demo")

    assert store.count() == 0


def test_code_assistant_uses_generator_without_document_sources(service) -> None:
    rag, generator = service

    result = rag.generate_code_help(
        "def add(a, b): return a + b",
        task="Expliquer",
        language="Python",
        instruction="Insiste sur les paramètres.",
    )

    assert result.endswith("[1]")
    assert "aucune source documentaire indexée" in generator.prompt
    assert "def add" in generator.prompt
    assert "Insiste sur les paramètres" in generator.prompt


def test_code_assistant_requires_api_key(tmp_path) -> None:
    store = ChromaDocumentStore(tmp_path / "code", collection_name="code_no_key")
    rag = RAGService(
        tmp_path / "unused",
        embedding_provider=FakeEmbeddings(),
        store=store,
        generator=None,
    )

    with pytest.raises(MissingAPIKeyError, match="GROQ_API_KEY"):
        rag.generate_code_help("console.log('ok')", task="Expliquer", language="JavaScript")


def test_code_assistant_rejects_empty_code(service) -> None:
    rag, _ = service

    with pytest.raises(ValueError, match="Collez du code"):
        rag.generate_code_help("   ", task="Expliquer", language="Python")
