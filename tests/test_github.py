from __future__ import annotations

import io
import zipfile

import pytest

import devmind_rag.github as github_module
from devmind_rag.github import (
    GitHubRepositoryClient,
    GitHubRepositoryError,
    extract_repository_files,
    parse_github_repository_url,
)


def make_archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, content in files.items():
            archive.writestr(f"owner-repo-sha/{path}", content)
    return buffer.getvalue()


def test_parse_public_repository_url() -> None:
    assert parse_github_repository_url("https://github.com/openai/openai-python.git") == (
        "openai",
        "openai-python",
    )


@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://gitlab.com/openai/openai-python",
        "https://github.com/openai",
        "https://github.com/openai/openai-python/issues",
        "file:///tmp/repository",
    ],
)
def test_invalid_repository_urls_are_rejected(url: str) -> None:
    with pytest.raises(GitHubRepositoryError):
        parse_github_repository_url(url)


def test_archive_excludes_dependencies_binary_and_large_files() -> None:
    archive = make_archive(
        {
            "README.md": b"# Demo\nArchitecture du projet.",
            "src/app.py": b"def run():\n    return 'ok'",
            "node_modules/pkg/index.js": b"ignored",
            "assets/logo.png": b"\x89PNG\x00binary",
            "src/generated.min.js": b"ignored",
            "src/huge.py": b"x" * 400_000,
        }
    )

    files = extract_repository_files(archive)

    assert {item.path for item in files} == {"README.md", "src/app.py"}


def test_archive_prioritizes_architecture_and_source_files_under_a_limit() -> None:
    archive = make_archive(
        {
            "tests/test_app.py": b"def test_app(): pass",
            "docs/guide.md": b"Guide",
            "src/app.py": b"def run(): return 'ok'",
            "pyproject.toml": b"[project]",
            "README.md": b"# Architecture",
        }
    )

    files = extract_repository_files(archive, max_files=3)

    assert [item.path for item in files] == [
        "pyproject.toml",
        "README.md",
        "src/app.py",
    ]


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}


class FakeSession:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def get(self, *args, **kwargs):
        return FakeResponse(self.status_code)


def test_inaccessible_repository_has_clear_error() -> None:
    client = GitHubRepositoryClient(session=FakeSession(404))

    with pytest.raises(GitHubRepositoryError, match="introuvable"):
        client.fetch("https://github.com/example/missing")


def test_repository_download_has_a_total_deadline(monkeypatch) -> None:
    class MetadataResponse:
        status_code = 200
        headers: dict[str, str] = {}

        @staticmethod
        def json():
            return {"private": False, "default_branch": "main"}

    class ArchiveResponse:
        status_code = 200
        headers: dict[str, str] = {}

        @staticmethod
        def iter_content(chunk_size: int):
            yield b"partial archive"

    class SequenceSession:
        def __init__(self) -> None:
            self.responses = [MetadataResponse(), ArchiveResponse()]

        def get(self, *args, **kwargs):
            return self.responses.pop(0)

    clock = iter([0.0, 31.0])
    monkeypatch.setattr(github_module, "monotonic", lambda: next(clock))
    client = GitHubRepositoryClient(
        session=SequenceSession(),
        max_download_seconds=30.0,
    )

    with pytest.raises(GitHubRepositoryError, match="durée autorisée"):
        client.fetch("https://github.com/example/demo")
