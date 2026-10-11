"""Safe ingestion of source files from public GitHub repositories."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from time import monotonic
from typing import Callable
from urllib.parse import quote, urlparse

import requests

from .documents import clean_text


GITHUB_HOSTS = {"github.com", "www.github.com"}
REPOSITORY_PART = re.compile(r"^[A-Za-z0-9_.-]+$")
EXCLUDED_DIRECTORIES = {
    ".git",
    ".idea",
    ".next",
    ".venv",
    ".vscode",
    "__pycache__",
    "bower_components",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
    "venv",
}
EXCLUDED_FILENAMES = {
    "bun.lockb",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "yarn.lock",
}
TEXT_EXTENSIONS = {
    ".c", ".cc", ".cfg", ".cpp", ".cs", ".css", ".csv", ".env.example",
    ".go", ".gradle", ".h", ".hpp", ".htm", ".html", ".ini", ".java",
    ".js", ".json", ".jsx", ".kt", ".kts", ".md", ".markdown", ".mjs",
    ".php", ".properties", ".ps1", ".py", ".rb", ".rs", ".rst", ".sass",
    ".scala", ".scss", ".sh", ".sql", ".svelte", ".swift", ".toml", ".ts",
    ".tsx", ".txt", ".vue", ".xml", ".yaml", ".yml",
}
TEXT_FILENAMES = {
    "dockerfile",
    "gemfile",
    "license",
    "makefile",
    "procfile",
    "readme",
    "requirements.txt",
}
ARCHITECTURE_FILENAMES = {
    "cargo.toml",
    "composer.json",
    "gemfile",
    "go.mod",
    "makefile",
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
}
SOURCE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".go", ".java", ".js", ".jsx",
    ".kt", ".php", ".py", ".rb", ".rs", ".scala", ".swift", ".ts", ".tsx",
}


class GitHubRepositoryError(ValueError):
    """Raised when a public repository cannot be imported safely."""


@dataclass(frozen=True)
class GitHubFile:
    path: str
    text: str


@dataclass(frozen=True)
class GitHubRepository:
    slug: str
    url: str
    default_branch: str
    document_hash: str
    files: list[GitHubFile]


def parse_github_repository_url(url: str) -> tuple[str, str]:
    """Validate a public GitHub repository URL and return owner/repository."""

    raw = url.strip()
    if not raw:
        raise GitHubRepositoryError("Saisissez l’URL d’un repository GitHub public.")
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in GITHUB_HOSTS:
        raise GitHubRepositoryError("Utilisez une URL publique de la forme https://github.com/proprietaire/repository.")
    if parsed.username or parsed.password or parsed.port:
        raise GitHubRepositoryError("L’URL GitHub contient des informations non autorisées.")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        raise GitHubRepositoryError("L’URL doit pointer vers la racine d’un repository GitHub.")
    owner, repository = parts
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository or not REPOSITORY_PART.fullmatch(owner) or not REPOSITORY_PART.fullmatch(repository):
        raise GitHubRepositoryError("L’URL GitHub n’est pas valide.")
    return owner, repository


def _is_supported_text_file(path: PurePosixPath) -> bool:
    name = path.name.casefold()
    if name in EXCLUDED_FILENAMES or name.endswith((".min.js", ".min.css", ".map")):
        return False
    if name in TEXT_FILENAMES or name.startswith("readme.") or name.startswith("license."):
        return True
    return path.suffix.casefold() in TEXT_EXTENSIONS


def _decode_source(data: bytes) -> str | None:
    if b"\x00" in data[:8192]:
        return None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return clean_text(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    return None


def _file_priority(info: zipfile.ZipInfo) -> tuple[int, str]:
    """Put architecture and source files before tests, examples and generated docs."""

    path = PurePosixPath(info.filename)
    relative = PurePosixPath(*path.parts[1:]) if len(path.parts) > 1 else path
    lowered_parts = {part.casefold() for part in relative.parts[:-1]}
    name = relative.name.casefold()
    if name.startswith("readme") or name in ARCHITECTURE_FILENAMES:
        priority = 0
    elif relative.suffix.casefold() in SOURCE_EXTENSIONS and not lowered_parts.intersection(
        {"examples", "fixtures", "tests", "test"}
    ):
        priority = 1
    elif lowered_parts.intersection({"examples", "fixtures", "tests", "test"}):
        priority = 3
    elif name.startswith("license") or name in {"copying", "notice"}:
        priority = 4
    else:
        priority = 2
    return priority, info.filename.casefold()


def extract_repository_files(
    archive: bytes,
    *,
    max_files: int = 250,
    max_file_bytes: int = 250_000,
    max_total_bytes: int = 5_000_000,
    max_total_characters: int = 400_000,
) -> list[GitHubFile]:
    """Extract a bounded set of useful text files from a GitHub ZIP archive."""

    try:
        zipped = zipfile.ZipFile(io.BytesIO(archive))
    except (zipfile.BadZipFile, OSError) as exc:
        raise GitHubRepositoryError("GitHub n’a pas renvoyé une archive valide.") from exc

    files: list[GitHubFile] = []
    total_bytes = 0
    total_characters = 0
    with zipped:
        for info in sorted(zipped.infolist(), key=_file_priority):
            if info.is_dir():
                continue
            archive_path = PurePosixPath(info.filename)
            relative_parts = archive_path.parts[1:]
            if not relative_parts or any(part in {"", ".", ".."} for part in relative_parts):
                continue
            relative_path = PurePosixPath(*relative_parts)
            lowered_parts = {part.casefold() for part in relative_path.parts[:-1]}
            if lowered_parts & EXCLUDED_DIRECTORIES or not _is_supported_text_file(relative_path):
                continue
            if info.file_size <= 0 or info.file_size > max_file_bytes:
                continue
            if total_bytes + info.file_size > max_total_bytes:
                continue
            text = _decode_source(zipped.read(info))
            if not text:
                continue
            if total_characters + len(text) > max_total_characters:
                continue
            total_bytes += info.file_size
            total_characters += len(text)
            files.append(GitHubFile(path=relative_path.as_posix(), text=text))
            if len(files) >= max_files:
                break

    if not files:
        raise GitHubRepositoryError("Aucun fichier source textuel exploitable n’a été trouvé dans ce repository.")
    return files


class GitHubRepositoryClient:
    """Download one public repository using GitHub's REST API archive endpoint."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout: float | tuple[float, float] = (5.0, 10.0),
        max_archive_bytes: int = 15_000_000,
        max_download_seconds: float = 30.0,
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_archive_bytes = max_archive_bytes
        self.max_download_seconds = max_download_seconds
        self.headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "DevMind-AI",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _get(self, url: str, *, stream: bool = False):
        try:
            response = self.session.get(
                url,
                headers=self.headers,
                timeout=self.timeout,
                allow_redirects=True,
                stream=stream,
            )
        except requests.RequestException as exc:
            raise GitHubRepositoryError("GitHub est momentanément inaccessible. Réessayez plus tard.") from exc
        if response.status_code == 404:
            raise GitHubRepositoryError("Repository introuvable ou non accessible publiquement.")
        if response.status_code == 403:
            raise GitHubRepositoryError("La limite d’accès GitHub a été atteinte. Réessayez plus tard.")
        if response.status_code >= 400:
            raise GitHubRepositoryError(f"GitHub a refusé la requête (erreur {response.status_code}).")
        return response

    def fetch(
        self,
        url: str,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> GitHubRepository:
        def report(label: str) -> None:
            if on_progress is not None:
                on_progress(label)

        owner, repository = parse_github_repository_url(url)
        canonical_url = f"https://github.com/{owner}/{repository}"
        report("Lecture des informations du repository…")
        try:
            metadata = self._get(f"https://api.github.com/repos/{owner}/{repository}").json()
        except GitHubRepositoryError:
            raise
        except (requests.RequestException, ValueError) as exc:
            raise GitHubRepositoryError("GitHub a renvoyé une réponse illisible.") from exc
        if bool(metadata.get("private")):
            raise GitHubRepositoryError("Seuls les repositories publics sont pris en charge.")
        default_branch = str(metadata.get("default_branch") or "").strip()
        if not default_branch:
            raise GitHubRepositoryError("La branche principale du repository est introuvable.")

        report("Téléchargement sécurisé de l’archive…")
        archive_response = self._get(
            f"https://api.github.com/repos/{owner}/{repository}/zipball/{quote(default_branch, safe='')}",
            stream=True,
        )
        try:
            content_length = int(archive_response.headers.get("content-length", "0") or 0)
        except ValueError:
            content_length = 0
        if content_length > self.max_archive_bytes:
            raise GitHubRepositoryError("L’archive du repository est trop volumineuse.")
        buffer = bytearray()
        download_started = monotonic()
        for block in archive_response.iter_content(chunk_size=65_536):
            if not block:
                continue
            if monotonic() - download_started > self.max_download_seconds:
                raise GitHubRepositoryError(
                    "Le téléchargement du repository a dépassé la durée autorisée."
                )
            buffer.extend(block)
            if len(buffer) > self.max_archive_bytes:
                raise GitHubRepositoryError("L’archive du repository est trop volumineuse.")
        archive = bytes(buffer)
        report("Sélection des fichiers utiles…")
        files = extract_repository_files(archive)
        return GitHubRepository(
            slug=f"{owner}/{repository}",
            url=canonical_url,
            default_branch=default_branch,
            document_hash=hashlib.sha256(archive).hexdigest(),
            files=files,
        )
