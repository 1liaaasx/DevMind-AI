"""Process- and session-scoped runtime helpers for public Streamlit deployments."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock


DEFAULT_STALE_PROCESS_AGE_SECONDS = 24 * 60 * 60
PROCESS_DIRECTORY_PREFIX = "process-"
SESSION_COLLECTION_PREFIX = "devmind_session_"

_PROCESS_ID = uuid.uuid4().hex
_PREPARED_ROOTS: set[Path] = set()
_PREPARE_LOCK = Lock()


def _private_directory(path: Path) -> Path:
    """Create a directory and restrict its permissions where supported."""

    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:
        # Windows ACLs and some hosted filesystems do not implement POSIX modes.
        pass
    return path


def session_storage_root(configured_root: str | Path | None = None) -> Path:
    """Return the administrator-controlled root for temporary session data."""

    raw_root = configured_root or os.getenv("DEVMIND_SESSION_ROOT")
    root = Path(raw_root).expanduser() if raw_root else Path(tempfile.gettempdir()) / "devmind-ai"
    return _private_directory(root.resolve())


def cleanup_stale_process_directories(
    root: str | Path,
    *,
    max_age_seconds: int = DEFAULT_STALE_PROCESS_AGE_SECONDS,
    now: float | None = None,
) -> int:
    """Remove only expired DevMind process directories from the temporary root."""

    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds doit être strictement positif.")
    storage_root = _private_directory(Path(root).resolve())
    current_time = time.time() if now is None else now
    removed = 0
    for candidate in storage_root.glob(f"{PROCESS_DIRECTORY_PREFIX}*"):
        if not candidate.is_dir() or candidate.name == f"{PROCESS_DIRECTORY_PREFIX}{_PROCESS_ID}":
            continue
        try:
            age = current_time - candidate.stat().st_mtime
        except OSError:
            continue
        if age <= max_age_seconds:
            continue
        try:
            shutil.rmtree(candidate)
        except OSError:
            continue
        removed += 1
    return removed


def process_chroma_directory(configured_root: str | Path | None = None) -> Path:
    """Return this server process's ephemeral Chroma directory.

    All sessions in one process reuse the same Chroma client files to keep the
    memory footprint small, but every session receives a separate, unguessable
    collection name. A server restart creates a new process namespace, so stale
    session data is never reopened by the application.
    """

    root = session_storage_root(configured_root)
    with _PREPARE_LOCK:
        if root not in _PREPARED_ROOTS:
            cleanup_stale_process_directories(root)
            _PREPARED_ROOTS.add(root)
    process_directory = _private_directory(root / f"{PROCESS_DIRECTORY_PREFIX}{_PROCESS_ID}")
    try:
        os.utime(process_directory, None)
    except OSError:
        pass
    return _private_directory(process_directory / "chroma")


def new_session_collection_name() -> str:
    """Create an unguessable Chroma collection name for one browser session."""

    return f"{SESSION_COLLECTION_PREFIX}{uuid.uuid4().hex}"
