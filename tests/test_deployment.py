from __future__ import annotations

import os
from pathlib import Path

from devmind_rag.runtime import (
    PROCESS_DIRECTORY_PREFIX,
    cleanup_stale_process_directories,
    new_session_collection_name,
    process_chroma_directory,
)


PROJECT_ROOT = Path(__file__).parents[1]


def test_session_collection_names_are_unique_and_cloud_safe() -> None:
    first = new_session_collection_name()
    second = new_session_collection_name()

    assert first != second
    assert first.startswith("devmind_session_")
    assert all(character.isalnum() or character == "_" for character in first)


def test_process_storage_is_temporary_and_scoped_under_configured_root(tmp_path) -> None:
    directory = process_chroma_directory(tmp_path)

    assert directory.is_dir()
    assert directory.parent.parent == tmp_path.resolve()
    assert directory.parent.name.startswith(PROCESS_DIRECTORY_PREFIX)


def test_cleanup_removes_only_expired_devmind_process_directories(tmp_path) -> None:
    stale = tmp_path / f"{PROCESS_DIRECTORY_PREFIX}stale"
    recent = tmp_path / f"{PROCESS_DIRECTORY_PREFIX}recent"
    unrelated = tmp_path / "other-data"
    for directory in (stale, recent, unrelated):
        directory.mkdir()
    os.utime(stale, (1.0, 1.0))
    os.utime(recent, (9_950.0, 9_950.0))

    removed = cleanup_stale_process_directories(
        tmp_path,
        max_age_seconds=100,
        now=10_000.0,
    )

    assert removed == 1
    assert not stale.exists()
    assert recent.exists()
    assert unrelated.exists()


def test_publication_ignores_secrets_private_data_and_logs() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    for required_pattern in (
        ".env",
        ".streamlit/secrets.toml",
        ".devmind/",
        "*.log",
    ):
        assert required_pattern in gitignore

    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "votre_cle_api_groq" in example
    assert "gsk_" not in example
