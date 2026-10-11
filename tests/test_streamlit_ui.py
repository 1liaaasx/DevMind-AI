from __future__ import annotations

import hashlib
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_FILE = Path(__file__).parents[1] / "streamlit_app.py"


def app(tmp_path, monkeypatch) -> AppTest:
    monkeypatch.setenv("DEVMIND_SESSION_ROOT", str(tmp_path / "sessions"))
    return AppTest.from_file(APP_FILE, default_timeout=30).run()


def html_contains(at: AppTest, text: str) -> bool:
    return any(text in (node.value or "") for node in at.get("html"))


def test_sources_panel_only_appears_for_current_answer(tmp_path, monkeypatch) -> None:
    at = app(tmp_path, monkeypatch)
    assert not at.exception
    assert not html_contains(at, "Sources consultées")


def test_chat_renders_code_blocks_and_github_source_links(tmp_path, monkeypatch) -> None:
    at = app(tmp_path, monkeypatch)
    at.session_state.current_view = "chat"
    at.session_state.messages = [
        {
            "role": "assistant",
            "content": "Exemple :\n```python\nprint('bonjour')\n```\nVoir [1].",
            "citations": [1],
            "sources": [
                {
                    "name": "example/demo",
                    "path": "src/app.py",
                    "page": None,
                    "text": "print('bonjour')",
                    "kind": "PY",
                    "url": "https://github.com/example/demo/blob/main/src/app.py",
                    "source_type": "github",
                }
            ],
        }
    ]
    at.run()

    assert not at.exception
    assert at.code[0].value == "print('bonjour')"
    assert any("Exemple" in (node.value or "") for node in at.markdown)
    assert html_contains(at, "Sources consultées")
    assert html_contains(at, "src/app.py")
    assert html_contains(at, "https://github.com/example/demo/blob/main/src/app.py")

    at.session_state.current_view = "chat"
    at.session_state.messages = [
        {
            "role": "assistant",
            "content": "Réponse actuelle",
            "sources": [
                {
                    "name": "guide.pdf",
                    "path": "guide.pdf",
                    "page": 2,
                    "text": "Extrait réellement retrouvé.",
                    "kind": "PDF",
                }
            ],
        }
    ]
    at.run()
    assert not at.exception
    assert html_contains(at, "Sources consultées")
    assert html_contains(at, "Extrait réellement retrouvé.")

    at.session_state.messages.append(
        {"role": "assistant", "content": "La nouvelle réponse ne contient aucune source."}
    )
    at.run()
    assert not at.exception
    assert not html_contains(at, "Sources consultées")


def test_chat_accepts_long_markdown_history_without_rendering_errors(tmp_path, monkeypatch) -> None:
    at = app(tmp_path, monkeypatch)
    at.session_state.current_view = "chat"
    at.session_state.messages = [
        {
            "role": "assistant" if index % 2 == 0 else "user",
            "content": (
                f"## Message {index + 1}\n\n"
                "**Résumé** avec une liste :\n\n- premier point\n- deuxième point\n\n"
                "```python\nfor value in range(3):\n    print(value)\n```\n\n"
                + "Paragraphe détaillé. " * 20
            ),
        }
        for index in range(10)
    ]

    at.run(timeout=30)

    assert not at.exception
    assert len(at.code) == 10
    assert sum("Message" in (node.value or "") for node in at.markdown) >= 10


def test_history_conversation_can_be_deleted(tmp_path, monkeypatch) -> None:
    at = app(tmp_path, monkeypatch)
    at.session_state.current_view = "history"
    at.session_state.history = [
        {
            "id": "conversation-1",
            "title": "Question sur le guide",
            "updated": "À l’instant",
            "messages": [{"role": "user", "content": "Question"}],
        }
    ]
    at.run()
    assert not at.exception

    at.button(key="delete_history_conversation-1").click().run()
    assert not at.exception
    at.button(key="confirm_conversation_conversation-1").click().run()

    assert not at.exception
    assert at.session_state.history == []


def test_github_page_is_enabled_and_rejects_invalid_url(tmp_path, monkeypatch) -> None:
    at = app(tmp_path, monkeypatch)
    assert not at.button(key="feature_github_button").disabled

    at.button(key="nav_github").click().run()
    assert not at.exception
    assert html_contains(at, "Explorateur GitHub")
    assert not html_contains(at, "Fonctionnalité à venir")

    at.text_input(key="github_repository_url").set_value("https://example.com/not-github")
    at.button(key="FormSubmitter:github_repository_form-Importer et indexer").click().run()

    assert not at.exception
    assert at.error
    assert "URL publique" in at.error[0].value


def test_code_assistant_is_enabled_and_reports_missing_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = app(tmp_path, monkeypatch)
    assert not at.button(key="feature_code_button").disabled

    at.button(key="nav_code").click().run()
    at.text_area(key="code_input").set_value("def hello():\n    return 'bonjour'")
    at.button(key="FormSubmitter:code_assistant_form-Analyser le code").click().run()

    assert not at.exception
    assert at.error
    assert "GROQ_API_KEY" in at.error[0].value
    assert not html_contains(at, "Sources consultées")


def test_document_upload_updates_library_and_can_be_deleted(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = app(tmp_path, monkeypatch)
    at.button(key="feature_docs_button").click().run()
    assert not at.exception

    content = b"Project UI Upload uses PostgreSQL."
    doc_hash = hashlib.sha256(content).hexdigest()
    at.file_uploader(key="document_uploader").upload(
        "ui-upload.txt", content, "text/plain"
    ).run()
    # A cold sentence-transformer import can be slow on CPU-only CI hosts.
    at.button(key="index_documents").click().run(timeout=120)

    assert not at.exception
    assert html_contains(at, "ui-upload.txt")
    assert at.success

    at.button(key=f"delete_{doc_hash}").click().run()
    assert not at.exception
    at.button(key=f"confirm_document_{doc_hash}").click().run()

    assert not at.exception
    assert html_contains(at, "Aucun document")


def test_history_reopens_and_settings_clear_it(tmp_path, monkeypatch) -> None:
    at = app(tmp_path, monkeypatch)
    at.session_state.history = [
        {
            "id": "conversation-1",
            "title": "Architecture Atlas",
            "updated": "À l’instant",
            "messages": [
                {"role": "user", "content": "Explique Atlas"},
                {"role": "assistant", "content": "Atlas utilise PostgreSQL."},
            ],
        }
    ]
    at.session_state.current_view = "history"
    at.run()

    at.button(key="open_conversation-1").click().run()
    assert not at.exception
    assert at.session_state.current_view == "chat"
    assert at.session_state.messages[-1]["content"] == "Atlas utilise PostgreSQL."

    at.button(key="nav_settings").click().run()
    at.button(key="clear_history").click().run()
    assert not at.exception
    at.button(key="confirm_clear").click().run()

    assert not at.exception
    assert at.session_state.history == []
    assert at.session_state.current_view == "welcome"
