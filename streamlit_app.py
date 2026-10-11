"""DevMind AI — document-grounded assistant with a polished Streamlit UI."""

from __future__ import annotations

import os
import logging
import re
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import Any, Callable, TypedDict

import streamlit as st
from dotenv import load_dotenv

from devmind_rag import (
    DocumentError,
    GitHubRepositoryError,
    GroqGenerationError,
    MissingAPIKeyError,
    RAGService,
)
from devmind_rag.embeddings import SentenceTransformerEmbedding
from devmind_rag.runtime import new_session_collection_name, process_chroma_directory


APP_DIR = Path(__file__).parent
UI_VERSION = 6
MAX_UPLOAD_MEGABYTES = 8
MAX_FILES_PER_UPLOAD = 5
MAX_QUESTION_CHARACTERS = 4_000
CODE_FENCE = re.compile(r"```([\w.+-]*)\s*\n(.*?)```", re.DOTALL)
LOGGER = logging.getLogger("devmind")

load_dotenv(APP_DIR / ".env", override=False)


class Message(TypedDict, total=False):
    role: str
    content: str
    citations: list[int]
    sources: list[dict[str, Any]]
    error: bool


class Conversation(TypedDict, total=False):
    id: str
    title: str
    updated: str
    messages: list[Message]
    document_filter_hash: str | None
    document_filter_name: str | None
    filter_source_type: str | None


st.set_page_config(
    page_title="DevMind AI — Assistant documentaire",
    page_icon=":material/terminal:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.html(f"<style>{(APP_DIR / 'styles.css').read_text(encoding='utf-8')}</style>")


@st.cache_resource(show_spinner=False)
def _shared_embedding_provider() -> SentenceTransformerEmbedding:
    """Share only the immutable model resource, never a user's document store."""

    return SentenceTransformerEmbedding()


def _groq_api_key() -> str | None:
    """Read Groq credentials from Streamlit Secrets or the local environment."""

    try:
        secret = str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception as exc:  # Missing local secrets file is expected during development.
        LOGGER.debug("Streamlit Secrets unavailable: %s", type(exc).__name__)
        secret = ""
    return secret or os.getenv("GROQ_API_KEY") or None


def get_rag_service() -> RAGService:
    """Return the RAG service isolated to the current Streamlit session."""

    service = st.session_state.get("_rag_service")
    if service is None:
        collection_name = st.session_state.setdefault(
            "_rag_collection_name",
            new_session_collection_name(),
        )
        service = RAGService(
            process_chroma_directory(),
            collection_name=collection_name,
            api_key=_groq_api_key(),
            embedding_provider=_shared_embedding_provider(),
        )
        st.session_state._rag_service = service
    return service


if st.session_state.get("ui_version") != UI_VERSION:
    st.session_state.ui_version = UI_VERSION
    st.session_state.current_view = "welcome"
    st.session_state.messages = []
    st.session_state.current_conversation_id = None
    st.session_state.conversation_title = "Nouvelle discussion"
    st.session_state.history = []
    st.session_state.conversation_seq = 0
    st.session_state.document_filter_hash = None
    st.session_state.document_filter_name = None
    st.session_state.filter_source_type = None
    st.session_state.document_notice = None
    st.session_state.repository_notice = None
    st.session_state.code_result = None
    st.session_state.code_error = None

st.session_state.setdefault("rag_request_in_progress", False)
st.session_state.setdefault("scroll_to_latest", False)


def navigate(view: str) -> None:
    st.session_state.current_view = view
    st.session_state.scroll_to_top = True


def open_assistant() -> None:
    st.session_state.current_view = "chat" if st.session_state.messages else "welcome"
    st.session_state.scroll_to_top = True


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.current_conversation_id = None
    st.session_state.conversation_title = "Nouvelle discussion"
    st.session_state.document_filter_hash = None
    st.session_state.document_filter_name = None
    st.session_state.filter_source_type = None
    st.session_state.current_view = "welcome"
    st.session_state.scroll_to_top = True


def start_assistant() -> None:
    st.session_state.current_view = "chat"
    st.session_state.scroll_to_top = True
    if not st.session_state.messages:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Bonjour ! Ajoutez vos documents, puis posez-moi une question. "
                    "Je répondrai uniquement à partir de leur contenu."
                ),
            }
        ]


def ask_document(doc_hash: str, filename: str) -> None:
    """Open a new conversation whose retrieval is restricted to one document."""

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"Je suis prêt à répondre à vos questions sur « {filename} ». "
                "Que souhaitez-vous comprendre ?"
            ),
        }
    ]
    st.session_state.current_conversation_id = None
    st.session_state.conversation_title = filename
    st.session_state.document_filter_hash = doc_hash
    st.session_state.document_filter_name = filename
    st.session_state.filter_source_type = "document"
    st.session_state.current_view = "chat"
    st.session_state.scroll_to_top = True


def ask_repository(repo_hash: str, repository: str) -> None:
    """Open a grounded conversation restricted to one indexed repository."""

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"Le repository « {repository} » est prêt. Vous pouvez me demander "
                "d’expliquer son architecture, un composant ou le rôle d’un fichier."
            ),
        }
    ]
    st.session_state.current_conversation_id = None
    st.session_state.conversation_title = repository
    st.session_state.document_filter_hash = repo_hash
    st.session_state.document_filter_name = repository
    st.session_state.filter_source_type = "github"
    st.session_state.current_view = "chat"
    st.session_state.scroll_to_top = True


def clear_document_filter() -> None:
    st.session_state.document_filter_hash = None
    st.session_state.document_filter_name = None
    st.session_state.filter_source_type = None


def save_current_conversation(prompt: str) -> None:
    conversation_id = st.session_state.current_conversation_id
    title = prompt.strip().replace("\n", " ")[:54] or "Documents ajoutés"
    if conversation_id is None:
        st.session_state.conversation_seq += 1
        conversation_id = f"conversation-{st.session_state.conversation_seq}"
        st.session_state.current_conversation_id = conversation_id
        st.session_state.conversation_title = title
        st.session_state.history.insert(
            0,
            {
                "id": conversation_id,
                "title": title,
                "updated": "À l’instant",
                "messages": deepcopy(st.session_state.messages),
                "document_filter_hash": st.session_state.document_filter_hash,
                "document_filter_name": st.session_state.document_filter_name,
                "filter_source_type": st.session_state.filter_source_type,
            },
        )
        return

    for conversation in st.session_state.history:
        if conversation["id"] == conversation_id:
            conversation["updated"] = "À l’instant"
            conversation["messages"] = deepcopy(st.session_state.messages)
            conversation["document_filter_hash"] = st.session_state.document_filter_hash
            conversation["document_filter_name"] = st.session_state.document_filter_name
            conversation["filter_source_type"] = st.session_state.filter_source_type
            break


def load_conversation(conversation_id: str) -> None:
    for conversation in st.session_state.history:
        if conversation["id"] == conversation_id:
            st.session_state.messages = deepcopy(conversation["messages"])
            st.session_state.current_conversation_id = conversation_id
            st.session_state.conversation_title = conversation["title"]
            st.session_state.document_filter_hash = conversation.get("document_filter_hash")
            st.session_state.document_filter_name = conversation.get("document_filter_name")
            st.session_state.filter_source_type = conversation.get("filter_source_type")
            st.session_state.current_view = "chat"
            st.session_state.scroll_to_top = True
            return


def clear_all_history() -> None:
    st.session_state.history = []
    st.session_state.messages = []
    st.session_state.current_conversation_id = None
    st.session_state.conversation_title = "Nouvelle discussion"
    st.session_state.document_filter_hash = None
    st.session_state.document_filter_name = None
    st.session_state.filter_source_type = None
    st.session_state.current_view = "welcome"
    st.session_state.scroll_to_top = True


def clear_session_data() -> None:
    """Erase sources and conversations belonging to the current session only."""

    get_rag_service().clear_session_data()
    clear_all_history()
    st.session_state.code_result = None
    st.session_state.code_error = None


@st.dialog("Effacer l’historique ?")
def confirm_clear_history() -> None:
    st.write("Cette action supprimera les conversations de la session. Vos documents resteront disponibles.")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Annuler", key="cancel_clear"):
            st.rerun()
        if st.button(
            "Effacer",
            type="primary",
            icon=":material/delete:",
            key="confirm_clear",
            on_click=clear_all_history,
        ):
            st.rerun()


@st.dialog("Effacer toutes les données de session ?")
def confirm_clear_session_data() -> None:
    st.write(
        "Cette action supprimera vos documents, repositories indexés et conversations "
        "de cette session de démonstration."
    )
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Annuler", key="cancel_clear_session"):
            st.rerun()
        if st.button(
            "Tout effacer",
            type="primary",
            icon=":material/delete_forever:",
            key="confirm_clear_session",
            on_click=clear_session_data,
        ):
            st.rerun()


def process_uploaded_files(files: list[Any]) -> tuple[list[str], list[str]]:
    """Index uploaded Streamlit files and return user-friendly outcomes."""

    added: list[str] = []
    errors: list[str] = []
    service = get_rag_service()
    selected_files = files[:MAX_FILES_PER_UPLOAD]
    if len(files) > MAX_FILES_PER_UPLOAD:
        errors.append(
            f"Ajoutez au maximum {MAX_FILES_PER_UPLOAD} fichiers par opération. "
            f"{len(files) - MAX_FILES_PER_UPLOAD} fichier(s) n’ont pas été traités."
        )
    for uploaded_file in selected_files:
        try:
            result = service.index_document(uploaded_file.name, uploaded_file.getvalue())
            if result.already_indexed:
                added.append(f"{result.document.filename} était déjà présent")
            else:
                added.append(f"{result.document.filename} a été ajouté")
        except DocumentError as exc:
            errors.append(f"{uploaded_file.name} : {exc}")
        except Exception:
            LOGGER.exception("Document indexing failed for %s", Path(uploaded_file.name).name)
            errors.append(
                f"{uploaded_file.name} : le traitement a échoué. Vérifiez le fichier et réessayez."
            )
    return added, errors


def process_github_repository(
    url: str,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[str | None, str | None]:
    """Index one public repository and return a user-facing result."""

    try:
        result = get_rag_service().index_github_repository_with_progress(
            url,
            on_progress=on_progress,
        )
    except GitHubRepositoryError as exc:
        return None, str(exc)
    except Exception:
        LOGGER.exception("GitHub repository indexing failed")
        return None, "L’indexation du repository a échoué. Vérifiez l’URL et réessayez."

    if result.already_indexed:
        return f"« {result.document.filename} » est déjà à jour dans votre espace.", None
    return (
        f"« {result.document.filename} » a été indexé avec {result.document.chunk_count} passage(s).",
        None,
    )


def source_from_result(source: Any) -> dict[str, Any]:
    return source.as_source()


def add_rag_exchange(
    prompt: str,
    files: list[Any] | None = None,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> bool:
    """Index attachments, retrieve passages, call Groq, and save the exchange."""

    def report(label: str) -> None:
        if on_progress is not None:
            on_progress(label)

    cleaned = prompt.strip()
    uploads = list(files or [])
    st.session_state.current_view = "chat"

    if len(cleaned) > MAX_QUESTION_CHARACTERS:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    f"La question dépasse la limite de {MAX_QUESTION_CHARACTERS} caractères "
                    "de cette démonstration. Raccourcissez-la, puis réessayez."
                ),
                "error": True,
            }
        )
        return False

    added, upload_errors = process_uploaded_files(uploads) if uploads else ([], [])
    user_content = cleaned
    if uploads:
        attachment_names = ", ".join(file.name for file in uploads)
        user_content = user_content or f"J’ai ajouté : {attachment_names}"
    if not user_content:
        return False

    st.session_state.messages.append({"role": "user", "content": user_content})
    if upload_errors:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Je n’ai pas pu ajouter certains fichiers :\n" + "\n".join(upload_errors),
                "error": True,
            }
        )

    if not cleaned:
        confirmation = "\n".join(f"• {message}." for message in added)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": confirmation or "Aucun document n’a été ajouté.",
                "error": not bool(added),
            }
        )
        save_current_conversation(user_content)
        return bool(added)

    try:
        service = get_rag_service()
        if getattr(service.embeddings, "is_ready", True):
            report("Recherche dans les sources…")
        else:
            report("Premier chargement du moteur local…")
        result = service.answer(
            cleaned,
            document_hash=st.session_state.document_filter_hash,
            on_sources_ready=lambda sources: report(
                "Rédaction de la réponse avec Groq…"
                if sources
                else "Vérification des informations disponibles…"
            ),
        )
        sources = [source_from_result(source) for source in result.sources]
        citations = []
        for raw_number in re.findall(r"\[(\d+)\]", result.answer):
            number = int(raw_number)
            if 1 <= number <= len(sources) and number not in citations:
                citations.append(number)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "citations": citations,
                "sources": sources,
            }
        )
    except MissingAPIKeyError as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": str(exc), "error": True}
        )
        success = False
    except GroqGenerationError as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": str(exc), "error": True}
        )
        success = False
    except Exception:
        LOGGER.exception("RAG answer generation failed")
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "La réponse n’a pas pu être générée. Vérifiez votre connexion, "
                    "votre clé Groq, puis réessayez."
                ),
                "error": True,
            }
        )
        success = False
    else:
        success = True
    save_current_conversation(cleaned)
    return success


def run_rag_exchange(prompt: str, files: list[Any] | None = None) -> None:
    """Run a chat turn with visible progress instead of a silent long wait."""

    if st.session_state.rag_request_in_progress:
        return
    st.session_state.rag_request_in_progress = True
    try:
        with st.status("Préparation de votre question…", expanded=False, type="compact") as status:
            succeeded = add_rag_exchange(
                prompt,
                files,
                on_progress=lambda label: status.update(label=label, state="running"),
            )
            status.update(
                label="Réponse prête" if succeeded else "La réponse n’a pas pu être générée",
                state="complete" if succeeded else "error",
            )
    finally:
        st.session_state.rag_request_in_progress = False
        st.session_state.scroll_to_latest = True


def delete_document(doc_hash: str, filename: str) -> None:
    get_rag_service().delete_document(doc_hash)
    if st.session_state.document_filter_hash == doc_hash:
        st.session_state.document_filter_hash = None
        st.session_state.document_filter_name = None
        st.session_state.filter_source_type = None
    st.session_state.document_notice = f"« {filename} » a été supprimé."


@st.dialog("Supprimer cette source ?")
def confirm_delete_document(doc_hash: str, filename: str) -> None:
    st.write(
        f"« {filename} » sera retiré de votre espace et ne pourra plus être consulté par l’assistant."
    )
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Annuler", key=f"cancel_document_{doc_hash}"):
            st.rerun()
        st.button(
            "Supprimer",
            type="primary",
            icon=":material/delete:",
            key=f"confirm_document_{doc_hash}",
            on_click=delete_document,
            args=(doc_hash, filename),
        )


def delete_conversation(conversation_id: str) -> None:
    st.session_state.history = [
        conversation
        for conversation in st.session_state.history
        if conversation["id"] != conversation_id
    ]
    if st.session_state.current_conversation_id == conversation_id:
        st.session_state.messages = []
        st.session_state.current_conversation_id = None
        st.session_state.conversation_title = "Nouvelle discussion"
        st.session_state.document_filter_hash = None
        st.session_state.document_filter_name = None
        st.session_state.filter_source_type = None
    st.session_state.current_view = "history"


@st.dialog("Supprimer cette conversation ?")
def confirm_delete_conversation(conversation_id: str, title: str) -> None:
    st.write(f"La conversation « {title} » sera supprimée de cette session.")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Annuler", key=f"cancel_conversation_{conversation_id}"):
            st.rerun()
        st.button(
            "Supprimer",
            type="primary",
            icon=":material/delete:",
            key=f"confirm_conversation_{conversation_id}",
            on_click=delete_conversation,
            args=(conversation_id,),
        )


def _message_parts(content: str) -> list[tuple[str, str, str | None]]:
    """Split Markdown and fenced code while preserving their original order."""

    parts: list[tuple[str, str, str | None]] = []
    cursor = 0
    for match in CODE_FENCE.finditer(content):
        if markdown := content[cursor:match.start()].strip():
            markdown = re.sub(r"<br\s*/?>", "  \n", markdown, flags=re.I)
            parts.append(("markdown", markdown, None))
        parts.append(("code", match.group(2).strip(), match.group(1) or "text"))
        cursor = match.end()
    if markdown := content[cursor:].strip():
        markdown = re.sub(r"<br\s*/?>", "  \n", markdown, flags=re.I)
        parts.append(("markdown", markdown, None))
    if not parts:
        parts.append(("markdown", content, None))
    return parts


def render_message(message: Message, index: int) -> None:
    """Render untrusted model text with native, safe Streamlit components."""

    is_user = message["role"] == "user"
    row_class = "user" if is_user else "assistant"
    avatar_class = "user" if is_user else "ai"
    avatar = "AM" if is_user else "DM"
    author = "Vous" if is_user else "DevMind"
    footer = ""
    if not is_user:
        if message.get("error"):
            footer = '<div class="answer-footer error">Action requise</div>'
        elif message.get("sources"):
            source_type = next(
                (source.get("source_type") for source in message.get("sources", []) if source),
                "document",
            )
            grounding = "Réponse fondée sur le repository" if source_type == "github" else "Réponse fondée sur vos documents"
            footer = f'<div class="answer-footer"><span>{grounding}</span><span>•</span><span>Références à droite</span></div>'

    with st.container(key=f"chat_message_{row_class}_{index}"):
        st.html(
            f'<div class="message-heading {row_class}">'
            f'<div class="message-avatar {avatar_class}">{avatar}</div>'
            f'<div class="message-meta"><span class="message-author">{author}</span>'
            '<span>•</span><span>à l’instant</span></div></div>'
        )
        for part_type, value, language in _message_parts(message.get("content", "")):
            if part_type == "code":
                st.code(value, language=language or "text", wrap_lines=False)
            else:
                # unsafe_allow_html remains False: model/user HTML is never executed.
                st.markdown(value)
        if footer:
            st.html(footer)


def source_html(source: dict[str, Any], number: int) -> str:
    page = source.get("page")
    source_type = source.get("source_type", "document")
    location = f"Page {page}" if page else ("Fichier GitHub" if source_type == "github" else "Document texte")
    repository_or_document = str(source["name"])
    source_path = str(source.get("path", repository_or_document))
    display_name = Path(source_path).name if source_type == "github" else repository_or_document
    name = escape(display_name)
    url = str(source.get("url") or "")
    if url.startswith("https://github.com/"):
        name = f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{name}</a>'
    return f"""
    <article class="source-card">
        <div class="source-top">
            <div class="file-icon">{escape(str(source.get('kind', 'DOC')))}</div>
            <div class="file-info">
                <div class="file-name"><span class="source-number">{number}</span>{name}</div>
                <div class="file-path">{escape(repository_or_document)} · {escape(source_path)}</div>
            </div>
        </div>
        <div class="source-snippet">{escape(str(source['text']))}</div>
        <div class="source-meta"><span class="source-tag">{location}</span></div>
    </article>
    """


def render_brand() -> None:
    st.html(
        """
        <div class="brand-lockup">
            <div class="brand-mark">&lt;/&gt;</div>
            <div><div class="brand-name">DevMind <span>AI</span></div><div class="brand-sub">Votre assistant développeur</div></div>
        </div>
        """
    )


def render_navigation() -> None:
    render_brand()
    st.button("Nouvelle discussion", icon=":material/add:", key="new_chat", on_click=reset_conversation)
    st.html('<div class="section-label">Espace de travail</div>')
    st.button("Assistant IA", icon=":material/auto_awesome:", key="nav_assistant", on_click=open_assistant)
    st.button("Mes documents", icon=":material/folder_open:", key="nav_documents", on_click=navigate, args=("documents",))
    st.button("Explorateur GitHub", icon=":material/account_tree:", key="nav_github", on_click=navigate, args=("github",))
    st.button("Assistant code", icon=":material/code:", key="nav_code", on_click=navigate, args=("code",))
    st.button("Historique", icon=":material/history:", key="nav_history", on_click=navigate, args=("history",))
    st.html('<div class="section-label">Préférences</div>')
    st.button("Paramètres", icon=":material/settings:", key="nav_settings", on_click=navigate, args=("settings",))


def render_welcome() -> None:
    with st.container(key="welcome_panel"):
        st.html(
            """
            <section class="welcome-hero">
                <div class="demo-badge"><span></span>Version de démonstration sécurisée</div>
                <div class="welcome-mark">&lt;/&gt;</div>
                <h1>DevMind AI</h1>
                <h2>Votre compagnon intelligent pour le code</h2>
                <p>Ajoutez vos ressources, retrouvez les informations utiles et obtenez des réponses ancrées dans leur contenu.</p>
            </section>
            """
        )
        first_row = st.columns(2, gap="small")
        second_row = st.columns(2, gap="small")
        feature_specs = [
            (first_row[0], "feature_ask", "auto_awesome", "Demander à l’IA", "Posez une question sur les documents de votre bibliothèque.", "Commencer", start_assistant, False),
            (first_row[1], "feature_docs", "description", "Explorer les documents", "Ajoutez et consultez vos PDF, fichiers texte et Markdown.", "Explorer", lambda: navigate("documents"), False),
            (second_row[0], "feature_github", "account_tree", "Explorer GitHub", "Importez un repository public et interrogez ses fichiers sources.", "Explorer", lambda: navigate("github"), False),
            (second_row[1], "feature_code", "code", "Assistant code", "Expliquez, corrigez et améliorez du code Python ou JavaScript.", "Ouvrir", lambda: navigate("code"), False),
        ]
        for column, key, icon, title, description, label, callback, disabled in feature_specs:
            with column:
                with st.container(key=key):
                    st.html(
                        f'<div class="feature-copy"><div class="feature-icon material-symbols-rounded">{icon}</div><div><div class="feature-title">{escape(title)}</div><div class="feature-description">{escape(description)}</div></div></div>'
                    )
                    st.button(label, icon=":material/arrow_forward:", key=f"{key}_button", on_click=callback, disabled=disabled)

        submission = st.chat_input(
            "Posez votre première question…",
            max_chars=MAX_QUESTION_CHARACTERS,
            max_upload_size=MAX_UPLOAD_MEGABYTES,
            accept_file="multiple",
            file_type=["pdf", "txt", "md", "markdown"],
            submit_mode="disable",
            key="welcome_prompt",
        )
        if submission:
            run_rag_exchange(submission.text, list(submission.files))
            st.rerun()


def render_chat() -> None:
    context_type = st.session_state.get("filter_source_type")
    with st.container(key="topbar"):
        header_copy, header_buttons = st.columns([0.66, 0.34], vertical_alignment="center", gap="small")
        with header_copy:
            title = st.session_state.conversation_title
            if title == "Nouvelle discussion":
                title = "Assistant IA"
            header_meta = (
                "Posez une question sur les fichiers du repository sélectionné"
                if context_type == "github"
                else "Posez une question sur le contenu de votre bibliothèque"
            )
            st.html(
                f"""
                <div class="chat-header"><div class="header-glyph">λ</div><div class="header-copy">
                    <div class="chat-title">{escape(title)}</div>
                    <div class="header-meta">{escape(header_meta)}</div>
                </div></div>
                """
            )
        with header_buttons:
            with st.container(horizontal=True, horizontal_alignment="right", key="header_actions"):
                st.download_button(
                    "Exporter",
                    data="\n\n".join(item["content"] for item in st.session_state.messages),
                    file_name="devmind-conversation.md",
                    mime="text/markdown",
                    icon=":material/download:",
                    key="export",
                )

    if filter_name := st.session_state.document_filter_name:
        context_icon = "account_tree" if context_type == "github" else "description"
        context_label = "Repository sélectionné" if context_type == "github" else "Document sélectionné"
        with st.container(horizontal=True, vertical_alignment="center", key="document_context"):
            st.html(
                f'<div class="context-copy"><span class="material-symbols-rounded">{context_icon}</span><div><strong>{context_label}</strong><br>{escape(filter_name)}</div></div>'
            )
            st.button(
                "Interroger toutes les sources",
                icon=":material/close:",
                key="clear_document_filter",
                on_click=clear_document_filter,
            )

    with st.container(border=False, key="message_area"):
        for index, item in enumerate(st.session_state.messages):
            render_message(item, index)

    if context_type == "github":
        quick_prompts = [
            ("Expliquer l’architecture", "Explique l’architecture générale de ce repository et le rôle de ses composants."),
            ("Points d’entrée", "Quels sont les principaux points d’entrée de ce repository ?"),
            ("Flux principal", "Décris le flux d’exécution principal à partir des fichiers disponibles."),
        ]
    else:
        quick_prompts = [
            ("Résumer les documents", "Résume les informations essentielles des documents ajoutés."),
            ("Points importants", "Quels sont les points les plus importants à retenir ?"),
            ("Trouver un exemple", "Trouve et explique un exemple concret présent dans les documents."),
        ]
    with st.container(horizontal=True, wrap=False, key="quick_actions"):
        for index, (label, question) in enumerate(quick_prompts):
            if st.button(label, icon=":material/arrow_forward:", key=f"quick_{index}"):
                run_rag_exchange(question)
                st.rerun()

    if context_type == "github":
        submission = st.chat_input(
            "Posez une question sur ce repository…",
            max_chars=MAX_QUESTION_CHARACTERS,
            submit_mode="stop",
            key="chat_prompt",
        )
    else:
        submission = st.chat_input(
            "Posez une question sur vos sources…",
            max_chars=MAX_QUESTION_CHARACTERS,
            max_upload_size=MAX_UPLOAD_MEGABYTES,
            accept_file="multiple",
            file_type=["pdf", "txt", "md", "markdown"],
            submit_mode="stop",
            key="chat_prompt",
        )
    if submission:
        if context_type == "github":
            run_rag_exchange(str(submission))
        else:
            run_rag_exchange(submission.text, list(submission.files))
        st.rerun()


def render_documents() -> None:
    with st.container(key="secondary_page"):
        st.html(
            '<div class="page-heading"><div class="page-icon material-symbols-rounded">folder_open</div><div><h1>Mes documents</h1><p>Ajoutez des PDF, fichiers texte ou Markdown, puis interrogez leur contenu avec DevMind AI.</p></div></div>'
        )
        notice = st.session_state.pop("document_notice", None)
        if notice:
            st.success(notice, icon=":material/check_circle:")

        with st.container(key="upload_card"):
            st.subheader("Ajouter des documents", icon=":material/upload_file:")
            uploads = st.file_uploader(
                "Sélectionner des documents",
                type=["pdf", "txt", "md", "markdown"],
                accept_multiple_files=True,
                max_upload_size=MAX_UPLOAD_MEGABYTES,
                help=(
                    "Formats acceptés : PDF, TXT et Markdown. "
                    f"{MAX_UPLOAD_MEGABYTES} Mo maximum par fichier, "
                    f"{MAX_FILES_PER_UPLOAD} fichiers par opération."
                ),
                key="document_uploader",
            )
            if st.button(
                "Analyser et ajouter",
                type="primary",
                icon=":material/auto_awesome:",
                disabled=not uploads,
                key="index_documents",
            ):
                with st.spinner("Lecture et préparation des documents…"):
                    added, errors = process_uploaded_files(list(uploads))
                for message in added:
                    if "déjà présent" in message:
                        st.info(message, icon=":material/info:")
                    else:
                        st.success(message, icon=":material/check_circle:")
                for message in errors:
                    st.error(message, icon=":material/error:")

        st.html(
            '<div class="privacy-note"><span class="material-symbols-rounded">shield</span><div><strong>Espace temporaire isolé</strong><p>Vos documents restent séparés des autres utilisateurs pendant cette session. Ils ne sont pas conservés durablement. Seuls les extraits utiles à une réponse sont transmis à Groq.</p></div></div>'
        )

        documents = get_rag_service().list_documents(source_type="document")
        st.html(f'<div class="document-list-heading"><strong>Bibliothèque</strong><span>{len(documents)} document(s)</span></div>')
        if not documents:
            st.html(
                '<div class="empty-state compact"><div class="empty-illustration material-symbols-rounded">description</div><h2>Aucun document</h2><p>Ajoutez votre premier fichier pour commencer à poser des questions.</p></div>'
            )
            return

        for document in documents:
            with st.container(key=f"document_{document.document_hash[:12]}"):
                info, ask_action, delete_action = st.columns(
                    [0.58, 0.25, 0.17],
                    vertical_alignment="center",
                    gap="small",
                )
                with info:
                    page_copy = f" · {document.page_count} page(s)" if document.page_count else ""
                    file_type = Path(document.filename).suffix.lstrip(".").upper() or "FICHIER"
                    st.html(
                        f'<div class="document-info"><div class="document-icon material-symbols-rounded">description</div><div><div class="document-title">{escape(document.filename)}</div><div class="document-meta"><span>{escape(file_type)}</span> · Prêt à être consulté{page_copy}</div></div></div>'
                    )
                with ask_action:
                    st.button(
                        "Poser une question",
                        icon=":material/chat:",
                        key=f"ask_{document.document_hash}",
                        on_click=ask_document,
                        args=(document.document_hash, document.filename),
                    )
                with delete_action:
                    if st.button(
                        "Supprimer",
                        icon=":material/delete:",
                        key=f"delete_{document.document_hash}",
                    ):
                        confirm_delete_document(document.document_hash, document.filename)


def render_github() -> None:
    with st.container(key="secondary_page"):
        st.html(
            '<div class="page-heading"><div class="page-icon material-symbols-rounded">account_tree</div><div><h1>Explorateur GitHub</h1><p>Importez un repository public, puis interrogez son architecture et ses fichiers sources.</p></div></div>'
        )
        notice = st.session_state.pop("repository_notice", None)
        if notice:
            st.success(notice, icon=":material/check_circle:")

        with st.container(key="github_import_card"):
            st.subheader("Importer un repository", icon=":material/cloud_download:")
            with st.form("github_repository_form", border=False):
                repository_url = st.text_input(
                    "URL du repository public",
                    placeholder="https://github.com/proprietaire/repository",
                    help="Seuls les repositories GitHub publics sont pris en charge.",
                    key="github_repository_url",
                )
                submitted = st.form_submit_button(
                    "Importer et indexer",
                    type="primary",
                    icon=":material/auto_awesome:",
                )
            if submitted:
                import_status = st.status(
                    "Validation de l’URL GitHub…",
                    expanded=False,
                    type="compact",
                )
                message, error = process_github_repository(
                    repository_url,
                    on_progress=lambda label: import_status.update(
                        label=label,
                        state="running",
                    ),
                )
                if error:
                    import_status.update(label="Import interrompu", state="error")
                    st.error(error, icon=":material/error:")
                elif message:
                    import_status.update(label="Repository prêt", state="complete")
                    st.success(message, icon=":material/check_circle:")

        st.html(
            '<div class="privacy-note"><span class="material-symbols-rounded">filter_alt</span><div><strong>Import sélectif et temporaire</strong><p>Les dépendances, fichiers binaires, fichiers générés et contenus volumineux sont exclus. Le repository reste isolé dans votre session et n’est pas conservé durablement.</p></div></div>'
        )

        repositories = get_rag_service().list_documents(source_type="github")
        st.html(
            f'<div class="document-list-heading"><strong>Repositories indexés</strong><span>{len(repositories)} repository(s)</span></div>'
        )
        if not repositories:
            st.html(
                '<div class="empty-state compact"><div class="empty-illustration material-symbols-rounded">account_tree</div><h2>Aucun repository</h2><p>Importez un repository GitHub public pour commencer son exploration.</p></div>'
            )
            return

        for repository in repositories:
            with st.container(key=f"repository_{repository.document_hash[:12]}"):
                info, ask_action, delete_action = st.columns(
                    [0.58, 0.25, 0.17],
                    vertical_alignment="center",
                    gap="small",
                )
                with info:
                    safe_url = escape(repository.source_url or "", quote=True)
                    title = escape(repository.filename)
                    if safe_url:
                        title = f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{title}</a>'
                    st.html(
                        f'<div class="document-info"><div class="document-icon material-symbols-rounded">account_tree</div><div><div class="document-title">{title}</div><div class="document-meta"><span>GITHUB</span> · {repository.chunk_count} passage(s) indexé(s)</div></div></div>'
                    )
                with ask_action:
                    st.button(
                        "Explorer",
                        icon=":material/chat:",
                        key=f"ask_repository_{repository.document_hash}",
                        on_click=ask_repository,
                        args=(repository.document_hash, repository.filename),
                    )
                with delete_action:
                    if st.button(
                        "Supprimer",
                        icon=":material/delete:",
                        key=f"delete_repository_{repository.document_hash}",
                    ):
                        confirm_delete_document(repository.document_hash, repository.filename)


def render_code_assistant() -> None:
    with st.container(key="code_page"):
        st.html(
            '<div class="page-heading"><div class="page-icon material-symbols-rounded">code</div><div><h1>Assistant code</h1><p>Expliquez, diagnostiquez, corrigez ou améliorez du code Python et JavaScript.</p></div></div>'
        )
        st.html(
            '<div class="code-mode-note"><span class="material-symbols-rounded">info</span><div><strong>Analyse générale</strong><p>Cette page analyse uniquement le code collé. Elle n’utilise pas vos documents ni vos repositories indexés.</p></div></div>'
        )
        with st.container(key="code_editor_card"):
            with st.form("code_assistant_form", border=False):
                selectors = st.columns(2, gap="small")
                with selectors[0]:
                    language = st.selectbox(
                        "Langage",
                        ["Python", "JavaScript"],
                        key="code_language",
                    )
                with selectors[1]:
                    task = st.selectbox(
                        "Type d’analyse",
                        ["Expliquer", "Analyser une erreur", "Corriger", "Améliorer", "Générer un exemple"],
                        key="code_task",
                    )
                code = st.text_area(
                    "Code à analyser",
                    height=260,
                    max_chars=50_000,
                    placeholder="Collez votre code ici…",
                    key="code_input",
                )
                error_message = st.text_area(
                    "Message d’erreur — facultatif",
                    height=90,
                    placeholder="Traceback, message de console ou comportement observé…",
                    key="code_error_message",
                )
                instruction = st.text_input(
                    "Consigne complémentaire — facultative",
                    placeholder="Ex. : privilégie une solution simple et compatible Python 3.12",
                    key="code_instruction",
                )
                analyze = st.form_submit_button(
                    "Analyser le code",
                    type="primary",
                    icon=":material/auto_awesome:",
                )

            if analyze:
                st.session_state.code_result = None
                st.session_state.code_error = None
                try:
                    with st.spinner("Analyse du code avec DevMind AI…"):
                        st.session_state.code_result = get_rag_service().generate_code_help(
                            code,
                            task=task,
                            language=language,
                            error_message=error_message,
                            instruction=instruction,
                        )
                except (GroqGenerationError, MissingAPIKeyError, ValueError) as exc:
                    st.session_state.code_error = str(exc)
                except Exception:
                    LOGGER.exception("Code assistant generation failed")
                    st.session_state.code_error = (
                        "L’analyse n’a pas pu être générée. Vérifiez votre connexion et votre clé Groq."
                    )

        if st.session_state.code_error:
            st.error(st.session_state.code_error, icon=":material/error:")
        if result := st.session_state.code_result:
            with st.container(key="code_result_card"):
                st.html(
                    '<div class="code-result-heading"><span class="material-symbols-rounded">auto_awesome</span><div><strong>Analyse DevMind AI</strong><small>Réponse générale · aucune source indexée utilisée</small></div></div>'
                )
                for part_type, value, language in _message_parts(result):
                    if part_type == "code":
                        st.code(value, language=language or "text", wrap_lines=False)
                    else:
                        st.markdown(value)


def render_history() -> None:
    with st.container(key="history_page"):
        st.html('<div class="page-heading"><div class="page-icon material-symbols-rounded">history</div><div><h1>Historique</h1><p>Reprenez une conversation récente. L’historique est conservé uniquement pendant cette session.</p></div></div>')
        if not st.session_state.history:
            st.html('<div class="empty-state compact"><div class="empty-illustration material-symbols-rounded">forum</div><h2>Aucune conversation</h2><p>Vos prochaines discussions apparaîtront ici.</p></div>')
            return
        for conversation in st.session_state.history:
            with st.container(key=f"history_{conversation['id']}"):
                info, open_action, delete_action = st.columns(
                    [0.68, 0.2, 0.12],
                    vertical_alignment="center",
                    gap="small",
                )
                with info:
                    st.html(f'<div class="history-info"><div class="history-title">{escape(conversation["title"])}</div><div class="history-meta">{escape(conversation["updated"])} · {len(conversation["messages"])} messages</div></div>')
                with open_action:
                    st.button("Ouvrir", icon=":material/arrow_forward:", key=f"open_{conversation['id']}", on_click=load_conversation, args=(conversation["id"],))
                with delete_action:
                    if st.button(
                        "Supprimer",
                        icon=":material/delete:",
                        key=f"delete_history_{conversation['id']}",
                    ):
                        confirm_delete_conversation(conversation["id"], conversation["title"])


def render_settings() -> None:
    with st.container(key="settings_page"):
        st.html('<div class="page-heading"><div class="page-icon material-symbols-rounded">settings</div><div><h1>Paramètres</h1><p>Gérez les préférences disponibles pour cette session.</p></div></div>')
        with st.container(key="settings_card"):
            st.subheader("Interface", icon=":material/tune:")
            st.caption("DevMind AI utilise actuellement une apparence claire et une interface en français.")
        with st.container(key="danger_card"):
            st.subheader("Historique des conversations", icon=":material/history:")
            st.caption("Supprimez les conversations de cette session sans effacer vos documents.")
            if st.button("Effacer l’historique", icon=":material/delete:", key="clear_history"):
                confirm_clear_history()
        with st.container(key="session_data_card"):
            st.subheader("Données de démonstration", icon=":material/shield:")
            st.caption(
                "Les sources et conversations sont temporaires et isolées par session. "
                "Vous pouvez les supprimer immédiatement."
            )
            if st.button(
                "Effacer toutes mes données",
                icon=":material/delete_forever:",
                key="clear_session_data",
            ):
                confirm_clear_session_data()


def latest_sources() -> list[dict[str, Any]]:
    for message in reversed(st.session_state.messages):
        if message.get("role") == "assistant":
            return message.get("sources", [])
    return []


def render_sources_panel(sources: list[dict[str, Any]]) -> None:
    count_label = f"{len(sources)} source" + ("s" if len(sources) > 1 else "")
    st.html(
        f'<div class="sources-heading"><div><div class="sources-title">Sources consultées</div><div class="sources-subtitle">Références utilisées dans la réponse</div></div><div class="source-count">{count_label}</div></div>'
    )
    query = st.text_input("Rechercher dans les sources", type="search", placeholder="Rechercher une source…", label_visibility="collapsed", key="source_search")
    normalized_query = query.casefold().strip()
    filtered_sources = [
        (number, source)
        for number, source in enumerate(sources, start=1)
        if not normalized_query
        or normalized_query in str(source["name"]).casefold()
        or normalized_query in str(source.get("path", "")).casefold()
        or normalized_query in str(source["text"]).casefold()
    ]
    cards = "".join(source_html(source, number) for number, source in filtered_sources)
    if not cards:
        cards = '<div class="source-card"><div class="source-snippet">Aucune source ne correspond à cette recherche.</div></div>'
    st.html(f'<div class="source-list">{cards}</div>')


scroll_destination: str | None = None
if st.session_state.pop("scroll_to_top", False):
    scroll_destination = "top"
elif st.session_state.pop("scroll_to_latest", False):
    scroll_destination = "latest"


view = st.session_state.current_view
active_nav_key = "nav_assistant" if view in {"welcome", "chat"} else f"nav_{view}"
st.html(f"<style>.st-key-{active_nav_key} .stButton > button {{ color:#5144c7; border-color:rgba(108,92,231,.11); background:linear-gradient(90deg,rgba(108,92,231,.13),rgba(108,92,231,.06)); }}</style>")

current_sources = latest_sources() if view == "chat" else []
show_sources = bool(current_sources)
if show_sources:
    shell_key = "app_shell_chat_sources"
elif view == "chat":
    shell_key = "app_shell_chat"
else:
    shell_key = "app_shell_workspace"

with st.container(key=shell_key):
    columns = st.columns([0.19, 0.60, 0.21] if show_sources else [0.19, 0.81], gap=None)
    left, center = columns[0], columns[1]
    with left:
        with st.container(key="left_panel"):
            render_navigation()
    with center:
        with st.container(key="chat_panel"):
            if view == "welcome":
                render_welcome()
            elif view == "chat":
                render_chat()
            elif view == "documents":
                render_documents()
            elif view == "github":
                render_github()
            elif view == "code":
                render_code_assistant()
            elif view == "history":
                render_history()
            elif view == "settings":
                render_settings()
    if show_sources:
        with columns[2]:
            with st.container(key="sources_panel"):
                render_sources_panel(current_sources)

# The embedding model is unnecessary for the welcome, history, settings and
# code-only pages. Load it once, in the background, only when RAG is in use.
if view in {"chat", "documents", "github"}:
    embedding_provider = _shared_embedding_provider()
    if hasattr(embedding_provider, "warm_up_async"):
        embedding_provider.warm_up_async()

if scroll_destination:
    st.html(
        f"""
        <script>
        requestAnimationFrame(() => requestAnimationFrame(() => {{
            const main = document.querySelector('[data-testid="stMain"]');
            const chat = document.querySelector('.st-key-chat_panel');
            const scroller = chat && getComputedStyle(chat).overflowY !== 'visible' ? chat : main;
            if (scroller) {{
                const top = {"0" if scroll_destination == "top" else "scroller.scrollHeight"};
                scroller.scrollTo({{ top, left: 0, behavior: 'auto' }});
            }}
        }}));
        </script>
        """,
        unsafe_allow_javascript=True,
    )
