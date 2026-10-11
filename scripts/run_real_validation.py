"""Bounded real-API validation campaign for DevMind AI.

This script intentionally uses a temporary ChromaDB directory and performs at
most twelve Groq calls. It never prints or persists the API key.
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from devmind_rag.embeddings import SentenceTransformerEmbedding
from devmind_rag.groq import DEFAULT_GROQ_MODEL, GroqGenerator
from devmind_rag.service import RAGService


sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class MeasuredGenerator:
    def __init__(self, api_key: str) -> None:
        self.inner = GroqGenerator(api_key)
        self.calls: list[dict[str, float | int]] = []

    def generate(self, prompt: str) -> str:
        started = time.perf_counter()
        response = self.inner.generate(prompt)
        self.calls.append(
            {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "response_characters": len(response),
            }
        )
        return response


def citations_are_real(answer: str, source_count: int) -> bool:
    citations = [
        int(left or right)
        for left, right in re.findall(r"\[(\d+)\]|【\s*(\d+)\s*】", answer)
    ]
    return bool(citations) and all(1 <= value <= source_count for value in citations)


def compact_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:900]


def checked_answer(
    service: RAGService,
    question: str,
    *,
    document_hash: str,
    expected_terms: tuple[str, ...] = (),
) -> dict:
    started = time.perf_counter()
    result = service.answer(question, document_hash=document_hash)
    elapsed = time.perf_counter() - started
    lowered = result.answer.casefold()
    sources = [source.as_source() for source in result.sources]
    return {
        "question": question,
        "answer": compact_answer(result.answer),
        "elapsed_seconds": round(elapsed, 3),
        "expected_terms_present": all(term.casefold() in lowered for term in expected_terms),
        "source_count": len(sources),
        "source_paths": sorted({str(source["path"]) for source in sources}),
        "source_types": sorted({str(source["source_type"]) for source in sources}),
        "citations_valid": citations_are_real(result.answer, len(sources)),
    }


def main() -> int:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    temp_root = Path(tempfile.mkdtemp(prefix="devmind-real-validation-"))
    output: dict = {
        "model": DEFAULT_GROQ_MODEL,
        "temporary_storage": str(temp_root),
        "results": {},
        "performance": {},
    }
    service = None
    try:
        embeddings = SentenceTransformerEmbedding()
        started = time.perf_counter()
        _ = embeddings.model
        output["performance"]["embedding_model_load_seconds"] = round(
            time.perf_counter() - started, 3
        )

        generator = MeasuredGenerator(api_key)
        started = time.perf_counter()
        basic_response = generator.generate(
            "Réponds uniquement par le mot OK, sans ponctuation."
        )
        output["results"]["basic_groq"] = {
            "answer": compact_answer(basic_response),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "responded": bool(basic_response.strip()),
        }
        service = RAGService(
            temp_root / "chroma",
            embedding_provider=embeddings,
            generator=generator,
        )

        atlas = (
            "Project Atlas uses Python 3.12.\n"
            "Its main database is PostgreSQL.\n"
            "The system processes 250 events per minute.\n"
            "The technical manager is named Karim."
        ).encode("utf-8")
        started = time.perf_counter()
        atlas_index = service.index_document("atlas_test.txt", atlas)
        output["performance"]["first_document_index_seconds"] = round(
            time.perf_counter() - started, 3
        )
        atlas_hash = atlas_index.document.document_hash

        started = time.perf_counter()
        atlas_retrieval = service.retrieve(
            "Which database does Project Atlas use?", document_hash=atlas_hash
        )
        output["performance"]["semantic_retrieval_seconds"] = round(
            time.perf_counter() - started, 4
        )

        atlas_questions = [
            ("Which database does Project Atlas use?", ("PostgreSQL",)),
            ("How many events per minute does Project Atlas process?", ("250",)),
        ]
        atlas_results = [
            checked_answer(
                service,
                question,
                document_hash=atlas_hash,
                expected_terms=expected,
            )
            for question, expected in atlas_questions
        ]
        hallucination = checked_answer(
            service,
            "What is the annual revenue of Project Atlas?",
            document_hash=atlas_hash,
        )
        refusal_text = hallucination["answer"].casefold()
        hallucination["declines_unsupported_answer"] = any(
            marker in refusal_text
            for marker in (
                "pas indiqu",
                "pas mention",
                "ne contient",
                "non disponible",
                "information manque",
                "pas suffisamment",
                "n’est pas fournie",
                "n'est pas fournie",
                "n’est pas présente",
                "n'est pas présente",
            )
        )
        output["results"]["document_rag"] = {
            "indexed_chunks": atlas_index.document.chunk_count,
            "retrieval_has_evidence": bool(atlas_retrieval)
            and all(source.filename == "atlas_test.txt" for source in atlas_retrieval),
            "questions": atlas_results,
            "hallucination_test": hallucination,
        }

        count_before_duplicate = service.store.count()
        duplicate = service.index_document("atlas_copy.txt", atlas)
        count_after_duplicate = service.store.count()

        orion = (
            "Project Orion is a JavaScript monitoring service.\n"
            "Its database is SQLite.\n"
            "It processes 80 alerts per minute.\n"
            "Its technical manager is Leila."
        ).encode("utf-8")
        orion_index = service.index_document("orion_test.md", orion)
        orion_hash = orion_index.document.document_hash
        multi_answers = [
            checked_answer(
                service,
                "Which database does Project Orion use?",
                document_hash=orion_hash,
                expected_terms=("SQLite",),
            )
        ]
        atlas_filtered = service.retrieve("database", document_hash=atlas_hash)
        orion_filtered = service.retrieve("database", document_hash=orion_hash)
        service.delete_document(atlas_hash)
        atlas_after_delete = service.retrieve("PostgreSQL", document_hash=atlas_hash)
        orion_after_delete = service.retrieve("SQLite", document_hash=orion_hash)
        output["results"]["multi_document"] = {
            "deduplication_detected": duplicate.already_indexed,
            "chunk_count_unchanged_after_duplicate": count_before_duplicate
            == count_after_duplicate,
            "atlas_filter_only_atlas": bool(atlas_filtered)
            and {source.filename for source in atlas_filtered} == {"atlas_test.txt"},
            "orion_filter_only_orion": bool(orion_filtered)
            and {source.filename for source in orion_filtered} == {"orion_test.md"},
            "answers": multi_answers,
            "atlas_removed": not atlas_after_delete,
            "orion_preserved": bool(orion_after_delete)
            and {source.filename for source in orion_after_delete} == {"orion_test.md"},
        }

        started = time.perf_counter()
        repository = service.index_github_repository(
            "https://github.com/PacktPublishing/Prompt-Engineering-in-Python-with-GPT-and-the-OpenAI-API"
        )
        output["performance"]["github_fetch_index_seconds"] = round(
            time.perf_counter() - started, 3
        )
        repository_hash = repository.document.document_hash
        repository_questions = [
            ("Le but principal de ce repo", ("prompt", "engineering")),
        ]
        github_answers = [
            checked_answer(
                service,
                question,
                document_hash=repository_hash,
                expected_terms=expected,
            )
            for question, expected in repository_questions
        ]
        output["results"]["github"] = {
            "repository": repository.document.filename,
            "indexed_chunks": repository.document.chunk_count,
            "answers": github_answers,
            "all_sources_are_github": all(
                item["source_types"] == ["github"] for item in github_answers
            ),
            "all_answers_have_real_paths": all(
                item["source_paths"] and item["citations_valid"]
                for item in github_answers
            ),
            "documents_still_present": any(
                item.document_hash == orion_hash
                for item in service.list_documents(source_type="document")
            ),
        }

        code_scenarios = [
            {
                "mode": "Expliquer",
                "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                "error": "",
                "expected": ("récurs", "factoriel"),
            },
            {
                "mode": "Analyser une erreur",
                "code": "numbers = [1, 2, 3]\nprint(numbers[5])",
                "error": "IndexError: list index out of range",
                "expected": ("indexerror",),
            },
            {
                "mode": "Améliorer",
                "code": "numbers = [1, 2, 3, 4, 5]\nresult = []\nfor n in numbers:\n    result.append(n * 2)",
                "error": "",
                "expected": ("compréhension",),
            },
        ]
        code_results = []
        for scenario in code_scenarios:
            started = time.perf_counter()
            answer = service.generate_code_help(
                scenario["code"],
                task=scenario["mode"],
                language="Python",
                error_message=scenario["error"],
            )
            lowered = answer.casefold()
            code_results.append(
                {
                    "mode": scenario["mode"],
                    "answer": compact_answer(answer),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "expected_terms_present": all(
                        term in lowered for term in scenario["expected"]
                    ),
                    "document_sources_returned": False,
                    "contains_code_fence": "```" in answer,
                }
            )
        output["results"]["code_assistant"] = code_results
        output["real_groq_calls"] = len(generator.calls)
        output["performance"]["groq_calls"] = generator.calls
        output["success"] = True
        print(json.dumps(output, ensure_ascii=True, indent=2))
        return 0
    finally:
        if service is not None:
            service.store.client.clear_system_cache()
        service = None
        gc.collect()
        try:
            shutil.rmtree(temp_root)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
