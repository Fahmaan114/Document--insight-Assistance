from fastapi.testclient import TestClient

import app.routes as routes_module
from app.config import Settings
from app.main import app as fastapi_app
from rag.embeddings import index_document_chunks
from rag.ingest import ingest_document
from rag.prompt_builder import (
    INSUFFICIENT_CONTEXT_ANSWER,
    build_grounded_prompt,
    generate_grounded_answer,
)
from rag.retrieve import retrieve_relevant_chunks


def test_build_grounded_prompt_includes_strict_grounding_instruction() -> None:
    prompt = build_grounded_prompt(
        question="What does the contract say about termination?",
        retrieved_chunks=[],
    )

    assert "only the provided document context" in prompt[0]["content"]
    assert INSUFFICIENT_CONTEXT_ANSWER in prompt[0]["content"]


def test_generate_grounded_answer_returns_supported_answer_with_sources(tmp_path) -> None:
    document = ingest_document(
        filename="contract.txt",
        file_bytes=(
            b"Termination requires thirty days written notice. "
            b"Payment is due monthly on the first business day."
        ),
        chunk_size=1000,
        chunk_overlap=200,
        max_upload_size_bytes=10_000,
    )
    index_document_chunks(
        chunks=document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
    )
    retrieval = retrieve_relevant_chunks(
        question="What is required for termination?",
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
        top_k=3,
        min_score=0.0,
    )

    answer = generate_grounded_answer(
        question="What is required for termination?",
        retrieved_chunks=retrieval.results,
        answer_provider="local",
        max_sources=3,
        snippet_chars=220,
    )

    assert answer.answer_supported is True
    assert "thirty days written notice" in answer.answer.lower()
    assert len(answer.sources) >= 1
    assert answer.sources[0].filename == "contract.txt"
    assert "Termination requires thirty days written notice" in answer.sources[0].snippet


def test_generate_grounded_answer_succeeds_when_retrieval_succeeds_for_file_types_query(tmp_path) -> None:
    document = ingest_document(
        filename="readme.txt",
        file_bytes=(
            b"The system accepts PDF and TXT files. "
            b"Uploaded files are chunked and indexed locally."
        ),
        chunk_size=1000,
        chunk_overlap=200,
        max_upload_size_bytes=10_000,
    )
    index_document_chunks(
        chunks=document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
    )
    retrieval = retrieve_relevant_chunks(
        question="What file types does the system accept?",
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
        top_k=3,
        min_score=0.0,
    )

    assert retrieval.no_results is False
    assert retrieval.results

    answer = generate_grounded_answer(
        question="What file types does the system accept?",
        retrieved_chunks=retrieval.results,
        answer_provider="local",
        max_sources=3,
        snippet_chars=220,
    )

    assert answer.answer_supported is True
    assert "pdf and txt files" in answer.answer.lower()
    assert answer.sources
    assert answer.sources[0].filename == "readme.txt"


def test_generate_grounded_answer_returns_fallback_when_context_is_insufficient() -> None:
    answer = generate_grounded_answer(
        question="Who won the World Cup in 2018?",
        retrieved_chunks=[],
        answer_provider="local",
        max_sources=3,
        snippet_chars=220,
    )

    assert answer.answer_supported is False
    assert answer.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert answer.sources == []


def test_answer_endpoint_returns_grounded_answer_and_source_snippets(tmp_path, monkeypatch) -> None:
    settings = Settings(
        EMBEDDING_PROVIDER="local_dummy",
        LOCAL_DUMMY_DIMENSION=64,
        EMBEDDING_BATCH_SIZE=4,
        RETRIEVAL_TOP_K=3,
        RETRIEVAL_MIN_SCORE=0.0,
        ANSWER_PROVIDER="local",
        ANSWER_MAX_SOURCES=2,
        ANSWER_SNIPPET_CHARS=180,
        vectorstore_dir=tmp_path,
    )
    monkeypatch.setattr(routes_module, "get_settings", lambda: settings)

    indexed_document = ingest_document(
        filename="policy.txt",
        file_bytes=(
            b"Termination requires thirty days written notice. "
            b"Payment is due monthly on the first business day."
        ),
        chunk_size=1000,
        chunk_overlap=200,
        max_upload_size_bytes=10_000,
    )
    index_document_chunks(
        chunks=indexed_document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
    )

    client = TestClient(fastapi_app)
    response = client.post(
        "/answer",
        json={"question": "What is required for termination?", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_supported"] is True
    assert "thirty days written notice" in payload["answer"].lower()
    assert payload["sources"][0]["filename"] == "policy.txt"
    assert "Termination requires thirty days written notice" in payload["sources"][0]["snippet"]


def test_answer_endpoint_matches_retrieve_success_for_same_relevant_query(tmp_path, monkeypatch) -> None:
    settings = Settings(
        EMBEDDING_PROVIDER="local_dummy",
        LOCAL_DUMMY_DIMENSION=64,
        EMBEDDING_BATCH_SIZE=4,
        RETRIEVAL_TOP_K=3,
        RETRIEVAL_MIN_SCORE=0.0,
        ANSWER_PROVIDER="local",
        ANSWER_MAX_SOURCES=2,
        ANSWER_SNIPPET_CHARS=180,
        vectorstore_dir=tmp_path,
    )
    monkeypatch.setattr(routes_module, "get_settings", lambda: settings)

    indexed_document = ingest_document(
        filename="guide.txt",
        file_bytes=(
            b"The system accepts PDF and TXT files. "
            b"Uploaded files are chunked and indexed locally."
        ),
        chunk_size=1000,
        chunk_overlap=200,
        max_upload_size_bytes=10_000,
    )
    index_document_chunks(
        chunks=indexed_document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
    )

    client = TestClient(fastapi_app)
    retrieve_response = client.post(
        "/retrieve",
        json={"question": "What file types does the system accept?", "top_k": 3},
    )
    answer_response = client.post(
        "/answer",
        json={"question": "What file types does the system accept?", "top_k": 3},
    )

    assert retrieve_response.status_code == 200
    retrieve_payload = retrieve_response.json()
    assert retrieve_payload["no_results"] is False
    assert retrieve_payload["results"]

    assert answer_response.status_code == 200
    answer_payload = answer_response.json()
    assert answer_payload["answer_supported"] is True
    assert "pdf and txt files" in answer_payload["answer"].lower()
    assert answer_payload["sources"]
    assert answer_payload["sources"][0]["chunk_id"] == retrieve_payload["results"][0]["chunk_id"]


def test_answer_endpoint_returns_insufficient_context_fallback(tmp_path, monkeypatch) -> None:
    settings = Settings(
        EMBEDDING_PROVIDER="local_dummy",
        LOCAL_DUMMY_DIMENSION=64,
        EMBEDDING_BATCH_SIZE=4,
        RETRIEVAL_TOP_K=3,
        RETRIEVAL_MIN_SCORE=0.2,
        ANSWER_PROVIDER="local",
        ANSWER_MAX_SOURCES=2,
        ANSWER_SNIPPET_CHARS=180,
        vectorstore_dir=tmp_path,
    )
    monkeypatch.setattr(routes_module, "get_settings", lambda: settings)

    indexed_document = ingest_document(
        filename="science.txt",
        file_bytes=b"Gravity affects orbit and mass.",
        chunk_size=1000,
        chunk_overlap=200,
        max_upload_size_bytes=10_000,
    )
    index_document_chunks(
        chunks=indexed_document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
    )

    client = TestClient(fastapi_app)
    response = client.post(
        "/answer",
        json={"question": "Who won the World Cup in 2018?", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer_supported"] is False
    assert payload["answer"] == INSUFFICIENT_CONTEXT_ANSWER
    assert payload["sources"] == []
