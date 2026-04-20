from pathlib import Path

from fastapi.testclient import TestClient

import app.routes as routes_module
from app.config import Settings
from app.main import app as fastapi_app
from rag.embeddings import index_document_chunks
from rag.ingest import ingest_document
from rag.retrieve import retrieve_relevant_chunks


def test_retrieve_relevant_chunks_returns_ranked_results_with_metadata(tmp_path: Path) -> None:
    fruit_document = ingest_document(
        filename="fruit.txt",
        file_bytes=b"apple banana apple banana",
        chunk_size=1000,
        chunk_overlap=200,
        max_upload_size_bytes=10_000,
    )
    engine_document = ingest_document(
        filename="engine.txt",
        file_bytes=b"engine motor piston engine",
        chunk_size=1000,
        chunk_overlap=200,
        max_upload_size_bytes=10_000,
    )

    index_document_chunks(
        chunks=fruit_document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
    )
    index_document_chunks(
        chunks=engine_document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
    )

    response = retrieve_relevant_chunks(
        question="banana engine",
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
        top_k=2,
    )

    assert response.no_results is False
    assert len(response.results) == 2
    assert {result.filename for result in response.results} == {"fruit.txt", "engine.txt"}
    assert {result.vector_id for result in response.results} == {0, 1}
    assert {result.chunk_id for result in response.results} == {
        fruit_document.chunks[0].chunk_id,
        engine_document.chunks[0].chunk_id,
    }
    assert [result.rank for result in response.results] == [1, 2]


def test_retrieve_relevant_chunks_handles_empty_index(tmp_path: Path) -> None:
    response = retrieve_relevant_chunks(
        question="What is in the document?",
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
        top_k=3,
    )

    assert response.no_results is True
    assert response.results == []
    assert "No indexed document chunks" in response.message


def test_retrieve_relevant_chunks_handles_no_useful_context_for_unrelated_question(tmp_path: Path) -> None:
    document = ingest_document(
        filename="science.txt",
        file_bytes=b"gravity orbit mass acceleration",
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

    response = retrieve_relevant_chunks(
        question="Who won the World Cup in 2018?",
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
        top_k=3,
        min_score=0.2,
    )

    assert response.no_results is True
    assert response.results == []
    assert "No useful context" in response.message


def test_retrieve_relevant_chunks_succeeds_for_file_types_query_with_local_dummy(tmp_path: Path) -> None:
    document = ingest_document(
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
        chunks=document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
    )

    response = retrieve_relevant_chunks(
        question="What file types does the system accept?",
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
        top_k=3,
        min_score=0.2,
    )

    assert response.no_results is False
    assert response.results
    assert response.results[0].filename == "guide.txt"
    assert "pdf and txt files" in response.results[0].chunk_text.lower()


def test_retrieve_relevant_chunks_deduplicates_duplicate_chunk_ids(tmp_path: Path) -> None:
    duplicated_document = ingest_document(
        filename="dup.txt",
        file_bytes=b"apple banana apple banana",
        chunk_size=1000,
        chunk_overlap=200,
        max_upload_size_bytes=10_000,
    )

    index_document_chunks(
        chunks=duplicated_document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
    )
    index_document_chunks(
        chunks=duplicated_document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
    )

    response = retrieve_relevant_chunks(
        question="apple banana",
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
        top_k=5,
        min_score=0.2,
    )

    assert response.no_results is False
    assert len(response.results) == 1
    assert response.results[0].chunk_id == duplicated_document.chunks[0].chunk_id


def test_retrieve_relevant_chunks_filters_out_low_score_matches(tmp_path: Path) -> None:
    document = ingest_document(
        filename="policy.txt",
        file_bytes=b"termination notice policy termination notice",
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

    response = retrieve_relevant_chunks(
        question="termination notice",
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=4,
        api_key="",
        local_dummy_dimension=64,
        top_k=3,
        min_score=1.1,
    )

    assert response.no_results is True
    assert response.results == []
    assert "No useful context" in response.message


def test_retrieve_endpoint_returns_ranked_results(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        EMBEDDING_PROVIDER="local_dummy",
        LOCAL_DUMMY_DIMENSION=64,
        EMBEDDING_BATCH_SIZE=4,
        RETRIEVAL_TOP_K=2,
        RETRIEVAL_MIN_SCORE=0.0,
        vectorstore_dir=tmp_path,
    )
    monkeypatch.setattr(routes_module, "get_settings", lambda: settings)

    indexed_document = ingest_document(
        filename="policy.txt",
        file_bytes=b"termination notice policy termination notice",
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
        "/retrieve",
        json={"question": "termination notice", "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["no_results"] is False
    assert payload["results"][0]["filename"] == "policy.txt"
    assert payload["results"][0]["chunk_id"] == indexed_document.chunks[0].chunk_id


def test_retrieve_endpoint_matches_answer_success_for_same_relevant_query(tmp_path: Path, monkeypatch) -> None:
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
    assert answer_payload["sources"]
    assert answer_payload["sources"][0]["chunk_id"] == retrieve_payload["results"][0]["chunk_id"]


def test_retrieve_endpoint_handles_empty_index(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        EMBEDDING_PROVIDER="local_dummy",
        LOCAL_DUMMY_DIMENSION=64,
        EMBEDDING_BATCH_SIZE=4,
        RETRIEVAL_TOP_K=2,
        RETRIEVAL_MIN_SCORE=0.0,
        vectorstore_dir=tmp_path,
    )
    monkeypatch.setattr(routes_module, "get_settings", lambda: settings)

    client = TestClient(fastapi_app)
    response = client.post(
        "/retrieve",
        json={"question": "anything", "top_k": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["no_results"] is True
    assert payload["results"] == []
