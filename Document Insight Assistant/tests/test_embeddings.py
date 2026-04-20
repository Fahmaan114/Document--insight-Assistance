from pathlib import Path

from fastapi.testclient import TestClient

import app.routes as routes_module
from app.config import Settings
from app.main import app as fastapi_app
from rag.embeddings import (
    get_vectorstore_paths,
    index_document_chunks,
    load_vector_store,
)
from rag.ingest import ingest_document


def test_index_document_chunks_saves_and_reloads_vector_store_with_local_dummy(
    tmp_path: Path,
) -> None:
    document = ingest_document(
        filename="notes.txt",
        file_bytes=("alpha beta gamma " * 80).encode("utf-8"),
        chunk_size=120,
        chunk_overlap=30,
        max_upload_size_bytes=1_000_000,
    )

    artifacts = index_document_chunks(
        chunks=document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_64",
        batch_size=2,
        api_key="",
        local_dummy_dimension=64,
    )

    index, metadata = load_vector_store(vectorstore_dir=tmp_path)
    index_path, metadata_path = get_vectorstore_paths(tmp_path)

    assert index is not None
    assert index.ntotal == len(document.chunks)
    assert index.d == 64
    assert artifacts.added_vector_count == len(document.chunks)
    assert artifacts.total_vector_count == len(document.chunks)
    assert artifacts.embedding_provider == "local_dummy"
    assert artifacts.embedding_dimension == 64
    assert Path(artifacts.index_path) == index_path
    assert Path(artifacts.metadata_path) == metadata_path
    assert metadata.embedding_provider == "local_dummy"
    assert metadata.embedding_model == "local_dummy_64"
    assert metadata.embedding_dimension == 64
    assert metadata.vector_count == len(document.chunks)
    assert metadata.entries[0].vector_id == 0
    assert metadata.entries[-1].vector_id == len(document.chunks) - 1
    assert metadata.entries[0].chunk_id == document.chunks[0].chunk_id


def test_index_document_chunks_appends_metadata_in_vector_order(
    tmp_path: Path,
) -> None:
    first_document = ingest_document(
        filename="first.txt",
        file_bytes=("first document content " * 40).encode("utf-8"),
        chunk_size=100,
        chunk_overlap=20,
        max_upload_size_bytes=1_000_000,
    )
    second_document = ingest_document(
        filename="second.txt",
        file_bytes=("second document content " * 40).encode("utf-8"),
        chunk_size=100,
        chunk_overlap=20,
        max_upload_size_bytes=1_000_000,
    )

    index_document_chunks(
        chunks=first_document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_32",
        batch_size=4,
        api_key="",
        local_dummy_dimension=32,
    )
    artifacts = index_document_chunks(
        chunks=second_document.chunks,
        vectorstore_dir=tmp_path,
        embedding_provider="local_dummy",
        embedding_model="local_dummy_32",
        batch_size=4,
        api_key="",
        local_dummy_dimension=32,
    )

    index, metadata = load_vector_store(vectorstore_dir=tmp_path)

    assert index is not None
    assert index.ntotal == len(first_document.chunks) + len(second_document.chunks)
    assert artifacts.total_vector_count == index.ntotal
    assert metadata.entries[len(first_document.chunks)].filename == "second.txt"
    assert metadata.entries[len(first_document.chunks)].vector_id == len(first_document.chunks)


def test_local_provider_falls_back_to_local_dummy_in_development(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "rag.embeddings._embed_with_sentence_transformers",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    test_settings = Settings(
        EMBEDDING_PROVIDER="local",
        OPENAI_API_KEY="",
        LOCAL_EMBEDDING_MODEL="test-local-model",
        LOCAL_DUMMY_DIMENSION=48,
        EMBEDDING_BATCH_SIZE=2,
        CHUNK_SIZE_CHARS=80,
        CHUNK_OVERLAP_CHARS=20,
        MAX_UPLOAD_SIZE_BYTES=10_000,
        vectorstore_dir=tmp_path,
    )
    monkeypatch.setattr(routes_module, "get_settings", lambda: test_settings)

    client = TestClient(fastapi_app)
    response = client.post(
        "/upload",
        files={"file": ("sample.txt", b"alpha beta gamma " * 50, "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["indexing"]["embedding_provider"] == "local_dummy"
    assert payload["indexing"]["embedding_model"] == "local_dummy_48"
    assert payload["indexing"]["embedding_dimension"] == 48
    assert payload["indexing"]["added_vector_count"] == payload["chunk_count"]
    assert payload["indexing"]["total_vector_count"] == payload["chunk_count"]

    index, metadata = load_vector_store(vectorstore_dir=tmp_path)
    assert index is not None
    assert index.d == 48
    assert index.ntotal == payload["chunk_count"]
    assert metadata.embedding_provider == "local_dummy"
    assert metadata.entries[0].vector_id == 0
