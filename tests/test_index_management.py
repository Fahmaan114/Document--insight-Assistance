from pathlib import Path

from fastapi.testclient import TestClient

import app.routes as routes_module
from app.config import Settings
from app.main import app as fastapi_app
from rag.embeddings import get_vectorstore_paths


client = TestClient(fastapi_app)


def _configure_test_settings(monkeypatch, tmp_path, **overrides) -> None:
    config_values = {
        "EMBEDDING_PROVIDER": "local_dummy",
        "OPENAI_API_KEY": "",
        "OPENAI_EMBEDDING_MODEL": "test-embedding-model",
        "LOCAL_EMBEDDING_MODEL": "test-local-model",
        "LOCAL_DUMMY_DIMENSION": 64,
        "EMBEDDING_BATCH_SIZE": 2,
        "CHUNK_SIZE_CHARS": 1000,
        "CHUNK_OVERLAP_CHARS": 200,
        "MAX_UPLOAD_SIZE_BYTES": 10_485_760,
        "vectorstore_dir": tmp_path,
    }
    config_values.update(overrides)
    test_settings = Settings(**config_values)
    monkeypatch.setattr(routes_module, "get_settings", lambda: test_settings)


def test_index_status_reports_empty_store(tmp_path: Path, monkeypatch) -> None:
    _configure_test_settings(monkeypatch, tmp_path)

    response = client.get("/index-status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["indexed"] is False
    assert payload["vector_count"] == 0
    assert payload["chunk_count"] == 0
    assert payload["document_count"] == 0
    assert payload["filenames"] == []


def test_reset_index_clears_artifacts_and_retrieval_behaves_empty(tmp_path: Path, monkeypatch) -> None:
    _configure_test_settings(monkeypatch, tmp_path)

    upload_response = client.post(
        "/upload",
        files={"file": ("sample.txt", b"alpha beta gamma " * 60, "text/plain")},
    )
    assert upload_response.status_code == 200

    status_response = client.get("/index-status")
    assert status_response.status_code == 200
    assert status_response.json()["indexed"] is True
    assert status_response.json()["document_count"] == 1

    index_path, metadata_path = get_vectorstore_paths(tmp_path)
    assert index_path.exists() is True
    assert metadata_path.exists() is True

    reset_response = client.delete("/index")
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["cleared"] is True
    assert reset_payload["deleted_artifact_count"] == 2

    assert index_path.exists() is False
    assert metadata_path.exists() is False

    empty_status_response = client.get("/index-status")
    assert empty_status_response.status_code == 200
    assert empty_status_response.json()["indexed"] is False

    retrieve_response = client.post(
        "/retrieve",
        json={"question": "alpha beta", "top_k": 3},
    )
    assert retrieve_response.status_code == 200
    assert retrieve_response.json()["no_results"] is True
    assert "No indexed document chunks are available yet" in retrieve_response.json()["message"]
