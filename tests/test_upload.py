from fastapi.testclient import TestClient

import app.routes as routes_module
from app.config import Settings
from app.main import app as fastapi_app
from tests.test_ingest import build_test_pdf


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


def test_upload_txt_returns_chunked_document(tmp_path, monkeypatch) -> None:
    _configure_test_settings(monkeypatch, tmp_path)
    response = client.post(
        "/upload",
        files={"file": ("sample.txt", b"alpha beta gamma " * 100, "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "sample.txt"
    assert payload["source_type"] == "txt"
    assert payload["chunk_count"] >= 1
    assert payload["indexing"]["embedding_provider"] == "local_dummy"
    assert payload["indexing"]["embedding_dimension"] == 64
    assert payload["indexing"]["added_vector_count"] == payload["chunk_count"]
    assert payload["chunks"][0]["page_number"] is None
    assert set(payload["chunks"][0]) == {
        "filename",
        "chunk_id",
        "chunk_text",
        "source_type",
        "page_number",
    }


def test_upload_pdf_returns_page_metadata(tmp_path, monkeypatch) -> None:
    _configure_test_settings(monkeypatch, tmp_path)
    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", build_test_pdf(["PDF page text"]), "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "sample.pdf"
    assert payload["source_type"] == "pdf"
    assert payload["indexing"]["total_vector_count"] >= payload["chunk_count"]
    assert payload["chunks"][0]["page_number"] == 1
    assert "-p0001-c0001" in payload["chunks"][0]["chunk_id"]
    assert "PDF page text" in payload["chunks"][0]["chunk_text"]


def test_upload_rejects_unsupported_file_type(tmp_path, monkeypatch) -> None:
    _configure_test_settings(monkeypatch, tmp_path)
    response = client.post(
        "/upload",
        files={"file": ("notes.docx", b"not supported", "application/octet-stream")},
    )

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_rejects_invalid_pdf(tmp_path, monkeypatch) -> None:
    _configure_test_settings(monkeypatch, tmp_path)
    response = client.post(
        "/upload",
        files={"file": ("broken.pdf", b"not a real pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_rejects_whitespace_only_txt(tmp_path, monkeypatch) -> None:
    _configure_test_settings(monkeypatch, tmp_path)
    response = client.post(
        "/upload",
        files={"file": ("blank.txt", b" \n\t\r\n ", "text/plain")},
    )

    assert response.status_code == 400
    assert "usable text" in response.json()["detail"]


def test_upload_enforces_max_upload_size_cleanly(tmp_path, monkeypatch) -> None:
    _configure_test_settings(
        monkeypatch,
        tmp_path,
        MAX_UPLOAD_SIZE_BYTES=8,
    )

    response = client.post(
        "/upload",
        files={"file": ("too-big.txt", b"123456789", "text/plain")},
    )

    assert response.status_code == 413
    assert "Maximum allowed size is 8 bytes" in response.json()["detail"]


def test_upload_returns_clean_openai_failure(tmp_path, monkeypatch, fake_openai_client) -> None:
    class FailingEmbeddingsAPI:
        def create(self, *, model: str, input: list[str]):
            raise RuntimeError("billing failure")

    fake_openai_client.embeddings = FailingEmbeddingsAPI()
    monkeypatch.setattr(
        "rag.embeddings.create_openai_client",
        lambda api_key: fake_openai_client,
    )
    _configure_test_settings(
        monkeypatch,
        tmp_path,
        EMBEDDING_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
    )

    response = client.post(
        "/upload",
        files={"file": ("sample.txt", b"alpha beta gamma " * 30, "text/plain")},
    )

    assert response.status_code == 502
    assert "OpenAI embedding generation failed" in response.json()["detail"]
