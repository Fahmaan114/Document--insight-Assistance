import httpx
import pytest

from frontend.api_client import (
    BackendClientError,
    answer_question,
    get_index_status,
    reset_index,
    upload_document,
)


def test_upload_document_posts_multipart_file(monkeypatch) -> None:
    captured_request: dict = {}

    def fake_request(method, url, timeout, **kwargs):  # noqa: ANN001
        captured_request["method"] = method
        captured_request["url"] = url
        captured_request["files"] = kwargs["files"]
        return httpx.Response(
            200,
            request=httpx.Request(method, url),
            json={"filename": "notes.txt", "chunk_count": 1},
        )

    monkeypatch.setattr("frontend.api_client.httpx.request", fake_request)

    response = upload_document(
        base_url="http://127.0.0.1:8000/",
        filename="notes.txt",
        file_bytes=b"hello world",
        content_type="text/plain",
    )

    assert captured_request["method"] == "POST"
    assert captured_request["url"] == "http://127.0.0.1:8000/upload"
    assert captured_request["files"]["file"] == ("notes.txt", b"hello world", "text/plain")
    assert response["filename"] == "notes.txt"


def test_answer_question_surfaces_backend_detail_message(monkeypatch) -> None:
    def fake_request(method, url, timeout, **kwargs):  # noqa: ANN001
        return httpx.Response(
            503,
            request=httpx.Request(method, url),
            json={"detail": "Vector store is empty."},
        )

    monkeypatch.setattr("frontend.api_client.httpx.request", fake_request)

    with pytest.raises(BackendClientError, match="Vector store is empty."):
        answer_question(
            base_url="http://127.0.0.1:8000",
            question="What file types does the system accept?",
            top_k=3,
        )


def test_answer_question_surfaces_connection_failures(monkeypatch) -> None:
    def fake_request(method, url, timeout, **kwargs):  # noqa: ANN001
        raise httpx.ConnectError("connection failed", request=httpx.Request(method, url))

    monkeypatch.setattr("frontend.api_client.httpx.request", fake_request)

    with pytest.raises(BackendClientError, match="Could not reach the backend"):
        answer_question(
            base_url="http://127.0.0.1:8000",
            question="What file types does the system accept?",
            top_k=3,
        )


def test_get_index_status_requests_status_endpoint(monkeypatch) -> None:
    captured_request: dict = {}

    def fake_request(method, url, timeout, **kwargs):  # noqa: ANN001
        captured_request["method"] = method
        captured_request["url"] = url
        return httpx.Response(
            200,
            request=httpx.Request(method, url),
            json={"indexed": False, "vector_count": 0},
        )

    monkeypatch.setattr("frontend.api_client.httpx.request", fake_request)

    response = get_index_status(base_url="http://127.0.0.1:8000/")

    assert captured_request["method"] == "GET"
    assert captured_request["url"] == "http://127.0.0.1:8000/index-status"
    assert response["indexed"] is False


def test_reset_index_requests_delete_endpoint(monkeypatch) -> None:
    captured_request: dict = {}

    def fake_request(method, url, timeout, **kwargs):  # noqa: ANN001
        captured_request["method"] = method
        captured_request["url"] = url
        return httpx.Response(
            200,
            request=httpx.Request(method, url),
            json={"cleared": True, "deleted_artifact_count": 2},
        )

    monkeypatch.setattr("frontend.api_client.httpx.request", fake_request)

    response = reset_index(base_url="http://127.0.0.1:8000/")

    assert captured_request["method"] == "DELETE"
    assert captured_request["url"] == "http://127.0.0.1:8000/index"
    assert response["cleared"] is True
