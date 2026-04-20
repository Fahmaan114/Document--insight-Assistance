from __future__ import annotations

from typing import Any

import httpx


DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=5.0)
HEALTH_TIMEOUT = httpx.Timeout(5.0, connect=2.0)


class BackendClientError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def get_health(*, base_url: str) -> dict[str, Any]:
    return _request_json(
        method="GET",
        path="/health",
        base_url=base_url,
        timeout=HEALTH_TIMEOUT,
    )


def upload_document(
    *,
    base_url: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> dict[str, Any]:
    return _request_json(
        method="POST",
        path="/upload",
        base_url=base_url,
        timeout=DEFAULT_TIMEOUT,
        files={"file": (filename, file_bytes, content_type)},
    )


def get_index_status(*, base_url: str) -> dict[str, Any]:
    return _request_json(
        method="GET",
        path="/index-status",
        base_url=base_url,
        timeout=HEALTH_TIMEOUT,
    )


def reset_index(*, base_url: str) -> dict[str, Any]:
    return _request_json(
        method="DELETE",
        path="/index",
        base_url=base_url,
        timeout=DEFAULT_TIMEOUT,
    )


def answer_question(
    *,
    base_url: str,
    question: str,
    top_k: int,
) -> dict[str, Any]:
    return _request_json(
        method="POST",
        path="/answer",
        base_url=base_url,
        timeout=DEFAULT_TIMEOUT,
        json={"question": question, "top_k": top_k},
    )


def _request_json(
    *,
    method: str,
    path: str,
    base_url: str,
    timeout: httpx.Timeout,
    **request_kwargs: Any,
) -> dict[str, Any]:
    request_url = f"{base_url.rstrip('/')}{path}"

    try:
        response = httpx.request(
            method=method,
            url=request_url,
            timeout=timeout,
            **request_kwargs,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise BackendClientError(
            _extract_error_message(exc.response),
            status_code=exc.response.status_code,
        ) from exc
    except httpx.RequestError as exc:
        raise BackendClientError(
            f"Could not reach the backend at {base_url.rstrip('/')}. "
            "Make sure the FastAPI server is running and reachable.",
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise BackendClientError("Backend returned a non-JSON response.") from exc

    if not isinstance(payload, dict):
        raise BackendClientError("Backend returned an unexpected response shape.")
    return payload


def _extract_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()

    return f"Backend request failed with status {response.status_code}."
