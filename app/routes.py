from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from rag.embeddings import (
    get_vector_store_status,
    index_document_chunks,
    reset_vector_store,
)
from rag.ingest import DocumentProcessingError, ingest_document
from rag.prompt_builder import generate_grounded_answer
from rag.retrieve import retrieve_relevant_chunks
from rag.schemas import (
    AnswerRequest,
    RetrievalRequest,
    UploadDocumentResponse,
)


router = APIRouter()


@router.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "environment": settings.environment,
    }


@router.post("/upload", tags=["documents"])
async def upload_document(file: UploadFile = File(...)) -> dict:
    settings = get_settings()

    try:
        file_bytes = await _read_upload_bytes(
            file=file,
            max_upload_size_bytes=settings.max_upload_size_bytes,
        )
        document = ingest_document(
            filename=file.filename or "",
            file_bytes=file_bytes,
            chunk_size=settings.chunk_size_chars,
            chunk_overlap=settings.chunk_overlap_chars,
            max_upload_size_bytes=settings.max_upload_size_bytes,
        )
        indexing = index_document_chunks(
            chunks=document.chunks,
            vectorstore_dir=settings.vectorstore_dir,
            embedding_provider=settings.embedding_provider,
            embedding_model=_select_embedding_model(settings),
            batch_size=settings.embedding_batch_size,
            api_key=settings.openai_api_key,
            local_dummy_dimension=settings.local_dummy_dimension,
        )
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    finally:
        await file.close()

    response = UploadDocumentResponse(
        **document.model_dump(),
        indexing=indexing,
    )
    return response.model_dump()


@router.get("/index-status", tags=["documents"])
async def index_status() -> dict:
    settings = get_settings()

    try:
        status = get_vector_store_status(vectorstore_dir=settings.vectorstore_dir)
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return status.model_dump()


@router.delete("/index", tags=["documents"])
async def clear_index() -> dict:
    settings = get_settings()

    try:
        reset_response = reset_vector_store(vectorstore_dir=settings.vectorstore_dir)
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return reset_response.model_dump()


@router.post("/retrieve", tags=["documents"])
async def retrieve_chunks(payload: RetrievalRequest) -> dict:
    settings = get_settings()
    top_k = payload.top_k or settings.retrieval_top_k

    try:
        retrieval = retrieve_relevant_chunks(
            question=payload.question,
            vectorstore_dir=settings.vectorstore_dir,
            embedding_provider=settings.embedding_provider,
            embedding_model=_select_embedding_model(settings),
            batch_size=settings.embedding_batch_size,
            api_key=settings.openai_api_key,
            local_dummy_dimension=settings.local_dummy_dimension,
            top_k=top_k,
            min_score=settings.retrieval_min_score,
        )
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return retrieval.model_dump()


@router.post("/answer", tags=["documents"])
async def answer_question(payload: AnswerRequest) -> dict:
    settings = get_settings()
    top_k = payload.top_k or settings.retrieval_top_k

    try:
        retrieval = retrieve_relevant_chunks(
            question=payload.question,
            vectorstore_dir=settings.vectorstore_dir,
            embedding_provider=settings.embedding_provider,
            embedding_model=_select_embedding_model(settings),
            batch_size=settings.embedding_batch_size,
            api_key=settings.openai_api_key,
            local_dummy_dimension=settings.local_dummy_dimension,
            top_k=top_k,
            min_score=settings.retrieval_min_score,
        )
        answer_response = generate_grounded_answer(
            question=payload.question,
            retrieved_chunks=retrieval.results,
            answer_provider=settings.answer_provider,
            max_sources=settings.answer_max_sources,
            snippet_chars=settings.answer_snippet_chars,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_answer_model,
        )
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    return answer_response.model_dump()


async def _read_upload_bytes(*, file: UploadFile, max_upload_size_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total_size = 0

    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break

        total_size += len(chunk)
        if total_size > max_upload_size_bytes:
            raise DocumentProcessingError(
                f"File is too large. Maximum allowed size is {max_upload_size_bytes} bytes.",
                status_code=413,
            )

        chunks.append(chunk)

    return b"".join(chunks)


def _select_embedding_model(settings) -> str:
    if settings.embedding_provider == "openai":
        return settings.openai_embedding_model
    if settings.embedding_provider == "local":
        return settings.local_embedding_model
    return f"local_dummy_{settings.local_dummy_dimension}"
