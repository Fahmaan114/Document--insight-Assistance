import hashlib
import json
from dataclasses import dataclass
import os
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from rag.ingest import DocumentProcessingError
from rag.schemas import (
    DocumentChunk,
    IndexStatusResponse,
    IndexedChunkMetadata,
    IndexingArtifacts,
    ResetIndexResponse,
    VectorStoreMetadata,
)


INDEX_FILENAME = "chunks.faiss"
METADATA_FILENAME = "chunks_metadata.json"
INDEX_TYPE = "IndexFlatIP"


@dataclass(frozen=True)
class EmbeddingResult:
    provider: str
    model: str
    vectors: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])


def create_openai_client(*, api_key: str) -> OpenAI:
    if not api_key:
        raise DocumentProcessingError(
            "OpenAI embeddings are not available because OPENAI_API_KEY is not configured.",
            status_code=503,
        )

    return OpenAI(api_key=api_key)


def embed_chunks(
    *,
    chunks: list[DocumentChunk],
    provider: str,
    embedding_model: str,
    batch_size: int,
    api_key: str = "",
    local_dummy_dimension: int = 384,
    client: OpenAI | None = None,
) -> EmbeddingResult:
    if not chunks:
        raise DocumentProcessingError("No chunks are available for embedding.", status_code=400)

    return embed_texts(
        texts=[chunk.chunk_text for chunk in chunks],
        provider=provider,
        embedding_model=embedding_model,
        batch_size=batch_size,
        api_key=api_key,
        local_dummy_dimension=local_dummy_dimension,
        client=client,
    )


def embed_texts(
    *,
    texts: list[str],
    provider: str,
    embedding_model: str,
    batch_size: int,
    api_key: str = "",
    local_dummy_dimension: int = 384,
    client: OpenAI | None = None,
) -> EmbeddingResult:
    if not texts:
        raise DocumentProcessingError("No texts are available for embedding.", status_code=400)

    if provider == "openai":
        vectors = _embed_with_openai(
            texts=texts,
            embedding_model=embedding_model,
            batch_size=batch_size,
            api_key=api_key,
            client=client,
        )
        return EmbeddingResult(provider="openai", model=embedding_model, vectors=vectors)

    if provider == "local":
        try:
            vectors = _embed_with_sentence_transformers(
                texts=texts,
                model_name=embedding_model,
                batch_size=batch_size,
            )
            return EmbeddingResult(
                provider="local",
                model=embedding_model,
                vectors=vectors,
            )
        except (DocumentProcessingError, Exception) as exc:
            if _is_development_environment():
                vectors = _embed_with_local_dummy(
                    texts=texts,
                    dimension=local_dummy_dimension,
                )
                return EmbeddingResult(
                    provider="local_dummy",
                    model=f"local_dummy_{local_dummy_dimension}",
                    vectors=vectors,
                )
            if isinstance(exc, DocumentProcessingError):
                raise exc
            raise DocumentProcessingError(
                (
                    "Local embeddings could not be initialized. "
                    "Use EMBEDDING_PROVIDER=local_dummy for development."
                ),
                status_code=503,
            ) from exc

    if provider == "local_dummy":
        vectors = _embed_with_local_dummy(
            texts=texts,
            dimension=local_dummy_dimension,
        )
        return EmbeddingResult(
            provider="local_dummy",
            model=f"local_dummy_{local_dummy_dimension}",
            vectors=vectors,
        )

    raise DocumentProcessingError(
        f"Unsupported embedding provider: {provider}.",
        status_code=500,
    )


def index_document_chunks(
    *,
    chunks: list[DocumentChunk],
    vectorstore_dir: Path,
    embedding_provider: str,
    embedding_model: str,
    batch_size: int,
    api_key: str,
    local_dummy_dimension: int,
    client: OpenAI | None = None,
) -> IndexingArtifacts:
    if not chunks:
        raise DocumentProcessingError("No chunks are available for indexing.", status_code=400)

    embedding_result = embed_chunks(
        chunks=chunks,
        provider=embedding_provider,
        embedding_model=embedding_model,
        batch_size=batch_size,
        client=client,
        api_key=api_key,
        local_dummy_dimension=local_dummy_dimension,
    )
    vectors = embedding_result.vectors
    faiss.normalize_L2(vectors)

    index, metadata = load_vector_store(vectorstore_dir=vectorstore_dir)
    if metadata.entries:
        if metadata.embedding_provider != embedding_result.provider:
            raise DocumentProcessingError(
                (
                    "The existing local index was created with a different embedding provider. "
                    "Clear the vector store before switching providers."
                ),
                status_code=409,
            )
        if metadata.embedding_model != embedding_result.model:
            raise DocumentProcessingError(
                (
                    "The existing local index was created with a different embedding model. "
                    "Clear the vector store before switching models."
                ),
                status_code=409,
            )

    if index is None:
        index = faiss.IndexFlatIP(embedding_result.dimension)
        metadata = VectorStoreMetadata(
            embedding_provider=embedding_result.provider,
            embedding_model=embedding_result.model,
            embedding_dimension=embedding_result.dimension,
            index_type=INDEX_TYPE,
            vector_count=0,
            entries=[],
        )
    elif index.d != embedding_result.dimension:
        raise DocumentProcessingError(
            "Embedding dimension mismatch with the existing local FAISS index.",
            status_code=409,
        )

    start_vector_id = metadata.vector_count
    indexed_entries = [
        IndexedChunkMetadata(
            vector_id=start_vector_id + offset,
            **chunk.model_dump(),
        )
        for offset, chunk in enumerate(chunks)
    ]

    index.add(vectors)
    metadata.entries.extend(indexed_entries)
    metadata.vector_count = len(metadata.entries)
    metadata.embedding_provider = embedding_result.provider
    metadata.embedding_model = embedding_result.model
    metadata.embedding_dimension = embedding_result.dimension
    metadata.index_type = INDEX_TYPE

    save_vector_store(
        index=index,
        metadata=metadata,
        vectorstore_dir=vectorstore_dir,
    )

    index_path, metadata_path = get_vectorstore_paths(vectorstore_dir)
    return IndexingArtifacts(
        embedding_provider=embedding_result.provider,
        embedding_model=embedding_result.model,
        embedding_dimension=embedding_result.dimension,
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        added_vector_count=len(chunks),
        total_vector_count=metadata.vector_count,
    )


def load_vector_store(*, vectorstore_dir: Path) -> tuple[faiss.Index | None, VectorStoreMetadata]:
    index_path, metadata_path = get_vectorstore_paths(vectorstore_dir)

    if index_path.exists() != metadata_path.exists():
        raise DocumentProcessingError(
            "Local vector store artifacts are incomplete. Expected both index and metadata files.",
            status_code=500,
        )

    if not index_path.exists():
        return (
            None,
            VectorStoreMetadata(
                embedding_provider="",
                embedding_model="",
                embedding_dimension=1,
                index_type=INDEX_TYPE,
                vector_count=0,
                entries=[],
            ),
        )

    index = faiss.read_index(str(index_path))
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = VectorStoreMetadata.model_validate(metadata_payload)
    _validate_vector_store_alignment(index=index, metadata=metadata)
    return index, metadata


def save_vector_store(
    *,
    index: faiss.Index,
    metadata: VectorStoreMetadata,
    vectorstore_dir: Path,
) -> None:
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    _validate_vector_store_alignment(index=index, metadata=metadata)

    index_path, metadata_path = get_vectorstore_paths(vectorstore_dir)
    faiss.write_index(index, str(index_path))
    metadata_path.write_text(
        metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )


def get_vectorstore_paths(vectorstore_dir: Path) -> tuple[Path, Path]:
    return (
        vectorstore_dir / INDEX_FILENAME,
        vectorstore_dir / METADATA_FILENAME,
    )


def get_vector_store_status(*, vectorstore_dir: Path) -> IndexStatusResponse:
    index, metadata = load_vector_store(vectorstore_dir=vectorstore_dir)
    if index is None or metadata.vector_count == 0:
        return IndexStatusResponse(
            indexed=False,
            message="No documents are currently indexed.",
            vector_count=0,
            chunk_count=0,
            document_count=0,
            embedding_provider=None,
            embedding_model=None,
            filenames=[],
        )

    filenames = list(dict.fromkeys(entry.filename for entry in metadata.entries))
    return IndexStatusResponse(
        indexed=True,
        message=f"{len(filenames)} document(s) currently indexed.",
        vector_count=metadata.vector_count,
        chunk_count=len(metadata.entries),
        document_count=len(filenames),
        embedding_provider=metadata.embedding_provider,
        embedding_model=metadata.embedding_model,
        filenames=filenames,
    )


def reset_vector_store(*, vectorstore_dir: Path) -> ResetIndexResponse:
    index_path, metadata_path = get_vectorstore_paths(vectorstore_dir)
    deleted_artifact_count = 0

    for artifact_path in (index_path, metadata_path):
        if artifact_path.exists():
            artifact_path.unlink()
            deleted_artifact_count += 1

    return ResetIndexResponse(
        cleared=True,
        deleted_artifact_count=deleted_artifact_count,
        message="Local indexed data was cleared. Upload a document to start a fresh demo.",
    )


def _embed_with_openai(
    *,
    texts: list[str],
    embedding_model: str,
    batch_size: int,
    api_key: str,
    client: OpenAI | None,
) -> np.ndarray:
    try:
        embedding_client = client or create_openai_client(api_key=api_key)
        all_vectors: list[list[float]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            response = embedding_client.embeddings.create(
                model=embedding_model,
                input=batch,
            )
            ordered_embeddings = sorted(response.data, key=lambda item: item.index)
            all_vectors.extend(item.embedding for item in ordered_embeddings)
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError(
            "OpenAI embedding generation failed. Check your API key, billing, and model configuration.",
            status_code=502,
        ) from exc

    return _coerce_vectors(vectors=all_vectors, expected_count=len(texts))


def _embed_with_sentence_transformers(
    *,
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise DocumentProcessingError(
            (
                "Local sentence-transformers embeddings are not available. "
                "Install the optional local embedding dependency or use EMBEDDING_PROVIDER=local_dummy."
            ),
            status_code=503,
        ) from exc

    try:
        model = SentenceTransformer(model_name)
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise DocumentProcessingError(
            (
                "Local sentence-transformers embeddings could not be initialized. "
                "If the model is not available locally, use EMBEDDING_PROVIDER=local_dummy for development."
            ),
            status_code=503,
        ) from exc

    return _coerce_vectors(vectors=vectors, expected_count=len(texts))


def _embed_with_local_dummy(*, texts: list[str], dimension: int) -> np.ndarray:
    if dimension <= 0:
        raise DocumentProcessingError(
            "LOCAL_DUMMY_DIMENSION must be greater than 0.",
            status_code=500,
        )

    vectors = np.zeros((len(texts), dimension), dtype="float32")
    for row_index, text in enumerate(texts):
        for token in text.split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            column_index = int.from_bytes(digest[:4], "big") % dimension
            value = ((int.from_bytes(digest[4:8], "big") % 2000) / 1000.0) - 1.0
            vectors[row_index, column_index] += value

        if not np.any(vectors[row_index]):
            vectors[row_index, row_index % dimension] = 1.0

    return _coerce_vectors(vectors=vectors, expected_count=len(texts))


def _coerce_vectors(*, vectors, expected_count: int) -> np.ndarray:
    array = np.asarray(vectors, dtype="float32")
    if array.ndim != 2 or array.shape[0] != expected_count or array.shape[1] <= 0:
        raise DocumentProcessingError(
            "Embedding generation returned an unexpected vector shape.",
            status_code=500,
        )
    return array


def _validate_vector_store_alignment(
    *,
    index: faiss.Index,
    metadata: VectorStoreMetadata,
) -> None:
    if index.ntotal != metadata.vector_count:
        raise DocumentProcessingError(
            "Vector store metadata count does not match the FAISS index size.",
            status_code=500,
        )

    if metadata.vector_count != len(metadata.entries):
        raise DocumentProcessingError(
            "Vector store metadata entries are out of sync with the stored vector count.",
            status_code=500,
        )

    if metadata.embedding_dimension != index.d:
        raise DocumentProcessingError(
            "Stored vector metadata does not match the FAISS index dimension.",
            status_code=500,
        )

    for expected_vector_id, entry in enumerate(metadata.entries):
        if entry.vector_id != expected_vector_id:
            raise DocumentProcessingError(
                "Vector metadata is out of order and can no longer be mapped safely.",
                status_code=500,
            )


def _is_development_environment() -> bool:
    return os.getenv("ENVIRONMENT", "development").lower() != "production"
