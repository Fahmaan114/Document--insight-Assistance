from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ExtractedSection:
    text: str
    page_number: int | None = None


class DocumentChunk(BaseModel):
    filename: str
    chunk_id: str = Field(
        ...,
        description=(
            "Stable chunk identifier composed from a safe document key, a 1-based PDF "
            "page marker or 0 for TXT files, and the chunk sequence number."
        ),
    )
    chunk_text: str
    source_type: str
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="1-based PDF page number for PDF chunks; null for TXT chunks.",
    )


class IngestedDocument(BaseModel):
    filename: str
    source_type: str
    chunk_count: int = Field(..., ge=0)
    chunks: list[DocumentChunk]


class IndexedChunkMetadata(DocumentChunk):
    vector_id: int = Field(
        ...,
        ge=0,
        description="0-based FAISS row id matching the vector position in the local index.",
    )


class VectorStoreMetadata(BaseModel):
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int = Field(..., gt=0)
    index_type: str
    vector_count: int = Field(..., ge=0)
    entries: list[IndexedChunkMetadata]


class IndexingArtifacts(BaseModel):
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int = Field(..., gt=0)
    index_path: str
    metadata_path: str
    added_vector_count: int = Field(..., ge=0)
    total_vector_count: int = Field(..., ge=0)


class UploadDocumentResponse(BaseModel):
    filename: str
    source_type: str
    chunk_count: int = Field(..., ge=0)
    chunks: list[DocumentChunk]
    indexing: IndexingArtifacts


class IndexStatusResponse(BaseModel):
    indexed: bool
    message: str
    vector_count: int = Field(..., ge=0)
    chunk_count: int = Field(..., ge=0)
    document_count: int = Field(..., ge=0)
    embedding_provider: str | None = None
    embedding_model: str | None = None
    filenames: list[str] = Field(default_factory=list)


class ResetIndexResponse(BaseModel):
    cleared: bool
    deleted_artifact_count: int = Field(..., ge=0)
    message: str


class RetrievalRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1)


class RetrievedChunk(IndexedChunkMetadata):
    rank: int = Field(..., ge=1)
    score: float


class RetrievalResponse(BaseModel):
    question: str
    top_k: int = Field(..., ge=1)
    no_results: bool
    message: str
    results: list[RetrievedChunk]


class AnswerRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1)


class SourceSnippet(BaseModel):
    filename: str
    chunk_id: str
    source_type: str
    page_number: int | None = Field(default=None, ge=1)
    rank: int = Field(..., ge=1)
    score: float
    snippet: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    answer_supported: bool
    message: str
    sources: list[SourceSnippet]
