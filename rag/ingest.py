from io import BytesIO
from pathlib import Path
import re
import hashlib

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag.chunking import build_document_chunks
from rag.schemas import ExtractedSection, IngestedDocument


SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".txt": "txt",
}


class DocumentProcessingError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def ingest_document(
    *,
    filename: str,
    file_bytes: bytes,
    chunk_size: int,
    chunk_overlap: int,
    max_upload_size_bytes: int,
) -> IngestedDocument:
    safe_filename = _validate_filename(filename)
    _validate_file_bytes(file_bytes, max_upload_size_bytes=max_upload_size_bytes)

    source_type = SUPPORTED_EXTENSIONS[Path(safe_filename).suffix.lower()]
    document_key = build_document_key(filename=safe_filename, file_bytes=file_bytes)
    sections = _extract_sections(file_bytes=file_bytes, source_type=source_type)
    chunks = build_document_chunks(
        document_key=document_key,
        filename=safe_filename,
        source_type=source_type,
        sections=sections,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise DocumentProcessingError(
            "The uploaded document did not produce any usable text chunks.",
            status_code=400,
        )

    return IngestedDocument(
        filename=safe_filename,
        source_type=source_type,
        chunk_count=len(chunks),
        chunks=chunks,
    )


def _validate_filename(filename: str | None) -> str:
    if not filename:
        raise DocumentProcessingError("A filename is required.", status_code=400)

    safe_name = Path(filename).name
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentProcessingError(
            f"Unsupported file type. Allowed extensions: {supported}.",
            status_code=415,
        )

    return safe_name


def _validate_file_bytes(file_bytes: bytes, *, max_upload_size_bytes: int) -> None:
    if not file_bytes:
        raise DocumentProcessingError("The uploaded file is empty.", status_code=400)

    if len(file_bytes) > max_upload_size_bytes:
        raise DocumentProcessingError(
            f"File is too large. Maximum allowed size is {max_upload_size_bytes} bytes.",
            status_code=413,
        )


def _extract_sections(*, file_bytes: bytes, source_type: str) -> list[ExtractedSection]:
    if source_type == "txt":
        text = _extract_text_from_txt(file_bytes)
        return [ExtractedSection(text=text, page_number=None)]

    if source_type == "pdf":
        return _extract_text_from_pdf(file_bytes)

    raise DocumentProcessingError("Unsupported source type.", status_code=415)


def _extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        decoded_text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentProcessingError(
            "TXT files must be valid UTF-8 text.",
            status_code=400,
        ) from exc

    cleaned_text = clean_extracted_text(decoded_text)
    if not cleaned_text:
        raise DocumentProcessingError(
            "The TXT file does not contain usable text after extraction.",
            status_code=400,
        )

    return cleaned_text


def _extract_text_from_pdf(file_bytes: bytes) -> list[ExtractedSection]:
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except PdfReadError as exc:
        raise DocumentProcessingError(
            "The PDF file could not be read.",
            status_code=400,
        ) from exc
    except Exception as exc:
        raise DocumentProcessingError(
            "The uploaded PDF is invalid or corrupted.",
            status_code=400,
        ) from exc

    sections: list[ExtractedSection] = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text() or ""
        cleaned_text = clean_extracted_text(extracted_text)
        if cleaned_text:
            sections.append(ExtractedSection(text=cleaned_text, page_number=page_number))

    if not sections:
        raise DocumentProcessingError(
            "The PDF file does not contain usable extractable text.",
            status_code=400,
        )

    return sections


def clean_extracted_text(text: str) -> str:
    cleaned_text = text.replace("\x00", "")
    cleaned_text = cleaned_text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_text = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned_text)
    cleaned_text = re.sub(r"[^\S\n]{2,}", " ", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+\n", "\n", cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def build_document_key(*, filename: str, file_bytes: bytes) -> str:
    stem = Path(filename).stem.lower()
    safe_stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or "document"
    safe_stem = safe_stem[:40]
    digest = hashlib.sha256(file_bytes).hexdigest()[:10]
    return f"{safe_stem}-{digest}"
