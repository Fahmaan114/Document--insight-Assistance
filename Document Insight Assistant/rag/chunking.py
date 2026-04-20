from rag.schemas import DocumentChunk, ExtractedSection


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    normalized_text = text.strip()
    if not normalized_text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(normalized_text):
        max_end = min(start + chunk_size, len(normalized_text))
        end = _find_chunk_end(normalized_text, start, max_end)
        chunk = normalized_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized_text):
            break

        next_start = max(end - chunk_overlap, start + 1)
        start = _advance_past_whitespace(normalized_text, next_start)

    return chunks


def build_document_chunks(
    *,
    document_key: str,
    filename: str,
    source_type: str,
    sections: list[ExtractedSection],
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    document_chunks: list[DocumentChunk] = []
    next_chunk_number = 1

    for section in sections:
        for chunk_text in split_text_into_chunks(
            section.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            document_chunks.append(
                DocumentChunk(
                    filename=filename,
                    chunk_id=_build_chunk_id(
                        document_key=document_key,
                        chunk_number=next_chunk_number,
                        page_number=section.page_number,
                    ),
                    chunk_text=chunk_text,
                    source_type=source_type,
                    page_number=section.page_number,
                )
            )
            next_chunk_number += 1

    return document_chunks


def _find_chunk_end(text: str, start: int, max_end: int) -> int:
    if max_end >= len(text):
        return len(text)

    minimum_end = start + max(1, int((max_end - start) * 0.6))
    preferred_breaks = {"\n", " ", "\t"}

    for index in range(max_end, minimum_end, -1):
        if text[index - 1] in preferred_breaks:
            return index

    return max_end


def _advance_past_whitespace(text: str, start: int) -> int:
    while start < len(text) and text[start].isspace():
        start += 1
    return start


def _build_chunk_id(*, document_key: str, chunk_number: int, page_number: int | None) -> str:
    page_marker = page_number if page_number is not None else 0
    return f"{document_key}-p{page_marker:04d}-c{chunk_number:04d}"
