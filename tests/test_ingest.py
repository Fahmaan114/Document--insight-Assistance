from rag.chunking import split_text_into_chunks
import pytest

from rag.ingest import (
    DocumentProcessingError,
    _extract_text_from_pdf,
    _extract_text_from_txt,
    build_document_key,
    ingest_document,
)


def test_extract_text_from_txt_cleans_obvious_noise() -> None:
    file_bytes = b"Hello   world\r\n\r\n\r\nSecond-\nline\x00"

    extracted_text = _extract_text_from_txt(file_bytes)

    assert extracted_text == "Hello world\n\nSecondline"


def test_extract_text_from_pdf_preserves_page_numbers() -> None:
    pdf_bytes = build_test_pdf(["First page text", "Second page text"])

    sections = _extract_text_from_pdf(pdf_bytes)

    assert len(sections) == 2
    assert sections[0].page_number == 1
    assert sections[1].page_number == 2
    assert "First page text" in sections[0].text
    assert "Second page text" in sections[1].text


def test_split_text_into_chunks_uses_overlap() -> None:
    chunks = split_text_into_chunks(
        "abcdefghijklmnopqrstuvwxyz",
        chunk_size=10,
        chunk_overlap=3,
    )

    assert chunks == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuvwx",
        "vwxyz",
    ]


def test_ingest_document_attaches_chunk_metadata() -> None:
    repeated_text = ("alpha beta gamma delta " * 80).encode("utf-8")

    document = ingest_document(
        filename="notes.txt",
        file_bytes=repeated_text,
        chunk_size=120,
        chunk_overlap=30,
        max_upload_size_bytes=1_000_000,
    )

    assert document.filename == "notes.txt"
    assert document.source_type == "txt"
    assert document.chunk_count == len(document.chunks)
    assert document.chunk_count > 1
    assert document.chunks[0].filename == "notes.txt"
    assert document.chunks[0].chunk_id.startswith("notes-")
    assert document.chunks[0].chunk_id.endswith("-p0000-c0001")
    assert document.chunks[0].source_type == "txt"
    assert document.chunks[0].page_number is None
    assert document.chunks[-1].chunk_id.startswith("notes-")


def test_ingest_document_rejects_txt_with_no_usable_text() -> None:
    with pytest.raises(DocumentProcessingError, match="does not contain usable text"):
        ingest_document(
            filename="blank.txt",
            file_bytes=b" \n\t\r\n ",
            chunk_size=120,
            chunk_overlap=30,
            max_upload_size_bytes=1_000_000,
        )


def test_ingest_document_rejects_pdf_with_no_usable_text() -> None:
    with pytest.raises(DocumentProcessingError, match="usable extractable text"):
        ingest_document(
            filename="blank.pdf",
            file_bytes=build_test_pdf(["", " "]),
            chunk_size=120,
            chunk_overlap=30,
            max_upload_size_bytes=1_000_000,
        )


def test_pdf_chunks_keep_consistent_page_numbers() -> None:
    document = ingest_document(
        filename="report.pdf",
        file_bytes=build_test_pdf(["PDF content " * 80]),
        chunk_size=120,
        chunk_overlap=30,
        max_upload_size_bytes=1_000_000,
    )

    assert document.chunk_count > 1
    assert all(chunk.page_number == 1 for chunk in document.chunks)
    assert all("-p0001-" in chunk.chunk_id for chunk in document.chunks)


def test_build_document_key_is_safe_and_traceable() -> None:
    document_key = build_document_key(
        filename="Quarterly Report 2026!.pdf",
        file_bytes=b"sample-bytes",
    )

    assert document_key.startswith("quarterly-report-2026-")
    assert len(document_key.split("-")[-1]) == 10


def build_test_pdf(page_texts: list[str]) -> bytes:
    objects: list[str] = []
    page_object_numbers: list[int] = []
    content_object_numbers: list[int] = []

    for page_index in range(len(page_texts)):
        page_object_numbers.append(3 + (page_index * 2))
        content_object_numbers.append(4 + (page_index * 2))

    font_object_number = 3 + (len(page_texts) * 2)
    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)

    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>")

    for page_index, page_text in enumerate(page_texts):
        escaped_text = _escape_pdf_text(page_text)
        content_stream = "\n".join(
            [
                "BT",
                "/F1 12 Tf",
                "72 720 Td",
                f"({escaped_text}) Tj",
                "ET",
            ]
        )
        page_object_number = page_object_numbers[page_index]
        content_object_number = content_object_numbers[page_index]

        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_object_number} 0 R "
                f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> >>"
            )
        )
        objects.append(
            (
                f"<< /Length {len(content_stream.encode('latin-1'))} >>\n"
                "stream\n"
                f"{content_stream}\n"
                "endstream"
            )
        )

    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf_parts = ["%PDF-1.4\n"]
    offsets: list[int] = [0]

    for object_number, object_body in enumerate(objects, start=1):
        offsets.append(sum(len(part.encode("latin-1")) for part in pdf_parts))
        pdf_parts.append(f"{object_number} 0 obj\n{object_body}\nendobj\n")

    xref_start = sum(len(part.encode("latin-1")) for part in pdf_parts)
    pdf_parts.append(f"xref\n0 {len(objects) + 1}\n")
    pdf_parts.append("0000000000 65535 f \n")

    for offset in offsets[1:]:
        pdf_parts.append(f"{offset:010d} 00000 n \n")

    pdf_parts.append(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF"
        )
    )

    return "".join(pdf_parts).encode("latin-1")


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
