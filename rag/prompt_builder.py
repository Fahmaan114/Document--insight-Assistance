import re
from dataclasses import dataclass

from rag.embeddings import create_openai_client
from rag.ingest import DocumentProcessingError
from rag.schemas import AnswerResponse, RetrievedChunk, SourceSnippet


INSUFFICIENT_CONTEXT_ANSWER = "I couldn't answer that from the uploaded documents."

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

TERM_NORMALIZATIONS = {
    "accept": "support",
    "accepted": "support",
    "accepting": "support",
    "accepts": "support",
    "file": "file",
    "files": "file",
    "require": "require",
    "required": "require",
    "requires": "require",
    "requiring": "require",
    "type": "type",
    "types": "type",
}


@dataclass(frozen=True)
class _SentenceCandidate:
    rank: int
    score: float
    filename: str
    chunk_id: str
    source_type: str
    page_number: int | None
    snippet: str


def build_grounded_prompt(
    *,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
) -> list[dict[str, str]]:
    context_blocks = []
    for chunk in retrieved_chunks:
        page_label = chunk.page_number if chunk.page_number is not None else "n/a"
        context_blocks.append(
            (
                f"[Chunk {chunk.chunk_id}] "
                f"file={chunk.filename}; page={page_label}; rank={chunk.rank}; score={chunk.score:.4f}\n"
                f"{chunk.chunk_text}"
            )
        )

    system_prompt = (
        "You answer questions using only the provided document context. "
        "Do not use outside knowledge, do not guess, and do not invent facts. "
        f"If the context is insufficient, reply with exactly: {INSUFFICIENT_CONTEXT_ANSWER}"
    )
    user_prompt = (
        f"Question:\n{question.strip()}\n\n"
        "Document context:\n"
        f"{'\n\n'.join(context_blocks)}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_grounded_answer(
    *,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    answer_provider: str,
    max_sources: int,
    snippet_chars: int,
    openai_api_key: str = "",
    openai_model: str = "",
) -> AnswerResponse:
    normalized_question = question.strip()
    if not normalized_question:
        raise DocumentProcessingError("A question is required for answer generation.", status_code=400)

    if not retrieved_chunks:
        return _build_unsupported_answer(
            question=normalized_question,
            message="No useful retrieved context was available for this question.",
        )

    if answer_provider == "local":
        return _generate_local_grounded_answer(
            question=normalized_question,
            retrieved_chunks=retrieved_chunks,
            max_sources=max_sources,
            snippet_chars=snippet_chars,
        )

    if answer_provider == "openai":
        return _generate_openai_grounded_answer(
            question=normalized_question,
            retrieved_chunks=retrieved_chunks,
            max_sources=max_sources,
            snippet_chars=snippet_chars,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
        )

    raise DocumentProcessingError(
        f"Unsupported answer provider: {answer_provider}.",
        status_code=500,
    )


def _generate_local_grounded_answer(
    *,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    max_sources: int,
    snippet_chars: int,
) -> AnswerResponse:
    question_terms = _extract_keywords(question)
    candidates = _select_sentence_candidates(
        question_terms=question_terms,
        retrieved_chunks=retrieved_chunks,
        max_sources=max_sources,
        snippet_chars=snippet_chars,
    )

    if not candidates:
        candidates = _select_fallback_candidates(
            retrieved_chunks=retrieved_chunks,
            max_sources=max_sources,
            snippet_chars=snippet_chars,
        )
    if not candidates:
        return _build_unsupported_answer(
            question=question,
            message="The retrieved document context was not sufficient to support an answer.",
        )

    answer_text = " ".join(candidate.snippet for candidate in candidates)
    sources = [_candidate_to_source(candidate) for candidate in candidates]

    return AnswerResponse(
        question=question,
        answer=answer_text,
        answer_supported=True,
        message="Answer generated strictly from retrieved document context.",
        sources=sources,
    )


def _generate_openai_grounded_answer(
    *,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    max_sources: int,
    snippet_chars: int,
    openai_api_key: str,
    openai_model: str,
) -> AnswerResponse:
    if not openai_model:
        raise DocumentProcessingError(
            "OpenAI answer generation is not configured. Set OPENAI_ANSWER_MODEL or use ANSWER_PROVIDER=local.",
            status_code=503,
        )

    client = create_openai_client(api_key=openai_api_key)
    prompt_messages = build_grounded_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    try:
        response = client.chat.completions.create(
            model=openai_model,
            messages=prompt_messages,
            temperature=0,
        )
        answer_text = (response.choices[0].message.content or "").strip()
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError(
            "OpenAI answer generation failed. Check your API key, billing, and model configuration.",
            status_code=502,
        ) from exc

    if not answer_text or answer_text == INSUFFICIENT_CONTEXT_ANSWER:
        return _build_unsupported_answer(
            question=question,
            message="The retrieved document context was not sufficient to support an answer.",
        )

    sources = build_source_snippets(
        retrieved_chunks=retrieved_chunks,
        max_sources=max_sources,
        snippet_chars=snippet_chars,
    )
    return AnswerResponse(
        question=question,
        answer=answer_text,
        answer_supported=True,
        message="Answer generated strictly from retrieved document context.",
        sources=sources,
    )


def build_source_snippets(
    *,
    retrieved_chunks: list[RetrievedChunk],
    max_sources: int,
    snippet_chars: int,
) -> list[SourceSnippet]:
    sources: list[SourceSnippet] = []
    seen_chunk_ids: set[str] = set()

    for chunk in retrieved_chunks:
        if len(sources) >= max_sources:
            break
        if chunk.chunk_id in seen_chunk_ids:
            continue
        sources.append(
            SourceSnippet(
                filename=chunk.filename,
                chunk_id=chunk.chunk_id,
                source_type=chunk.source_type,
                page_number=chunk.page_number,
                rank=chunk.rank,
                score=chunk.score,
                snippet=_trim_snippet(chunk.chunk_text, snippet_chars),
            )
        )
        seen_chunk_ids.add(chunk.chunk_id)

    return sources


def _select_sentence_candidates(
    *,
    question_terms: set[str],
    retrieved_chunks: list[RetrievedChunk],
    max_sources: int,
    snippet_chars: int,
) -> list[_SentenceCandidate]:
    scored_candidates: list[tuple[float, _SentenceCandidate]] = []

    for chunk in retrieved_chunks:
        for sentence in _split_sentences(chunk.chunk_text):
            sentence_terms = _extract_keywords(sentence)
            overlap = len(question_terms.intersection(sentence_terms))
            if question_terms and overlap == 0:
                continue
            if len(sentence.strip()) < 12:
                continue

            candidate = _SentenceCandidate(
                rank=chunk.rank,
                score=chunk.score,
                filename=chunk.filename,
                chunk_id=chunk.chunk_id,
                source_type=chunk.source_type,
                page_number=chunk.page_number,
                snippet=_trim_snippet(sentence, snippet_chars),
            )
            weighted_score = (overlap * 10.0) + chunk.score - (chunk.rank * 0.01)
            scored_candidates.append((weighted_score, candidate))

    if not scored_candidates:
        return []

    scored_candidates.sort(
        key=lambda item: (-item[0], item[1].rank, -item[1].score, item[1].chunk_id),
    )

    selected_candidates: list[_SentenceCandidate] = []
    seen_chunk_ids: set[str] = set()
    seen_snippets: set[str] = set()

    for _, candidate in scored_candidates:
        if len(selected_candidates) >= max_sources:
            break
        if candidate.chunk_id in seen_chunk_ids:
            continue
        if candidate.snippet in seen_snippets:
            continue
        selected_candidates.append(candidate)
        seen_chunk_ids.add(candidate.chunk_id)
        seen_snippets.add(candidate.snippet)

    return selected_candidates


def _select_fallback_candidates(
    *,
    retrieved_chunks: list[RetrievedChunk],
    max_sources: int,
    snippet_chars: int,
) -> list[_SentenceCandidate]:
    fallback_candidates: list[_SentenceCandidate] = []
    seen_chunk_ids: set[str] = set()

    for chunk in retrieved_chunks:
        if len(fallback_candidates) >= max_sources:
            break
        if chunk.chunk_id in seen_chunk_ids:
            continue

        sentence = _select_best_fallback_sentence(chunk.chunk_text, snippet_chars)
        if not sentence:
            continue

        fallback_candidates.append(
            _SentenceCandidate(
                rank=chunk.rank,
                score=chunk.score,
                filename=chunk.filename,
                chunk_id=chunk.chunk_id,
                source_type=chunk.source_type,
                page_number=chunk.page_number,
                snippet=sentence,
            )
        )
        seen_chunk_ids.add(chunk.chunk_id)

    return fallback_candidates


def _select_best_fallback_sentence(text: str, snippet_chars: int) -> str:
    sentences = _split_sentences(text)
    scored_sentences = []

    for sentence in sentences:
        stripped_sentence = sentence.strip()
        if len(stripped_sentence) < 12:
            continue
        score = _sentence_information_score(stripped_sentence)
        scored_sentences.append((score, stripped_sentence))

    if not scored_sentences:
        return ""

    scored_sentences.sort(key=lambda item: (-item[0], -len(item[1])))
    return _trim_snippet(scored_sentences[0][1], snippet_chars)


def _sentence_information_score(sentence: str) -> float:
    terms = _extract_keywords(sentence)
    uppercase_bonus = len(re.findall(r"\b[A-Z]{2,}\b", sentence))
    colon_bonus = 1 if ":" in sentence else 0
    return float(len(terms) + uppercase_bonus + colon_bonus)


def _candidate_to_source(candidate: _SentenceCandidate) -> SourceSnippet:
    return SourceSnippet(
        filename=candidate.filename,
        chunk_id=candidate.chunk_id,
        source_type=candidate.source_type,
        page_number=candidate.page_number,
        rank=candidate.rank,
        score=candidate.score,
        snippet=candidate.snippet,
    )


def _build_unsupported_answer(*, question: str, message: str) -> AnswerResponse:
    return AnswerResponse(
        question=question,
        answer=INSUFFICIENT_CONTEXT_ANSWER,
        answer_supported=False,
        message=message,
        sources=[],
    )


def _split_sentences(text: str) -> list[str]:
    normalized_text = text.replace("\n", " ")
    raw_sentences = re.split(r"(?<=[.!?])\s+", normalized_text)
    sentences = [sentence.strip() for sentence in raw_sentences if sentence.strip()]
    if sentences:
        return sentences
    stripped_text = normalized_text.strip()
    return [stripped_text] if stripped_text else []


def _trim_snippet(text: str, max_chars: int) -> str:
    stripped_text = " ".join(text.split())
    if len(stripped_text) <= max_chars:
        return stripped_text
    return stripped_text[: max_chars - 3].rstrip() + "..."


def _extract_keywords(text: str) -> set[str]:
    return {
        _normalize_term(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def _normalize_term(token: str) -> str:
    if token in TERM_NORMALIZATIONS:
        return TERM_NORMALIZATIONS[token]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token
