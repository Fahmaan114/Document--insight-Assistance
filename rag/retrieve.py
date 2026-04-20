import faiss
import numpy as np
import re

from rag.embeddings import embed_texts, load_vector_store
from rag.ingest import DocumentProcessingError
from rag.schemas import RetrievalResponse, RetrievedChunk


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
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
    "accept": "accept",
    "accepted": "accept",
    "accepting": "accept",
    "accepts": "accept",
    "file": "file",
    "files": "file",
    "require": "require",
    "required": "require",
    "requires": "require",
    "requiring": "require",
    "type": "type",
    "types": "type",
}


def retrieve_relevant_chunks(
    *,
    question: str,
    vectorstore_dir,
    embedding_provider: str,
    embedding_model: str,
    batch_size: int,
    api_key: str,
    local_dummy_dimension: int,
    top_k: int,
    min_score: float = 0.0,
) -> RetrievalResponse:
    normalized_question = question.strip()
    if not normalized_question:
        raise DocumentProcessingError("A question is required for retrieval.", status_code=400)
    if top_k <= 0:
        raise DocumentProcessingError("top_k must be greater than 0.", status_code=400)

    index, metadata = load_vector_store(vectorstore_dir=vectorstore_dir)
    if index is None or metadata.vector_count == 0:
        return RetrievalResponse(
            question=normalized_question,
            top_k=top_k,
            no_results=True,
            message="No indexed document chunks are available yet.",
            results=[],
        )

    query_embedding = embed_texts(
        texts=[normalized_question],
        provider=embedding_provider,
        embedding_model=embedding_model,
        batch_size=batch_size,
        api_key=api_key,
        local_dummy_dimension=local_dummy_dimension,
    )

    if metadata.embedding_provider != query_embedding.provider:
        raise DocumentProcessingError(
            (
                "The current embedding provider does not match the persisted index. "
                "Use the same provider that was used during indexing or rebuild the vector store."
            ),
            status_code=409,
        )
    if metadata.embedding_model != query_embedding.model:
        raise DocumentProcessingError(
            (
                "The current embedding model does not match the persisted index. "
                "Use the same model that was used during indexing or rebuild the vector store."
            ),
            status_code=409,
        )
    if metadata.embedding_dimension != query_embedding.dimension:
        raise DocumentProcessingError(
            "The current embedding dimension does not match the persisted index.",
            status_code=409,
        )

    query_vector = np.asarray(query_embedding.vectors, dtype="float32")
    faiss.normalize_L2(query_vector)

    search_k = min(max(top_k * 3, top_k), metadata.vector_count)
    scores, vector_ids = index.search(query_vector, search_k)
    results = _build_ranked_results(
        question=normalized_question,
        embedding_provider=query_embedding.provider,
        scores=scores[0],
        vector_ids=vector_ids[0],
        metadata_entries=metadata.entries,
        min_score=min_score,
        top_k=top_k,
    )

    if not results:
        return RetrievalResponse(
            question=normalized_question,
            top_k=top_k,
            no_results=True,
            message="No useful context was found in the indexed documents for this question.",
            results=[],
        )

    return RetrievalResponse(
        question=normalized_question,
        top_k=top_k,
        no_results=False,
        message=f"Retrieved {len(results)} relevant chunk(s).",
        results=results,
    )


def _build_ranked_results(
    *,
    question: str,
    embedding_provider: str,
    scores: np.ndarray,
    vector_ids: np.ndarray,
    metadata_entries,
    min_score: float,
    top_k: int,
) -> list[RetrievedChunk]:
    ranked_results: list[RetrievedChunk] = []
    seen_chunk_ids: set[str] = set()
    current_rank = 1
    question_terms = _extract_keywords(question)

    for score, vector_id in zip(scores.tolist(), vector_ids.tolist()):
        if len(ranked_results) >= top_k:
            break
        if vector_id < 0:
            continue
        metadata_entry = metadata_entries[vector_id]
        if metadata_entry.chunk_id in seen_chunk_ids:
            continue
        if not _is_useful_match(
            question_terms=question_terms,
            chunk_text=metadata_entry.chunk_text,
            score=float(score),
            min_score=min_score,
            embedding_provider=embedding_provider,
        ):
            continue
        ranked_results.append(
            RetrievedChunk(
                rank=current_rank,
                score=float(score),
                **metadata_entry.model_dump(),
            )
        )
        seen_chunk_ids.add(metadata_entry.chunk_id)
        current_rank += 1

    return ranked_results


def _is_useful_match(
    *,
    question_terms: set[str],
    chunk_text: str,
    score: float,
    min_score: float,
    embedding_provider: str,
) -> bool:
    if not question_terms:
        return score >= min_score

    chunk_terms = _extract_keywords(chunk_text)
    has_keyword_overlap = bool(question_terms.intersection(chunk_terms))
    if has_keyword_overlap:
        overlap_threshold = _effective_score_threshold(
            embedding_provider=embedding_provider,
            min_score=min_score,
            has_keyword_overlap=True,
        )
        return score >= overlap_threshold

    # Allow strong semantic matches even without literal overlap, but keep
    # the local_dummy path conservative enough to reject unrelated queries.
    strong_semantic_score = max(
        _effective_score_threshold(
            embedding_provider=embedding_provider,
            min_score=min_score,
            has_keyword_overlap=False,
        ),
        0.55 if embedding_provider != "local_dummy" else 0.45,
    )
    return score >= strong_semantic_score


def _extract_keywords(text: str) -> set[str]:
    terms = {
        _normalize_term(token)
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }
    return terms


def _normalize_term(token: str) -> str:
    if token in TERM_NORMALIZATIONS:
        return TERM_NORMALIZATIONS[token]
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _effective_score_threshold(
    *,
    embedding_provider: str,
    min_score: float,
    has_keyword_overlap: bool,
) -> float:
    if embedding_provider == "local_dummy":
        if has_keyword_overlap:
            # local_dummy scores are weaker for obvious lexical matches, so we
            # relax the default development threshold a bit without ignoring
            # intentionally strict caller-provided thresholds.
            if min_score <= 0.3:
                return max(0.05, min_score * 0.4)
            return min_score
        return max(0.2, min_score)
    return min_score
