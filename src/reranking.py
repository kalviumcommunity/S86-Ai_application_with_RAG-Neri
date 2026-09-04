"""
Reranking utilities for improving retrieval precision.

Uses embedding similarity to rerank retrieved candidates.
"""

from typing import Optional

from .embeddings import (
    embed,
    rank_chunks,
)


# ============================================================
# RERANK CANDIDATES
# ============================================================

def rerank_candidates(
    query: str,
    candidates: list,
    top_k: int = 5,
    client=None,
    model: Optional[str] = None,
) -> list:
    """
    Rerank retrieved candidates using semantic similarity.

    Args:
        query:
            User query.

        candidates:
            Retrieved chunks.

        top_k:
            Number of results to return.

        client:
            Optional embedding client.

        model:
            Optional embedding model.

    Returns:
        Reranked candidate list.
    """

    if not candidates:
        return []

    if top_k <= 0:
        return []

    # --------------------------------------------------------
    # Generate query embedding
    # --------------------------------------------------------

    query_embedding = embed(
        [query],
        client=client,
        model=model,
    )[0]

    # --------------------------------------------------------
    # Some retrieval results already contain embeddings.
    # --------------------------------------------------------

    candidates_with_embeddings = []

    for candidate in candidates:

        if isinstance(candidate, dict):

            embedding = candidate.get(
                "embedding",
                [],
            )

        else:

            embedding = getattr(
                candidate,
                "embedding",
                [],
            )

        if embedding:

            candidates_with_embeddings.append(
                candidate
            )

    # --------------------------------------------------------
    # If candidates contain embeddings, rank directly.
    # --------------------------------------------------------

    if candidates_with_embeddings:

        return rank_chunks(
            query_embedding,
            candidates_with_embeddings,
            top_k=top_k,
        )

    # --------------------------------------------------------
    # If candidates do not contain embeddings, create them.
    # --------------------------------------------------------

    texts = []

    valid_candidates = []

    for candidate in candidates:

        if isinstance(candidate, dict):

            text = candidate.get(
                "text",
                "",
            )

        else:

            text = getattr(
                candidate,
                "text",
                "",
            )

        if text:

            texts.append(text)

            valid_candidates.append(
                candidate
            )

    if not valid_candidates:
        return []

    vectors = embed(
        texts,
        client=client,
        model=model,
    )

    # --------------------------------------------------------
    # Create temporary objects for ranking.
    # --------------------------------------------------------

    temporary = []

    for candidate, vector in zip(
        valid_candidates,
        vectors,
    ):

        if isinstance(candidate, dict):

            item = dict(candidate)

            item["embedding"] = vector

        else:

            # For object-based candidates, create a dictionary
            # containing the information needed for ranking.

            item = {
                "original": candidate,
                "text": getattr(
                    candidate,
                    "text",
                    "",
                ),
                "metadata": getattr(
                    candidate,
                    "metadata",
                    {},
                ),
                "embedding": vector,
            }

        temporary.append(item)

    ranked = rank_chunks(
        query_embedding,
        temporary,
        top_k=top_k,
    )

    # --------------------------------------------------------
    # Restore original objects where applicable.
    # --------------------------------------------------------

    final_results = []

    for item in ranked:

        if (
            isinstance(item, dict)
            and "original" in item
        ):

            final_results.append(
                item["original"]
            )

        else:

            final_results.append(item)

    return final_results


# ============================================================
# OPTIONAL SCORE FUNCTION
# ============================================================

def rerank_with_scores(
    query: str,
    candidates: list,
    top_k: int = 5,
    client=None,
    model: Optional[str] = None,
) -> list[dict]:
    """
    Rerank candidates and return explicit similarity scores.

    Returns dictionaries like:

        {
            "candidate": ...,
            "similarity": 0.82
        }
    """

    if not candidates:
        return []

    query_embedding = embed(
        [query],
        client=client,
        model=model,
    )[0]

    results = []

    for candidate in candidates:

        if isinstance(candidate, dict):

            embedding = candidate.get(
                "embedding",
                [],
            )

        else:

            embedding = getattr(
                candidate,
                "embedding",
                [],
            )

        if not embedding:
            continue

        from .embeddings import cosine_similarity

        similarity = cosine_similarity(
            query_embedding,
            embedding,
        )

        results.append(
            {
                "candidate": candidate,
                "similarity": similarity,
            }
        )

    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return results[:top_k]