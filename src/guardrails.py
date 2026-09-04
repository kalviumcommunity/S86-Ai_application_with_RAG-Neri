"""
Hallucination Guardrails & Refusal Handling

Checks retrieval quality before allowing the LLM to generate an answer.
"""

from typing import Any


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Minimum similarity score required for a chunk to be considered
# sufficiently relevant.
MIN_TOP_SCORE = 0.50

# At least this many relevant chunks must exist.
MIN_SUPPORTING_CHUNKS = 1


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_chunk_score(chunk: dict[str, Any]) -> float:
    """
    Extract the similarity/relevance score from a retrieved chunk.

    Supports both:
        chunk["similarity"]
    and
        chunk["score"]

    Returns 0.0 if no valid score exists.
    """

    similarity = chunk.get("similarity")

    if similarity is not None:
        try:
            return float(similarity)
        except (TypeError, ValueError):
            pass

    score = chunk.get("score")

    if score is not None:
        try:
            return float(score)
        except (TypeError, ValueError):
            pass

    return 0.0


def retrieval_is_strong(
    chunks: list[dict[str, Any]],
    min_score: float = MIN_TOP_SCORE,
    min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
) -> bool:
    """
    Determine whether retrieved context is strong enough
    to safely generate an answer.

    Conditions:
    1. Retrieval must return at least one chunk.
    2. At least `min_supporting_chunks` chunks must meet
       the relevance threshold.
    """

    if not chunks:
        return False

    strong_chunks = [
        chunk
        for chunk in chunks
        if get_chunk_score(chunk) >= min_score
    ]

    return len(strong_chunks) >= min_supporting_chunks


def get_strong_chunks(
    chunks: list[dict[str, Any]],
    min_score: float = MIN_TOP_SCORE,
) -> list[dict[str, Any]]:
    """
    Return only chunks that meet the relevance threshold.
    """

    return [
        chunk
        for chunk in chunks
        if get_chunk_score(chunk) >= min_score
    ]


def retrieval_diagnostics(
    chunks: list[dict[str, Any]],
    min_score: float = MIN_TOP_SCORE,
) -> dict[str, Any]:
    """
    Return useful information for debugging and evaluation.
    """

    scores = [get_chunk_score(chunk) for chunk in chunks]

    strong_chunks = get_strong_chunks(
        chunks,
        min_score=min_score,
    )

    return {
        "retrieved_count": len(chunks),
        "scores": scores,
        "top_score": max(scores) if scores else 0.0,
        "strong_chunk_count": len(strong_chunks),
        "threshold": min_score,
        "passed": len(strong_chunks) >= MIN_SUPPORTING_CHUNKS,
    }


# ---------------------------------------------------------
# Safe refusal
# ---------------------------------------------------------

def refusal_response(
    reason: str = "weak_context",
) -> dict[str, Any]:
    """
    Return a safe response when retrieval is insufficient.
    """

    if reason == "no_context":
        message = (
            "I don't have enough information in the provided "
            "context to answer that reliably."
        )

    else:
        message = (
            "I don't have enough reliable context to answer "
            "that question."
        )

    return {
        "answer": message,
        "sources": [],
        "status": "refused",
        "reason": reason,
    }


# ---------------------------------------------------------
# Guardrail
# ---------------------------------------------------------

def check_retrieval(
    chunks: list[dict[str, Any]],
    min_score: float = MIN_TOP_SCORE,
    min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
) -> dict[str, Any]:
    """
    Perform the complete retrieval-quality check.
    """

    if not chunks:
        return {
            "allowed": False,
            "reason": "no_context",
            "message": (
                "No supporting context was retrieved."
            ),
            "diagnostics": retrieval_diagnostics(
                chunks,
                min_score,
            ),
        }

    strong_chunks = get_strong_chunks(
        chunks,
        min_score=min_score,
    )

    if len(strong_chunks) < min_supporting_chunks:
        return {
            "allowed": False,
            "reason": "weak_context",
            "message": (
                "Retrieved context did not meet the "
                "minimum relevance threshold."
            ),
            "diagnostics": retrieval_diagnostics(
                chunks,
                min_score,
            ),
        }

    return {
        "allowed": True,
        "reason": "strong_context",
        "message": "Retrieved context is strong enough.",
        "diagnostics": retrieval_diagnostics(
            chunks,
            min_score,
        ),
    }


# ---------------------------------------------------------
# Guarded answer
# ---------------------------------------------------------

def guarded_answer(
    question: str,
    retrieve_function,
    generate_function,
    candidate_k: int = 10,
    final_k: int = 3,
    min_score: float = MIN_TOP_SCORE,
) -> dict[str, Any]:
    """
    Complete guarded RAG flow.

    1. Retrieve chunks.
    2. Check retrieval quality.
    3. Refuse if context is weak.
    4. Generate only when context is strong.
    """

    chunks = retrieve_function(
        question,
        candidate_k,
    )

    check = check_retrieval(
        chunks,
        min_score=min_score,
    )

    if not check["allowed"]:
        refusal = refusal_response(
            reason=check["reason"],
        )

        return {
            "question": question,
            **refusal,
            "retrieved_chunks": chunks,
            "diagnostics": check["diagnostics"],
        }

    strong_chunks = get_strong_chunks(
        chunks,
        min_score=min_score,
    )

    strong_chunks = strong_chunks[:final_k]

    result = generate_function(
        question,
        strong_chunks,
    )

    return {
        **result,
        "status": "answered",
        "reason": "strong_context",
        "retrieved_chunks": strong_chunks,
        "diagnostics": check["diagnostics"],
    }


# ---------------------------------------------------------
# Testing
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("HALLUCINATION GUARDRAIL TEST")
    print("=" * 60)

    # Fake chunks for unit testing
    good_chunks = [
        {
            "id": "vibration_manual.txt:0",
            "similarity": 0.64,
            "text": (
                "If abnormal vibration is detected, "
                "stop the machine and begin the "
                "approved inspection procedure."
            ),
            "metadata": {
                "source": "vibration_manual.txt",
                "chunk_index": 0,
                "section": "Document body",
            },
        }
    ]

    weak_chunks = [
        {
            "id": "unrelated.txt:0",
            "similarity": 0.20,
            "text": "This document contains unrelated information.",
            "metadata": {
                "source": "unrelated.txt",
                "chunk_index": 0,
                "section": "Document body",
            },
        }
    ]

    empty_chunks = []

    print("\n[1] Strong retrieval")
    print("-" * 40)
    print(check_retrieval(good_chunks))

    print("\n[2] Weak retrieval")
    print("-" * 40)
    print(check_retrieval(weak_chunks))

    print("\n[3] Empty retrieval")
    print("-" * 40)
    print(check_retrieval(empty_chunks))

    print("\n[4] Refusal response")
    print("-" * 40)
    print(refusal_response("weak_context"))

    print("\n" + "=" * 60)
    print("Guardrail test complete!")
    print("=" * 60)