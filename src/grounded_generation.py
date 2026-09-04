"""
Grounded Generation + Hallucination Guardrails + Citations

Pipeline:

Question
   ↓
Embedding
   ↓
Vector Retrieval
   ↓
Retrieval Quality Check
   ↓
 ┌───────────────┐
 │               │
Weak           Strong
 │               │
Refuse        Generate
                 ↓
             Citations
                 ↓
          Source Verification

Uses Gemini through Google's OpenAI-compatible API.
Environment variables:

OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_API_KEY=your_key
CHAT_MODEL=gemini-3.1-flash-lite
EMBED_MODEL=gemini-embedding-001
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

try:
    from .embeddings import embed
    from .vector_store import VectorStore
    from .citations import (
        FALLBACK_MESSAGE,
        build_citation_map,
        build_cited_prompt,
        extract_citations,
        format_citation_sources,
        missing_context_response,
        validate_citations,
        verify_all_citations,
    )
except ImportError:
    from embeddings import embed
    from vector_store import VectorStore
    from citations import (
        FALLBACK_MESSAGE,
        build_citation_map,
        build_cited_prompt,
        extract_citations,
        format_citation_sources,
        missing_context_response,
        validate_citations,
        verify_all_citations,
    )


# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

load_dotenv()

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DEFAULT_VECTOR_DIR = "outputs/chroma_db"
DEFAULT_VECTOR_DIMENSION = 3072
DEFAULT_COLLECTION = "rag_chunks"

DEFAULT_GENERATION_MODEL = os.getenv(
    "CHAT_MODEL",
    "gemini-3.1-flash-lite",
)

CANDIDATE_K = 10
FINAL_K = 3

# Your current corpus produces scores around 0.60+
# for strongly related questions.
MIN_TOP_SCORE = 0.50

MIN_SUPPORTING_CHUNKS = 1


# ==================================================================
# Vector Store
# ==================================================================

_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """
    Return the shared vector store instance.
    """

    global _vector_store

    if _vector_store is None:
        _vector_store = VectorStore(
            persist_dir=DEFAULT_VECTOR_DIR,
            collection_name=DEFAULT_COLLECTION,
            vector_dimension=DEFAULT_VECTOR_DIMENSION,
        )

    return _vector_store


# ==================================================================
# Retrieval
# ==================================================================

def retrieve_context(
    query_vector: list[float],
    vector_store: VectorStore,
    k: int = 4,
) -> list[dict]:
    """
    Retrieve chunks from the existing VectorStore.

    Handles VectorStore implementations where the search
    method expects a different parameter name.
    """

    # First try the common positional form.
    try:
        return vector_store.search(
            query_vector,
            k,
        )
    except TypeError:
        pass

    # Try n_results if supported.
    try:
        return vector_store.search(
            query_vector,
            n_results=k,
        )
    except TypeError:
        pass

    # Try top_k if supported.
    try:
        return vector_store.search(
            query_vector,
            top_k=k,
        )
    except TypeError:
        pass

    raise TypeError(
        "VectorStore.search() does not accept "
        "k, n_results, or top_k."
    )


def retrieve_chunks(
    question: str,
    k: int = 3,
) -> list[dict]:
    """
    Convert the question to an embedding and retrieve
    the most relevant chunks.
    """

    vector_store = get_vector_store()

    query_vector = embed([question])[0]

    chunks = retrieve_context(
        query_vector,
        vector_store,
        k=k,
    )

    return chunks


# ==================================================================
# Context Formatting
# ==================================================================

def format_context(
    chunks: list[dict],
) -> str:
    """
    Format retrieved chunks with stable citation markers.

    Example:

    [1] Source: vibration_manual.txt
    Chunk: vibration_manual.txt:0
    Section: Document body

    Content:
    ...
    """

    if not chunks:
        return ""

    parts = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        metadata = chunk.get(
            "metadata",
            {},
        ) or {}

        source = metadata.get(
            "source",
            metadata.get(
                "source_path",
                "unknown",
            ),
        )

        chunk_id = metadata.get(
            "chunk_id",
            chunk.get(
                "id",
                f"{source}:{metadata.get('chunk_index', index - 1)}",
            ),
        )

        chunk_index = metadata.get(
            "chunk_index",
            index - 1,
        )

        section = metadata.get(
            "section",
            "unknown",
        )

        text = chunk.get(
            "text",
            "",
        )

        parts.append(
            f"[{index}] Source: {source}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Chunk Index: {chunk_index}\n"
            f"Section: {section}\n\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(parts)


# ==================================================================
# Prompt Builder
# ==================================================================

def build_prompt(
    question: str,
    chunks: list[dict],
) -> dict[str, Any]:
    """
    Build a citation-aware grounded prompt.
    """

    citation_map = build_citation_map(
        chunks
    )

    # Use the citation module's prompt builder.
    prompt = build_cited_prompt(
        question,
        chunks,
    )

    context = format_context(
        chunks
    )

    return {
        "prompt": prompt,
        "citation_map": citation_map,
        "context": context,
    }


# ==================================================================
# LLM Client
# ==================================================================

def get_llm_client():
    """
    Create the Gemini client using Google's
    OpenAI-compatible API.

    IMPORTANT:
    Uses OPENAI_API_KEY, NOT GEMINI_API_KEY.
    """

    load_dotenv()

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in the environment."
        )

    try:
        from openai import OpenAI

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required. "
            "Install it with: pip install openai"
        ) from exc


# ==================================================================
# Grounded Generation
# ==================================================================

def generate_answer(
    query: str,
    context: str,
    client=None,
    model: str | None = None,
) -> str:
    """
    Generate an answer using ONLY the supplied context.

    The model is explicitly instructed not to use
    outside knowledge.
    """

    if client is None:
        client = get_llm_client()

    if model is None:
        model = os.getenv(
            "CHAT_MODEL",
            DEFAULT_GENERATION_MODEL,
        )

    if not context.strip():
        return FALLBACK_MESSAGE

    prompt = f"""
You are a grounded RAG assistant.

Answer the question using ONLY the retrieved context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Do not make assumptions.
4. Every factual claim must be supported by the context.
5. Cite factual claims using [1], [2], [3], etc.
6. Use ONLY citation numbers that exist in the provided context.
7. Do not fabricate citations.
8. If the context does not contain enough information, say:
   "I don't have enough information in the provided context to answer that reliably."

Retrieved Context:
{context}

Question:
{query}

Return a concise answer with citations.
"""

    logger.info(
        "LLM REQUEST"
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a grounded RAG assistant. "
                    "You must answer only from supplied context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = (
        response
        .choices[0]
        .message
        .content
        or ""
    )

    logger.info(
        "LLM RESPONSE: %s",
        answer,
    )

    return answer.strip()


# ==================================================================
# Citation Verification
# ==================================================================

def verify_answer_against_sources(
    answer: str,
    chunks: list[dict],
) -> dict[str, Any]:
    """
    Verify every citation in the generated answer
    against the original retrieved chunks.
    """

    citation_map = build_citation_map(
        chunks
    )

    validation = validate_citations(
        answer,
        citation_map,
    )

    verification = verify_all_citations(
        answer,
        citation_map,
    )

    return {
        "citation_validation": validation,
        "citation_verification": verification,
        "citations": citation_map,
        "verified": (
            validation.get(
                "valid",
                False,
            )
            and verification.get(
                "all_verified",
                False,
            )
        ),
    }


# ==================================================================
# Grounded Answer
# ==================================================================

def generate_grounded_answer(
    question: str,
    retrieved_chunks: list[dict],
) -> dict:
    """
    Generate an answer from injected retrieved chunks.

    This function does NOT perform retrieval.

    This directly satisfies the assignment requirement:

    "Generate an answer using only the injected retrieved context."
    """

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": FALLBACK_MESSAGE,
            "context": "",
            "sources": [],
            "grounded": False,
            "status": "refused",
            "reason": "no_context",
            "citations": {},
        }

    prompt_data = build_prompt(
        question,
        retrieved_chunks,
    )

    answer = generate_answer(
        query=question,
        context=prompt_data["context"],
    )

    citation_result = verify_answer_against_sources(
        answer,
        retrieved_chunks,
    )

    sources = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        metadata = chunk.get(
            "metadata",
            {},
        ) or {}

        source = metadata.get(
            "source",
            metadata.get(
                "source_path",
                "unknown",
            ),
        )

        chunk_id = metadata.get(
            "chunk_id",
            chunk.get(
                "id",
                f"{source}:{metadata.get('chunk_index', index - 1)}",
            ),
        )

        chunk_index = metadata.get(
            "chunk_index",
            index - 1,
        )

        section = metadata.get(
            "section",
            "unknown",
        )

        sources.append(
            {
                "citation": f"[{index}]",
                "source": source,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "section": section,
            }
        )

    return {
        "question": question,
        "answer": answer,
        "context": prompt_data["context"],
        "sources": sources,
        "grounded": True,
        "status": "answered",
        "reason": "retrieved_context",
        "citations": citation_result["citations"],
        "citation_validation": citation_result[
            "citation_validation"
        ],
        "citation_verification": citation_result[
            "citation_verification"
        ],
        "verified": citation_result[
            "verified"
        ],
        "retrieved_chunks": retrieved_chunks,
    }


# ==================================================================
# Retrieval Guardrails
# ==================================================================

def check_retrieval(
    chunks: list[dict],
    min_score: float = MIN_TOP_SCORE,
    min_supporting_chunks: int = MIN_SUPPORTING_CHUNKS,
) -> dict:
    """
    Check whether retrieval is strong enough to answer.

    Conditions:

    - At least one chunk must exist.
    - At least one chunk must have score >= threshold.
    """

    if not chunks:
        return {
            "allowed": False,
            "reason": "no_context",
            "message": (
                "No supporting context was retrieved."
            ),
            "diagnostics": {
                "retrieved_count": 0,
                "scores": [],
                "top_score": 0.0,
                "strong_chunk_count": 0,
                "threshold": min_score,
                "passed": False,
            },
        }

    scores = []

    for chunk in chunks:
        score = chunk.get(
            "similarity",
            chunk.get(
                "score",
                0.0,
            ),
        )

        try:
            score = float(score)
        except (
            TypeError,
            ValueError,
        ):
            score = 0.0

        scores.append(score)

    top_score = max(
        scores
    ) if scores else 0.0

    strong_chunk_count = sum(
        1
        for score in scores
        if score >= min_score
    )

    passed = (
        strong_chunk_count
        >= min_supporting_chunks
    )

    return {
        "allowed": passed,
        "reason": (
            "strong_context"
            if passed
            else "weak_context"
        ),
        "message": (
            "Retrieved context is strong enough."
            if passed
            else (
                "Retrieved context did not meet "
                "the minimum relevance threshold."
            )
        ),
        "diagnostics": {
            "retrieved_count": len(chunks),
            "scores": scores,
            "top_score": top_score,
            "strong_chunk_count": strong_chunk_count,
            "threshold": min_score,
            "passed": passed,
        },
    }


def get_strong_chunks(
    chunks: list[dict],
    min_score: float = MIN_TOP_SCORE,
) -> list[dict]:
    """
    Return only chunks that pass the relevance threshold.
    """

    strong = []

    for chunk in chunks:
        score = chunk.get(
            "similarity",
            chunk.get(
                "score",
                0.0,
            ),
        )

        try:
            score = float(score)
        except (
            TypeError,
            ValueError,
        ):
            score = 0.0

        if score >= min_score:
            strong.append(chunk)

    return strong


def has_supporting_context(
    question: str,
    chunks: list[dict],
) -> bool:
    """
    Return True if retrieval contains strong context.
    """

    result = check_retrieval(
        chunks,
        min_score=MIN_TOP_SCORE,
    )

    return bool(
        result["allowed"]
    )


# ==================================================================
# Refusal
# ==================================================================

def refusal_response(
    reason: str = "weak_context",
) -> dict:
    """
    Return a safe refusal response.
    """

    if reason == "no_context":
        message = (
            "I don't have enough information in the "
            "provided context to answer that reliably."
        )

    else:
        message = (
            "I don't have enough reliable context "
            "to answer that question."
        )

    return {
        "answer": message,
        "sources": [],
        "status": "refused",
        "reason": reason,
    }


# ==================================================================
# Guarded Answer
# ==================================================================

def guarded_answer(
    question: str,
    candidate_k: int = CANDIDATE_K,
    final_k: int = FINAL_K,
    min_score: float = MIN_TOP_SCORE,
) -> dict:
    """
    Complete hallucination-safe RAG pipeline.

    Question
       ↓
    Retrieval
       ↓
    Relevance threshold
       ↓
    Refuse OR Generate
    """

    logger.info(
        "GUARDED QUERY: %s",
        question,
    )

    # --------------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------------

    chunks = retrieve_chunks(
        question,
        k=candidate_k,
    )

    logger.info(
        "Retrieved %d chunks",
        len(chunks),
    )

    # --------------------------------------------------------------
    # Guardrail
    # --------------------------------------------------------------

    check = check_retrieval(
        chunks,
        min_score=min_score,
    )

    logger.info(
        "Retrieval diagnostics: %s",
        check["diagnostics"],
    )

    # --------------------------------------------------------------
    # Refuse weak context
    # --------------------------------------------------------------

    if not check["allowed"]:
        refusal = refusal_response(
            check["reason"]
        )

        return {
            "question": question,
            "answer": refusal["answer"],
            "sources": [],
            "status": "refused",
            "reason": check["reason"],
            "grounded": False,
            "retrieved_chunks": chunks,
            "diagnostics": check[
                "diagnostics"
            ],
            "citations": {},
        }

    # --------------------------------------------------------------
    # Keep strong chunks
    # --------------------------------------------------------------

    strong_chunks = get_strong_chunks(
        chunks,
        min_score=min_score,
    )

    strong_chunks = strong_chunks[
        :final_k
    ]

    # --------------------------------------------------------------
    # Generate
    # --------------------------------------------------------------

    result = generate_grounded_answer(
        question,
        strong_chunks,
    )

    result["status"] = "answered"
    result["reason"] = "strong_context"
    result["diagnostics"] = check[
        "diagnostics"
    ]

    return result


# ==================================================================
# Answer Query
# ==================================================================

def answer_query(
    question: str,
    candidate_k: int = CANDIDATE_K,
    final_k: int = FINAL_K,
) -> dict:
    """
    Main backwards-compatible query function.

    Uses guarded retrieval + grounded generation.
    """

    return guarded_answer(
        question,
        candidate_k=candidate_k,
        final_k=final_k,
    )


# ==================================================================
# Ungrounded Answer
# ==================================================================

def generate_ungrounded_answer(
    question: str,
) -> str:
    """
    Generate an answer WITHOUT retrieval.

    This exists only for the assignment's
    with-vs-without retrieval comparison.

    It is intentionally NOT grounded.
    """

    client = get_llm_client()

    model = os.getenv(
        "CHAT_MODEL",
        DEFAULT_GENERATION_MODEL,
    )

    prompt = f"""
Answer this question using your general language-model knowledge.

Do not retrieve or use the project's documents.

Question:
{question}
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# ==================================================================
# Print Grounding Result
# ==================================================================

def print_grounding_check(
    result: dict,
) -> None:
    """
    Print a readable grounded-generation result.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GROUNDING + CITATION CHECK"
    )

    print(
        "=" * 70
    )

    print(
        "\nQuestion:"
    )

    print(
        result.get(
            "question",
            "",
        )
    )

    print(
        "\nStatus:"
    )

    print(
        result.get(
            "status",
            "",
        )
    )

    print(
        "\nGrounded:"
    )

    print(
        result.get(
            "grounded",
            False,
        )
    )

    print(
        "\nAnswer:"
    )

    print(
        result.get(
            "answer",
            "",
        )
    )

    print(
        "\nSources:"
    )

    for source in result.get(
        "sources",
        [],
    ):
        print(
            f"  {source['citation']} "
            f"{source['source']} "
            f"(chunk {source['chunk_index']})"
        )

    print(
        "\nCitation verification:"
    )

    print(
        result.get(
            "citation_verification",
            {},
        )
    )

    diagnostics = result.get(
        "diagnostics"
    )

    if diagnostics:

        print(
            "\nRetrieval diagnostics:"
        )

        print(
            f"  Retrieved: "
            f"{diagnostics.get('retrieved_count')}"
        )

        print(
            f"  Top score: "
            f"{diagnostics.get('top_score'):.4f}"
        )

        print(
            f"  Strong chunks: "
            f"{diagnostics.get('strong_chunk_count')}"
        )

        print(
            f"  Threshold: "
            f"{diagnostics.get('threshold')}"
        )

        print(
            f"  Passed: "
            f"{diagnostics.get('passed')}"
        )

    print(
        "\n"
        + "=" * 70
    )


# ==================================================================
# Assignment Demonstration
# ==================================================================

def run_assignment_demo() -> None:
    """
    Demonstrate the assignment requirements.

    1. Strong context → answer
    2. Weak/no context → refusal
    3. Citation verification
    4. With retrieval vs without retrieval
    """

    question = (
        "What should a technician do if abnormal "
        "vibration is detected?"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RAG GENERATION + CITATION + GUARDRAIL DEMO"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------------
    # Test 1: Guarded answer
    # --------------------------------------------------------------

    print(
        "\n[1] GUARDED ANSWER"
    )

    print(
        "-" * 70
    )

    try:

        result = guarded_answer(
            question,
            candidate_k=10,
            final_k=3,
        )

        print_grounding_check(
            result
        )

    except Exception as exc:

        print(
            f"ERROR: {exc}"
        )

        print(
            "\nCheck your .env configuration:"
        )

        print(
            "OPENAI_API_KEY=your_key"
        )

        print(
            "OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        print(
            "CHAT_MODEL=gemini-3.1-flash-lite"
        )

        print(
            "EMBED_MODEL=gemini-embedding-001"
        )

        return

    # --------------------------------------------------------------
    # Test 2: Missing context
    # --------------------------------------------------------------

    print(
        "\n[2] MISSING CONTEXT FALLBACK"
    )

    print(
        "-" * 70
    )

    fallback = generate_grounded_answer(
        "What is the refund policy for a product?",
        [],
    )

    print(
        fallback
    )

    # --------------------------------------------------------------
    # Test 3: Citation verification
    # --------------------------------------------------------------

    print(
        "\n[3] CITATION VERIFICATION"
    )

    print(
        "-" * 70
    )

    print(
        "Verified:",
        result.get(
            "verified",
            False,
        ),
    )

    print(
        "Citation validation:"
    )

    print(
        result.get(
            "citation_validation",
            {},
        )
    )

    print(
        "Citation verification:"
    )

    print(
        result.get(
            "citation_verification",
            {},
        )
    )

    # --------------------------------------------------------------
    # Test 4: Without retrieval
    # --------------------------------------------------------------

    print(
        "\n[4] WITHOUT RETRIEVAL"
    )

    print(
        "-" * 70
    )

    try:

        ungrounded = generate_ungrounded_answer(
            question
        )

        print(
            ungrounded
        )

    except Exception as exc:

        print(
            "Unable to generate ungrounded answer:"
        )

        print(
            exc
        )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DEMO COMPLETE"
    )

    print(
        "=" * 70
    )


# ==================================================================
# Main
# ==================================================================

def main() -> None:
    run_assignment_demo()


if __name__ == "__main__":
    main()