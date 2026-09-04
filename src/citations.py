"""
Citation and attribution utilities for the RAG pipeline.

Responsibilities:
1. Build stable citation markers from retrieved chunks.
2. Map citations to real source metadata.
3. Verify citation claims against original retrieved text.
4. Detect fabricated/invalid citation markers.
5. Provide safe fallback when supporting evidence is insufficient.
"""

from __future__ import annotations

import re
from typing import Any


# -------------------------------------------------------------------
# Citation map
# -------------------------------------------------------------------

def build_citation_map(chunks: list[dict]) -> dict[str, dict[str, Any]]:
    """
    Create stable citation markers such as [1], [2], [3].

    Each citation points to one actual retrieved chunk.
    """

    citation_map: dict[str, dict[str, Any]] = {}

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {}) or {}

        citation_key = f"[{index}]"

        citation_map[citation_key] = {
            "source": metadata.get(
                "source",
                metadata.get("source_path", "unknown")
            ),
            "source_path": metadata.get("source_path"),
            "chunk_id": metadata.get(
                "chunk_id",
                chunk.get("id")
            ),
            "chunk_index": metadata.get("chunk_index"),
            "section": metadata.get("section"),
            "page": metadata.get("page"),
            "text": chunk.get("text", ""),
        }

    return citation_map


# -------------------------------------------------------------------
# Prompt
# -------------------------------------------------------------------

def build_cited_prompt(
    question: str,
    chunks: list[dict],
) -> str:
    """
    Build a prompt that forces the model to cite only
    the supplied retrieved chunks.
    """

    citation_map = build_citation_map(chunks)

    context_parts = []

    for citation, data in citation_map.items():
        context_parts.append(
            f"""
SOURCE {citation}
Source: {data["source"]}
Chunk ID: {data["chunk_id"]}
Chunk Index: {data["chunk_index"]}
Section: {data["section"]}
Page: {data["page"]}

Content:
{data["text"]}
""".strip()
        )

    context = "\n\n".join(context_parts)

    return f"""
You are a grounded knowledge assistant.

Answer the question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Cite factual claims using citation markers such as [1], [2], or [3].
3. Every citation must correspond to a source provided below.
4. Do not create or invent citation numbers.
5. If the context does not contain enough information to answer the
   question, say:

"I don't have enough information in the provided context to answer
this question."

6. Do not guess.
7. Do not cite a source unless the source actually supports the claim.
8. Keep the answer concise and factual.

Provided Context:
-----------------
{context}
-----------------

Question:
{question}

Return only the answer with citations.
""".strip()


# -------------------------------------------------------------------
# Extract citations
# -------------------------------------------------------------------

def extract_citations(answer: str) -> list[str]:
    """
    Extract citation markers from generated answer.

    Example:
        "Stop the machine [1] and inspect it [2]."

    returns:
        ["[1]", "[2]"]
    """

    if not answer:
        return []

    matches = re.findall(r"\[\d+\]", answer)

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(matches))


# -------------------------------------------------------------------
# Validate citations
# -------------------------------------------------------------------

def validate_citations(
    answer: str,
    citation_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Verify that every citation appearing in the answer
    corresponds to a real retrieved chunk.
    """

    cited_markers = extract_citations(answer)

    invalid = [
        marker
        for marker in cited_markers
        if marker not in citation_map
    ]

    valid = [
        marker
        for marker in cited_markers
        if marker in citation_map
    ]

    return {
        "valid": len(invalid) == 0,
        "cited_markers": cited_markers,
        "valid_citations": valid,
        "invalid_citations": invalid,
    }


# -------------------------------------------------------------------
# Citation lookup
# -------------------------------------------------------------------

def get_citation(
    citation: str,
    citation_map: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Return metadata and original text for a citation.
    """

    return citation_map.get(citation)


# -------------------------------------------------------------------
# Verify a citation against source text
# -------------------------------------------------------------------

def verify_citation(
    citation: str,
    claim: str,
    citation_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Verify that a citation exists and that the source text contains
    meaningful words from the claim.

    This is intentionally a lightweight verification mechanism.
    It does not claim semantic entailment.
    """

    source = citation_map.get(citation)

    if source is None:
        return {
            "verified": False,
            "citation": citation,
            "reason": "Citation does not exist in retrieved context.",
        }

    source_text = source.get("text", "")

    if not source_text.strip():
        return {
            "verified": False,
            "citation": citation,
            "reason": "Source chunk contains no text.",
        }

    # Normalize text.
    claim_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            claim.lower(),
        )
    )

    source_words = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            source_text.lower(),
        )
    )

    overlap = claim_words.intersection(source_words)

    # Require some meaningful lexical overlap.
    verified = len(overlap) >= 2

    return {
        "verified": verified,
        "citation": citation,
        "source": source.get("source"),
        "chunk_id": source.get("chunk_id"),
        "chunk_index": source.get("chunk_index"),
        "section": source.get("section"),
        "page": source.get("page"),
        "original_text": source_text,
        "matching_terms": sorted(overlap),
        "reason": (
            "Claim has supporting terms in the original retrieved chunk."
            if verified
            else "Insufficient textual overlap with the original chunk."
        ),
    }


# -------------------------------------------------------------------
# Verify all citations
# -------------------------------------------------------------------

def verify_all_citations(
    answer: str,
    citation_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Verify all citations appearing in an answer.

    Each citation is checked against its corresponding
    original retrieved chunk.
    """

    markers = extract_citations(answer)

    results = {}

    for marker in markers:
        citation_data = citation_map.get(marker)

        if citation_data is None:
            results[marker] = {
                "verified": False,
                "reason": "Citation does not exist.",
            }
            continue

        # Use the complete answer as the claim context.
        results[marker] = verify_citation(
            marker,
            answer,
            citation_map,
        )

    all_verified = all(
        result.get("verified", False)
        for result in results.values()
    ) if results else True

    return {
        "all_verified": all_verified,
        "results": results,
    }


# -------------------------------------------------------------------
# Safe fallback
# -------------------------------------------------------------------

FALLBACK_MESSAGE = (
    "I don't have enough information in the provided context "
    "to answer this question."
)


def missing_context_response() -> dict[str, Any]:
    """
    Standard response when retrieved context is insufficient.
    """

    return {
        "answer": FALLBACK_MESSAGE,
        "citations": {},
        "sources": [],
        "grounded": False,
        "citation_validation": {
            "valid": True,
            "cited_markers": [],
            "valid_citations": [],
            "invalid_citations": [],
        },
    }


# -------------------------------------------------------------------
# Citation formatting
# -------------------------------------------------------------------

def format_citation_sources(
    citation_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert citation map into a clean source list for display.
    """

    sources = []

    for citation, data in citation_map.items():
        sources.append(
            {
                "citation": citation,
                "source": data.get("source"),
                "chunk_id": data.get("chunk_id"),
                "chunk_index": data.get("chunk_index"),
                "section": data.get("section"),
                "page": data.get("page"),
            }
        )

    return sources


# -------------------------------------------------------------------
# Module test
# -------------------------------------------------------------------

if __name__ == "__main__":

    sample_chunks = [
        {
            "id": "vibration_manual.txt:0",
            "text": (
                "If abnormal vibration is detected, "
                "stop the machine and begin the approved "
                "inspection procedure."
            ),
            "metadata": {
                "source": "vibration_manual.txt",
                "chunk_index": 0,
                "section": "Document body",
            },
        },
        {
            "id": "vibration_procedure.txt:0",
            "text": (
                "Do not restart the equipment until the "
                "inspection has been completed and the "
                "machine is considered safe."
            ),
            "metadata": {
                "source": "vibration_procedure.txt",
                "chunk_index": 0,
                "section": "Document body",
            },
        },
    ]

    print("Testing citation module...")
    print("-" * 50)

    citation_map = build_citation_map(sample_chunks)

    print("Citation map:")
    for key, value in citation_map.items():
        print(f"{key} -> {value['source']}")

    answer = (
        "Stop the machine and begin the approved "
        "inspection procedure [1]. Do not restart it "
        "until it is considered safe [2]."
    )

    validation = validate_citations(
        answer,
        citation_map,
    )

    print("\nCitation validation:")
    print(validation)

    verification = verify_all_citations(
        answer,
        citation_map,
    )

    print("\nCitation verification:")
    print(verification)

    print("\n✓ Citation module test complete!")