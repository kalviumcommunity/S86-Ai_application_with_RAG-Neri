"""
Grounded Answer Generation

This module:
1. Retrieves relevant chunks.
2. Builds a prompt using only retrieved context.
3. Generates grounded answers.
4. Handles missing-context fallback.
5. Compares answers with and without retrieval.
6. Provides source information for citation work.
"""

import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .embeddings import embed
from .llm_api import generate_answer
from .reranking import rerank_candidates
from .vector_store import VectorStore


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VECTOR_DIR = Path("outputs/chroma_db")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "grounded_answers.md"

CANDIDATE_K = 10
FINAL_K = 3

FALLBACK_MESSAGE = (
    "I don't have enough information in the provided context."
)

SUPPORTED_QUERY = (
    "What should a technician do if abnormal vibration is detected?"
)

UNSUPPORTED_QUERY = (
    "Who is the president of India?"
)


# ---------------------------------------------------------
# Vector Store
# ---------------------------------------------------------

def get_vector_store() -> VectorStore:
    """
    Return the project's existing vector store.
    """

    return VectorStore(
        persist_dir=VECTOR_DIR,
        collection_name="rag_chunks",
        vector_dimension=3072,
    )


# ---------------------------------------------------------
# Retrieval
# ---------------------------------------------------------

def retrieve_chunks(
    question: str,
    k: int = FINAL_K,
) -> list[dict]:
    """
    Retrieve relevant chunks from Chroma.

    Important:
    VectorStore.search() expects top_k, not k.
    """

    if not question or not question.strip():
        return []

    vector_store = get_vector_store()

    # Generate embedding for the query.
    query_vector = embed([question])[0]

    # IMPORTANT:
    # Your VectorStore.search() uses top_k.
    chunks = vector_store.search(
        query_vector,
        top_k=k,
    )

    return chunks


# ---------------------------------------------------------
# Context Formatting
# ---------------------------------------------------------

def format_context(
    chunks: list[dict],
) -> str:
    """
    Format retrieved chunks with source information.
    """

    if not chunks:
        return ""

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):

        metadata = chunk.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        chunk_id = metadata.get(
            "chunk_id",
            chunk.get("id", "unknown"),
        )

        chunk_index = metadata.get(
            "chunk_index",
            "unknown",
        )

        section = metadata.get(
            "section",
            "unknown",
        )

        page = metadata.get(
            "page",
            None,
        )

        text = chunk.get(
            "text",
            "",
        )

        location = (
            f"chunk_id={chunk_id}, "
            f"chunk_index={chunk_index}"
        )

        if page is not None:
            location += f", page={page}"

        context_parts.append(
            f"[{index}]\n"
            f"Source: {source}\n"
            f"Location: {location}\n"
            f"Section: {section}\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(context_parts)


# ---------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------

def build_prompt(
    question: str,
    chunks: list[dict],
) -> dict:
    """
    Build a grounded prompt.

    The model is explicitly instructed to:
    - use only supplied context
    - cite claims using [1], [2], etc.
    - never invent citations
    - fall back when context is insufficient
    """

    context = format_context(chunks)

    if not context:
        return {
            "prompt": "",
            "context": "",
            "sources_used": [],
        }

    sources = []

    for chunk in chunks:

        metadata = chunk.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        if source not in sources:
            sources.append(source)

    prompt = f"""
You are a technical support assistant.

GROUNDING RULES:

1. Answer ONLY using the supplied context.
2. Do NOT use outside knowledge.
3. Do NOT guess.
4. Do NOT invent facts.
5. Do NOT invent source names.
6. Every factual claim must be supported by the supplied context.
7. Cite factual claims using the source markers [1], [2], [3], etc.
8. Only use citation numbers that actually appear in the supplied context.
9. If the context does not contain enough information, respond exactly with:

"I don't have enough information in the provided context."

10. Never create a citation for information that is not present in the context.

CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

    return {
        "prompt": prompt,
        "context": context,
        "sources_used": sources,
    }


# ---------------------------------------------------------
# Grounded Answer Generation
# ---------------------------------------------------------

def generate_grounded_answer(
    question: str,
    retrieved_chunks: list[dict],
) -> dict:
    """
    Generate an answer using only retrieved context.
    """

    if not retrieved_chunks:

        return {
            "question": question,
            "answer": FALLBACK_MESSAGE,
            "context": "",
            "sources": [],
            "grounded": False,
            "retrieved_chunks": [],
        }

    prompt_data = build_prompt(
        question,
        retrieved_chunks,
    )

    if not prompt_data["prompt"]:

        return {
            "question": question,
            "answer": FALLBACK_MESSAGE,
            "context": "",
            "sources": [],
            "grounded": False,
            "retrieved_chunks": retrieved_chunks,
        }

    answer = generate_answer(
        question,
        prompt_data["prompt"],
    )

    return {
        "question": question,
        "answer": answer,
        "context": prompt_data["context"],
        "sources": prompt_data["sources_used"],
        "grounded": True,
        "retrieved_chunks": retrieved_chunks,
    }


# ---------------------------------------------------------
# Supporting Context Check
# ---------------------------------------------------------

def has_supporting_context(
    question: str,
    chunks: list[dict],
) -> bool:
    """
    Conservative lexical check to determine whether
    retrieved chunks contain meaningful terms from the question.
    """

    if not chunks:
        return False

    question_terms = set(
        re.findall(
            r"\b[a-zA-Z]{4,}\b",
            question.lower(),
        )
    )

    stop_words = {
        "what",
        "should",
        "could",
        "would",
        "does",
        "this",
        "that",
        "with",
        "from",
        "when",
        "where",
        "which",
        "have",
        "into",
        "your",
        "their",
        "there",
        "technician",
    }

    question_terms -= stop_words

    if not question_terms:
        return False

    combined_text = " ".join(
        str(chunk.get("text", ""))
        for chunk in chunks
    ).lower()

    matches = sum(
        1
        for term in question_terms
        if term in combined_text
    )

    return matches >= 1


# ---------------------------------------------------------
# Citation Number Validation
# ---------------------------------------------------------

def extract_citation_markers(
    answer: str,
) -> list[str]:
    """
    Extract citations such as [1], [2], [3].
    """

    if not answer:
        return []

    markers = re.findall(
        r"\[\d+\]",
        answer,
    )

    # Preserve order while removing duplicates.
    unique_markers = []

    for marker in markers:
        if marker not in unique_markers:
            unique_markers.append(marker)

    return unique_markers


def validate_citations(
    answer: str,
    chunks: list[dict],
) -> dict:
    """
    Verify that every citation in the answer maps
    to an actual retrieved chunk.
    """

    cited_markers = extract_citation_markers(
        answer
    )

    available_markers = {
        f"[{index}]"
        for index in range(
            1,
            len(chunks) + 1,
        )
    }

    valid_citations = [
        marker
        for marker in cited_markers
        if marker in available_markers
    ]

    invalid_citations = [
        marker
        for marker in cited_markers
        if marker not in available_markers
    ]

    return {
        "valid": len(invalid_citations) == 0,
        "cited_markers": cited_markers,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
    }


# ---------------------------------------------------------
# Source Accuracy Check
# ---------------------------------------------------------

def check_source_accuracy(
    answer: str,
    chunks: list[dict],
) -> dict:
    """
    Check whether cited sources contain supporting
    text for the claims in the answer.

    This is a simple verification aid, not a substitute
    for human review.
    """

    citation_validation = validate_citations(
        answer,
        chunks,
    )

    results = {}

    for marker in citation_validation["valid_citations"]:

        index = int(
            marker.strip("[]")
        ) - 1

        chunk = chunks[index]

        source_text = str(
            chunk.get(
                "text",
                "",
            )
        )

        source_lower = source_text.lower()

        # Remove the citation marker.
        answer_without_marker = answer.replace(
            marker,
            "",
        )

        answer_words = set(
            re.findall(
                r"\b[a-zA-Z]{4,}\b",
                answer_without_marker.lower(),
            )
        )

        source_words = set(
            re.findall(
                r"\b[a-zA-Z]{4,}\b",
                source_lower,
            )
        )

        matching_terms = sorted(
            answer_words.intersection(
                source_words
            )
        )

        supported = len(matching_terms) >= 2

        metadata = chunk.get(
            "metadata",
            {},
        )

        results[marker] = {
            "verified": supported,
            "source": metadata.get(
                "source",
                "unknown",
            ),
            "chunk_id": metadata.get(
                "chunk_id",
                chunk.get("id"),
            ),
            "chunk_index": metadata.get(
                "chunk_index",
            ),
            "section": metadata.get(
                "section",
            ),
            "page": metadata.get(
                "page",
            ),
            "original_text": source_text,
            "matching_terms": matching_terms,
        }

    return {
        "all_verified": (
            citation_validation["valid"]
            and all(
                item["verified"]
                for item in results.values()
            )
        ),
        "citation_validation": citation_validation,
        "results": results,
    }


# ---------------------------------------------------------
# Complete Query
# ---------------------------------------------------------

def answer_query(
    question: str,
    candidate_k: int = CANDIDATE_K,
    final_k: int = FINAL_K,
) -> dict:
    """
    Complete retrieval + reranking + generation flow.
    """

    candidates = retrieve_chunks(
        question,
        k=candidate_k,
    )

    if not candidates:

        return {
            "question": question,
            "answer": FALLBACK_MESSAGE,
            "context": "",
            "sources": [],
            "grounded": False,
            "retrieved_chunks": [],
            "citation_check": {
                "all_verified": False,
                "citation_validation": {},
                "results": {},
            },
        }

    # -----------------------------------------------------
    # Reranking
    # -----------------------------------------------------

    try:

        reranked = rerank_candidates(
            question,
            candidates,
            top_k=final_k,
        )

        if reranked:

            selected = []

            for item in reranked:

                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                ):
                    selected.append(
                        item[1]
                    )
                else:
                    selected.append(
                        item
                    )

            chunks = selected[:final_k]

        else:

            chunks = candidates[:final_k]

    except Exception:

        # If reranking fails, retrieval still works.
        chunks = candidates[:final_k]

    # -----------------------------------------------------
    # Supporting context check
    # -----------------------------------------------------

    if not has_supporting_context(
        question,
        chunks,
    ):

        return {
            "question": question,
            "answer": FALLBACK_MESSAGE,
            "context": format_context(chunks),
            "sources": [],
            "grounded": False,
            "retrieved_chunks": chunks,
            "citation_check": {
                "all_verified": False,
                "citation_validation": {},
                "results": {},
            },
        }

    # -----------------------------------------------------
    # Generate
    # -----------------------------------------------------

    result = generate_grounded_answer(
        question,
        chunks,
    )

    # -----------------------------------------------------
    # Citation validation
    # -----------------------------------------------------

    citation_check = check_source_accuracy(
        result["answer"],
        chunks,
    )

    result["citation_check"] = citation_check

    return result


# ---------------------------------------------------------
# Without Retrieval
# ---------------------------------------------------------

def generate_ungrounded_answer(
    question: str,
) -> str:
    """
    Generate an answer without retrieval.

    This is used only for Task 4:
    comparing answers with and without retrieval.
    """

    prompt = f"""
Answer the following question using your general knowledge.

Question:
{question}

Answer:
"""

    return generate_answer(
        question,
        prompt,
    )


# ---------------------------------------------------------
# Print Grounding Result
# ---------------------------------------------------------

def print_grounding_check(
    result: dict,
) -> None:

    print("\n" + "=" * 80)
    print("GROUNDED ANSWER")
    print("=" * 80)

    print(
        f"\nQuestion:\n{result['question']}"
    )

    print(
        f"\nAnswer:\n{result['answer']}"
    )

    print(
        f"\nGrounded: "
        f"{result.get('grounded', False)}"
    )

    print("\nSources:")

    for source in result.get(
        "sources",
        [],
    ):
        print(
            f"  - {source}"
        )

    print("\nCitation Check:")

    print(
        json.dumps(
            result.get(
                "citation_check",
                {},
            ),
            indent=2,
            default=str,
        )
    )


# ---------------------------------------------------------
# Save Report
# ---------------------------------------------------------

def save_report(
    supported: dict,
    fallback: dict,
    ungrounded: str,
) -> None:
    """
    Save assignment evidence to Markdown.
    """

    lines = []

    lines.append(
        "# Grounded Answer Generation Report"
    )

    lines.append("")

    # -----------------------------------------------------
    # Task 1
    # -----------------------------------------------------

    lines.append(
        "## Task 1 - Grounded Answer"
    )

    lines.append("")

    lines.append(
        f"**Question:** {supported['question']}"
    )

    lines.append("")

    lines.append(
        f"**Answer:** {supported['answer']}"
    )

    lines.append("")

    lines.append(
        "### Supporting Chunks"
    )

    lines.append("")

    for index, chunk in enumerate(
        supported.get(
            "retrieved_chunks",
            [],
        ),
        start=1,
    ):

        metadata = chunk.get(
            "metadata",
            {},
        )

        lines.append(
            f"#### [{index}] "
            f"{metadata.get('source', 'unknown')}"
        )

        lines.append("")

        lines.append(
            f"- Chunk ID: "
            f"{metadata.get('chunk_id', chunk.get('id'))}"
        )

        lines.append(
            f"- Chunk index: "
            f"{metadata.get('chunk_index')}"
        )

        lines.append(
            f"- Section: "
            f"{metadata.get('section')}"
        )

        lines.append("")

        lines.append(
            "```text"
        )

        lines.append(
            str(
                chunk.get(
                    "text",
                    "",
                )
            )
        )

        lines.append(
            "```"
        )

        lines.append("")

    # -----------------------------------------------------
    # Task 2
    # -----------------------------------------------------

    lines.append(
        "## Task 2 - Source Accuracy"
    )

    lines.append("")

    lines.append(
        "```json"
    )

    lines.append(
        json.dumps(
            supported.get(
                "citation_check",
                {},
            ),
            indent=2,
            default=str,
        )
    )

    lines.append(
        "```"
    )

    lines.append("")

    # -----------------------------------------------------
    # Task 3
    # -----------------------------------------------------

    lines.append(
        "## Task 3 - Missing Context Fallback"
    )

    lines.append("")

    lines.append(
        f"**Question:** {fallback['question']}"
    )

    lines.append("")

    lines.append(
        f"**Answer:** {fallback['answer']}"
    )

    lines.append("")

    lines.append(
        "Sources: "
        + str(
            fallback.get(
                "sources",
                [],
            )
        )
    )

    lines.append("")

    # -----------------------------------------------------
    # Task 4
    # -----------------------------------------------------

    lines.append(
        "## Task 4 - With vs Without Retrieval"
    )

    lines.append("")

    lines.append(
        f"**Question:** {supported['question']}"
    )

    lines.append("")

    lines.append(
        "### Without Retrieval"
    )

    lines.append("")

    lines.append(
        ungrounded
    )

    lines.append("")

    lines.append(
        "### With Retrieval"
    )

    lines.append("")

    lines.append(
        supported["answer"]
    )

    lines.append("")

    lines.append(
        "### Sources Used"
    )

    lines.append("")

    for source in supported.get(
        "sources",
        [],
    ):

        lines.append(
            f"- {source}"
        )

    lines.append("")

    OUTPUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print("=" * 80)
    print("GROUNDED ANSWER + SOURCE CITATION")
    print("=" * 80)

    # -----------------------------------------------------
    # Task 1 + Task 2
    # -----------------------------------------------------

    print(
        "\n[1] Testing grounded answer..."
    )

    supported = answer_query(
        SUPPORTED_QUERY,
        candidate_k=CANDIDATE_K,
        final_k=FINAL_K,
    )

    print_grounding_check(
        supported
    )

    # -----------------------------------------------------
    # Task 3
    # -----------------------------------------------------

    print(
        "\n[2] Testing missing-context fallback..."
    )

    fallback = answer_query(
        UNSUPPORTED_QUERY,
        candidate_k=CANDIDATE_K,
        final_k=FINAL_K,
    )

    print(
        f"\nQuestion:\n"
        f"{fallback['question']}"
    )

    print(
        f"\nAnswer:\n"
        f"{fallback['answer']}"
    )

    print(
        f"\nGrounded: "
        f"{fallback.get('grounded', False)}"
    )

    print(
        f"\nSources:\n"
        f"{fallback.get('sources', [])}"
    )

    # -----------------------------------------------------
    # Task 4
    # -----------------------------------------------------

    print(
        "\n[3] Comparing with and without retrieval..."
    )

    ungrounded = (
        generate_ungrounded_answer(
            SUPPORTED_QUERY
        )
    )

    print(
        "\n" + "-" * 80
    )

    print(
        "WITHOUT RETRIEVAL"
    )

    print(
        "-" * 80
    )

    print(
        ungrounded
    )

    print(
        "\n" + "-" * 80
    )

    print(
        "WITH RETRIEVAL"
    )

    print(
        "-" * 80
    )

    print(
        supported["answer"]
    )

    print(
        "\nSources:"
    )

    for source in supported.get(
        "sources",
        [],
    ):

        print(
            f"  - {source}"
        )

    # -----------------------------------------------------
    # Save report
    # -----------------------------------------------------

    save_report(
        supported,
        fallback,
        ungrounded,
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "✓ Report saved:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        "=" * 80
    )


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()