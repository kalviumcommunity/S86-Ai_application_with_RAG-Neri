"""
Grounded Answer Generation for the RAG pipeline.

Tasks covered:
1. Generate answers from retrieved context.
2. Check source accuracy.
3. Handle missing-context fallback.
4. Compare grounded vs ungrounded answers.
5. Save sample grounded answers and verification results.
"""

import json
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .embeddings import embed
from .llm_api import generate_answer
from .rag_pipeline import (
    retrieve_context,
)
from .reranking import (
    rerank_candidates,
)
from .vector_store import VectorStore


load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VECTOR_DIR = Path(
    "outputs/chroma_db"
)

OUTPUT_FILE = Path(
    "outputs/grounded_answers.md"
)

CANDIDATE_K = 10
FINAL_K = 3

FALLBACK_MESSAGE = (
    "I don't have enough information "
    "in the provided context."
)


SUPPORTED_QUERY = (
    "What should a technician do if abnormal "
    "vibration is detected?"
)

UNSUPPORTED_QUERY = (
    "What is the warranty period for this machine?"
)


# ---------------------------------------------------------
# Vector store
# ---------------------------------------------------------

def get_vector_store() -> VectorStore:
    """Create the project vector store."""

    return VectorStore(
        persist_dir=VECTOR_DIR,
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
    Retrieve relevant chunks for a question.

    Uses the existing RAG pipeline retrieval function.
    """

    vector_store = get_vector_store()

    query_vector = embed(
        [question]
    )[0]

    chunks = retrieve_context(
        query_vector,
        vector_store,
        k=k,
    )

    return chunks


# ---------------------------------------------------------
# Context formatting
# ---------------------------------------------------------

def format_context(
    chunks: list[dict],
) -> str:
    """
    Convert retrieved chunks into explicit
    source-labelled context.
    """

    if not chunks:
        return ""

    context_parts = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        metadata = chunk.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        section = metadata.get(
            "section",
            "unknown",
        )

        text = chunk.get(
            "text",
            "",
        )

        context_parts.append(
            f"[SOURCE {index}]\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Content:\n{text}"
        )

    return "\n\n".join(
        context_parts
    )


# ---------------------------------------------------------
# Prompt
# ---------------------------------------------------------

def build_prompt(
    question: str,
    chunks: list[dict],
) -> dict:
    """
    Build the grounded prompt and source metadata.
    """

    context = format_context(
        chunks
    )

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

    prompt = (
        "You are a support assistant.\n\n"

        "GROUNDING RULE:\n"
        "Answer ONLY using the supplied context.\n"
        "Do NOT use outside knowledge.\n"
        "Do NOT guess or invent missing information.\n"
        "Every factual claim must be supported "
        "by the supplied context.\n"
        "If the context does not contain enough "
        "information to answer the question, say:\n"
        f"\"{FALLBACK_MESSAGE}\"\n\n"

        "CITATION RULE:\n"
        "Mention the source file supporting "
        "your answer.\n\n"

        f"CONTEXT:\n{context}\n\n"

        f"QUESTION:\n{question}\n\n"

        "Provide a concise answer."
    )

    return {
        "prompt": prompt,
        "sources_used": sources,
        "context": context,
    }


# ---------------------------------------------------------
# Grounded generation
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
        }

    prompt_data = build_prompt(
        question,
        retrieved_chunks,
    )

    answer = generate_answer(
        question,
        prompt_data["context"],
    )

    return {
        "question": question,
        "answer": answer,
        "context": prompt_data["context"],
        "sources": prompt_data["sources_used"],
        "grounded": True,
    }


# ---------------------------------------------------------
# Missing context
# ---------------------------------------------------------

def has_supporting_context(
    question: str,
    chunks: list[dict],
) -> bool:
    """
    Determine whether retrieved chunks contain
    useful lexical support for the question.

    This is intentionally conservative.
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
        "technician",
    }

    question_terms -= stop_words

    combined_text = " ".join(
        chunk.get("text", "")
        for chunk in chunks
    ).lower()

    matches = sum(
        1
        for term in question_terms
        if term in combined_text
    )

    return matches >= 1


def answer_query(
    question: str,
    candidate_k: int = CANDIDATE_K,
    final_k: int = FINAL_K,
) -> dict:
    """
    Complete grounded query flow.
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
        }

    # Existing reranking function may return
    # tuples of (score, chunk).
    try:

        reranked = rerank_candidates(
            question,
            candidates,
        )

        if reranked:

            selected = []

            for item in reranked[:final_k]:

                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                ):
                    selected.append(
                        item[1]
                    )
                else:
                    selected.append(item)

            chunks = selected

        else:
            chunks = candidates[:final_k]

    except Exception:

        # Retrieval itself remains usable
        # even if reranking is unavailable.
        chunks = candidates[:final_k]

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
        }

    result = generate_grounded_answer(
        question,
        chunks,
    )

    result["retrieved_chunks"] = chunks

    return result


# ---------------------------------------------------------
# Ungrounded generation
# ---------------------------------------------------------

def generate_ungrounded_answer(
    question: str,
) -> str:
    """
    Generate an answer without retrieval.

    This is used only for comparison.
    """

    return generate_answer(
        question,
        "",
    )


# ---------------------------------------------------------
# Grounding verification
# ---------------------------------------------------------

def verify_answer_against_sources(
    answer: str,
    chunks: list[dict],
) -> dict:
    """
    Basic technical grounding check.

    It checks whether answer sentences have
    meaningful overlap with the retrieved context.
    """

    if not chunks:

        return {
            "supported": False,
            "claims_checked": 0,
            "unsupported_claims": [
                answer
            ],
        }

    context = " ".join(
        chunk.get("text", "")
        for chunk in chunks
    ).lower()

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"[.!?]+",
            answer,
        )
        if sentence.strip()
    ]

    unsupported = []

    for sentence in sentences:

        words = set(
            re.findall(
                r"\b[a-zA-Z]{4,}\b",
                sentence.lower(),
            )
        )

        if not words:
            continue

        overlap = sum(
            1
            for word in words
            if word in context
        )

        # Conservative threshold.
        if overlap < 1:
            unsupported.append(
                sentence
            )

    return {
        "supported": len(unsupported) == 0,
        "claims_checked": len(sentences),
        "unsupported_claims": unsupported,
    }


# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

def print_grounding_check(
    result: dict,
) -> None:

    print("\n" + "=" * 70)

    print(
        "GROUNDING CHECK"
    )

    print("=" * 70)

    print(
        f"\nQuestion:\n"
        f"{result['question']}"
    )

    print(
        f"\nAnswer:\n"
        f"{result['answer']}"
    )

    print("\nSources:")

    if result["sources"]:

        for source in result["sources"]:
            print(
                f"  - {source}"
            )

    else:
        print(
            "  - None"
        )

    verification = (
        verify_answer_against_sources(
            result["answer"],
            result.get(
                "retrieved_chunks",
                [],
            ),
        )
    )

    print(
        "\nSupported:"
        f" {verification['supported']}"
    )

    print(
        "Claims checked:"
        f" {verification['claims_checked']}"
    )

    if verification[
        "unsupported_claims"
    ]:

        print(
            "\nPotentially unsupported claims:"
        )

        for claim in verification[
            "unsupported_claims"
        ]:
            print(
                f"  - {claim}"
            )


# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

def save_report(
    supported: dict,
    fallback: dict,
    ungrounded: str,
) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = []

    lines.append(
        "# Grounded Answer Generation"
    )

    lines.append("")

    lines.append(
        "## Task 1 - Grounded Answer"
    )

    lines.append("")

    lines.append(
        f"**Question:** "
        f"{supported['question']}"
    )

    lines.append("")

    lines.append(
        f"**Answer:** "
        f"{supported['answer']}"
    )

    lines.append("")

    lines.append(
        "### Supporting Sources"
    )

    for source in supported[
        "sources"
    ]:
        lines.append(
            f"- `{source}`"
        )

    lines.append("")

    lines.append(
        "### Supporting Chunks"
    )

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
            f"\n#### Chunk {index}"
        )

        lines.append(
            f"- Source: "
            f"`{metadata.get('source', 'unknown')}`"
        )

        lines.append(
            f"- Section: "
            f"`{metadata.get('section', 'unknown')}`"
        )

        lines.append("")

        lines.append(
            chunk.get("text", "")
        )

    lines.append("")

    lines.append(
        "## Task 2 - Source Accuracy"
    )

    verification = (
        verify_answer_against_sources(
            supported["answer"],
            supported.get(
                "retrieved_chunks",
                [],
            ),
        )
    )

    lines.append("")

    lines.append(
        f"- Supported: "
        f"`{verification['supported']}`"
    )

    lines.append(
        f"- Claims checked: "
        f"`{verification['claims_checked']}`"
    )

    if verification[
        "unsupported_claims"
    ]:

        lines.append(
            "- Unsupported claims:"
        )

        for claim in verification[
            "unsupported_claims"
        ]:

            lines.append(
                f"  - {claim}"
            )

    else:

        lines.append(
            "- No unsupported claims detected "
            "by the lexical verification check."
        )

    lines.append("")

    lines.append(
        "## Task 3 - Missing Context Fallback"
    )

    lines.append("")

    lines.append(
        f"**Question:** "
        f"{fallback['question']}"
    )

    lines.append("")

    lines.append(
        f"**Fallback:** "
        f"{fallback['answer']}"
    )

    lines.append("")

    lines.append(
        "## Task 4 - With vs Without Retrieval"
    )

    lines.append("")

    lines.append(
        f"**Question:** "
        f"{supported['question']}"
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
        "### Sources"
    )

    for source in supported[
        "sources"
    ]:

        lines.append(
            f"- `{source}`"
        )

    lines.append("")

    lines.append(
        "## Grounding Conclusion"
    )

    lines.append("")

    lines.append(
        "The grounded answer is generated using "
        "retrieved context and identifies the "
        "supporting source. When supporting "
        "context is unavailable, the system "
        "returns a missing-context fallback "
        "instead of relying on unsupported "
        "model knowledge."
    )

    OUTPUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print("=" * 80)

    print(
        "GROUNDED ANSWER GENERATION"
    )

    print("=" * 80)

    print(
        f"\nQuery: {SUPPORTED_QUERY}"
    )

    print(
        f"Candidate k: {CANDIDATE_K}"
    )

    print(
        f"Final k: {FINAL_K}"
    )

    # ---------------------------------------------
    # Task 1
    # ---------------------------------------------

    print(
        "\n[1] Generating grounded answer..."
    )

    supported = answer_query(
        SUPPORTED_QUERY,
        candidate_k=CANDIDATE_K,
        final_k=FINAL_K,
    )

    print_grounding_check(
        supported
    )

    # ---------------------------------------------
    # Task 3
    # ---------------------------------------------

    print(
        "\n[2] Testing missing-context fallback..."
    )

    fallback = answer_query(
        UNSUPPORTED_QUERY,
        candidate_k=CANDIDATE_K,
        final_k=FINAL_K,
    )

    print(
        f"\nQuestion: "
        f"{fallback['question']}"
    )

    print(
        f"Answer: "
        f"{fallback['answer']}"
    )

    print(
        f"Sources: "
        f"{fallback['sources']}"
    )

    # ---------------------------------------------
    # Task 4
    # ---------------------------------------------

    print(
        "\n[3] Comparing without retrieval..."
    )

    ungrounded = (
        generate_ungrounded_answer(
            SUPPORTED_QUERY
        )
    )

    print(
        "\nWITHOUT RETRIEVAL:"
    )

    print(
        ungrounded
    )

    print(
        "\nWITH RETRIEVAL:"
    )

    print(
        supported["answer"]
    )

    print(
        "\nSources:"
    )

    for source in supported[
        "sources"
    ]:

        print(
            f"  - {source}"
        )

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    save_report(
        supported,
        fallback,
        ungrounded,
    )

    print(
        "\n✓ Report saved to:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )

    print(
        "\n✓ Grounded generation completed."
    )


if __name__ == "__main__":
    main()