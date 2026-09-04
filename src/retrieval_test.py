"""Deterministic retrieval tests for metadata-filtered and hybrid search."""

import math
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent

for entry in [str(ROOT), str(SRC)]:
    if entry not in sys.path:
        sys.path.insert(0, entry)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from src.retrieval import retrieve
    from src.retrieval_evaluation import evaluate, evaluate_query
except ImportError:
    from retrieval import retrieve
    from retrieval_evaluation import evaluate, evaluate_query


# ---------------------------------------------------------------------------
# Deterministic in-memory vector store
# ---------------------------------------------------------------------------

class DemoStore:
    """Small in-memory vector store used for deterministic tests."""

    def __init__(self, documents):
        self.documents = documents

    def count(self):
        return len(self.documents)

    def search(self, query_embedding, top_k=5, where=None):
        matches = []

        for document in self.documents:

            metadata = document["metadata"]

            # Metadata filtering
            if where and not all(
                metadata.get(key) == value
                for key, value in where.items()
            ):
                continue

            similarity = _cosine_similarity(
                query_embedding,
                document["embedding"],
            )

            matches.append(
                {
                    "id": document["id"],
                    "distance": 1.0 - similarity,
                    "similarity": similarity,
                    "text": document["text"],
                    "metadata": metadata,
                    "embedding": document["embedding"],
                }
            )

        matches.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        return matches[:top_k]


# ---------------------------------------------------------------------------
# Similarity helper
# ---------------------------------------------------------------------------

def _cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""

    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        sum(x * x for x in a)
    )

    norm_b = math.sqrt(
        sum(x * x for x in b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------

def create_test_corpus():
    """
    Create a deterministic corpus.

    The vectors are deliberately constructed so that:

    - Account queries favor account documents.
    - The unfiltered maintenance test favors a non-maintenance document.
    - Metadata filtering forces maintenance documents.
    - Hybrid keyword scoring promotes the vibration document.
    """

    documents = [

        # ---------------------------------------------------------------
        # Account documents
        # ---------------------------------------------------------------

        {
            "id": "account-1",
            "text": (
                "How to reset your password: "
                "Click on Forgot Password on the login page."
            ),
            "metadata": {
                "source": "account-guide.md",
                "section": "Account Access",
            },
            "embedding": (
                [0.9] + [0.1] * 1535
            ),
        },

        {
            "id": "account-2",
            "text": (
                "Password reset links expire after "
                "24 hours for security."
            ),
            "metadata": {
                "source": "account-guide.md",
                "section": "Security",
            },
            "embedding": (
                [0.8] + [0.2] * 1535
            ),
        },

        # ---------------------------------------------------------------
        # Safety document
        #
        # This is intentionally the strongest semantic match for the
        # unfiltered maintenance query.
        # ---------------------------------------------------------------

        {
            "id": "safety-1",
            "text": (
                "Safety procedures: Always follow emergency "
                "protocols before inspection."
            ),
            "metadata": {
                "source": "safety-guide.md",
                "section": "Emergency",
            },
            "embedding": (
                [0.15] + [0.95] * 1535
            ),
        },

        # ---------------------------------------------------------------
        # Maintenance documents
        #
        # Slightly lower semantic similarity than safety-1.
        # Filtering should therefore be what selects these.
        # ---------------------------------------------------------------

        {
            "id": "maintenance-1",
            "text": (
                "Maintenance schedule: Equipment is serviced "
                "every 30 days."
            ),
            "metadata": {
                "source": "maintenance.md",
                "section": "Schedule",
            },
            "embedding": (
                [0.10] + [0.85] * 1535
            ),
        },

        {
            "id": "maintenance-2",
            "text": (
                "Report any equipment issues immediately "
                "to the maintenance team."
            ),
            "metadata": {
                "source": "maintenance.md",
                "section": "Procedures",
            },
            "embedding": (
                [0.12] + [0.88] * 1535
            ),
        },

        {
            "id": "maintenance-3",
            "text": (
                "Vibration analysis shows abnormal oscillation "
                "near the motor casing."
            ),
            "metadata": {
                "source": "maintenance.md",
                "section": "Diagnostics",
            },
            "embedding": (
                [0.14] + [0.90] * 1535
            ),
        },

        {
            "id": "maintenance-4",
            "text": (
                "Keep maintenance logs updated for all "
                "work performed."
            ),
            "metadata": {
                "source": "maintenance.md",
                "section": "Documentation",
            },
            "embedding": (
                [0.13] + [0.78] * 1535
            ),
        },
    ]

    return DemoStore(documents)


# ---------------------------------------------------------------------------
# Test 1: basic retrieval
# ---------------------------------------------------------------------------

def test_retrieval_with_dummy_embeddings():
    """
    Check top-k ordering with a deterministic in-memory corpus.
    """

    store = create_test_corpus()

    query_embedding = (
        [0.95] + [0.05] * 1535
    )

    response = retrieve(
        "reset password",
        store,
        top_k=3,
        query_embedding=query_embedding,
    )

    assert (
        response.results[0].metadata["source"]
        == "account-guide.md"
    )

    assert response.total_retrieved == 3

    assert response.results[0].similarity > 0.8


# ---------------------------------------------------------------------------
# Test 2: metadata filtering + hybrid retrieval
# ---------------------------------------------------------------------------

def test_filtered_hybrid_retrieval():
    """
    Ensure metadata filtering and keyword boosting improve precision.
    """

    store = create_test_corpus()

    query_embedding = (
        [0.15] + [0.95] * 1535
    )

    # ---------------------------------------------------------------
    # Unfiltered retrieval
    # ---------------------------------------------------------------

    unfiltered = retrieve(
        "vibration diagnosis and maintenance logs",
        store,
        top_k=3,
        query_embedding=query_embedding,
    )

    # ---------------------------------------------------------------
    # Metadata-filtered retrieval
    # ---------------------------------------------------------------

    filtered = retrieve(
        "vibration diagnosis and maintenance logs",
        store,
        top_k=3,
        where={
            "source": "maintenance.md"
        },
        query_embedding=query_embedding,
    )

    # ---------------------------------------------------------------
    # Hybrid retrieval
    # ---------------------------------------------------------------

    hybrid = retrieve(
        "vibration diagnosis and maintenance logs",
        store,
        top_k=3,
        where={
            "source": "maintenance.md"
        },
        keyword_terms=[
            "vibration",
            "maintenance",
            "diagnosis",
        ],
        hybrid_weight=0.6,
        query_embedding=query_embedding,
    )

    # ---------------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------------

    # Without filtering, the top result should not be maintenance.
    assert (
        unfiltered.results[0].metadata["source"]
        != "maintenance.md"
    )

    # Filtering should force maintenance results.
    assert (
        filtered.results[0].metadata["source"]
        == "maintenance.md"
    )

    # Hybrid retrieval should also return maintenance.
    assert (
        hybrid.results[0].metadata["source"]
        == "maintenance.md"
    )

    # Keyword boosting should preserve the vibration result.
    assert any(
        "vibration" in result.text.lower()
        for result in hybrid.results
    )


# ---------------------------------------------------------------------------
# Test 3: retrieval evaluation
# ---------------------------------------------------------------------------

def test_evaluation_reports_recall_precision_and_failures():
    """
    Measure labelled IDs and retain failed-query details.
    """

    store = create_test_corpus()

    query_embedding = (
        [0.95] + [0.05] * 1535
    )

    labelled_queries = [
        {
            "query": "reset password",
            "relevant_chunk_ids": {
                "account-1",
                "account-2",
            },
        }
    ]

    report = evaluate(
        labelled_queries,
        store,
        k=1,
        retrieval_options={
            "query_embedding": query_embedding
        },
    )

    row = report["rows"][0]

    assert row["retrieved_ids"] == [
        "account-1"
    ]

    assert row["hits"] == [
        "account-1"
    ]

    assert row["recall"] == 0.5

    assert row["precision"] == 1.0

    assert (
        report["summary"]["queries"]
        == 1
    )

    assert (
        report["summary"]["failures"]
        == [row]
    )


# ---------------------------------------------------------------------------
# Test 4: validation of evaluation labels
# ---------------------------------------------------------------------------

def test_evaluation_rejects_missing_labels_and_empty_sets():
    """
    Labels must be trustworthy and an empty suite has zero metrics.
    """

    store = create_test_corpus()

    with _expect_value_error():

        evaluate_query(
            {
                "query": "reset password",
                "relevant_chunk_ids": set(),
            },
            store,
        )

    report = evaluate(
        [],
        store,
    )

    assert (
        report["summary"]["recall_at_k"]
        == 0.0
    )

    assert (
        report["summary"]["precision_at_k"]
        == 0.0
    )


# ---------------------------------------------------------------------------
# Context manager for expected ValueError
# ---------------------------------------------------------------------------

class _expect_value_error:

    def __enter__(self):
        return self

    def __exit__(
        self,
        exception_type,
        exception,
        traceback,
    ):
        assert exception_type is ValueError
        return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    test_filtered_hybrid_retrieval()

    test_retrieval_with_dummy_embeddings()

    test_evaluation_reports_recall_precision_and_failures()

    test_evaluation_rejects_missing_labels_and_empty_sets()

    print("retrieval tests passed")