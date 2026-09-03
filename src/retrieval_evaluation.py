"""Evaluate retrieval against manually labelled relevant chunk IDs."""

from typing import Any, Iterable, Optional

try:
    from .retrieval import retrieve
    from .vector_store import VectorStore
except ImportError:
    from retrieval import retrieve
    from vector_store import VectorStore


def evaluate_query(
    item: dict,
    vector_store: Optional[VectorStore] = None,
    k: int = 5,
    retrieval_options: Optional[dict[str, Any]] = None,
) -> dict:
    """Evaluate one labelled query and return its detailed retrieval row."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    query = item.get("query", "")
    relevant = set(item.get("relevant_chunk_ids", set()))
    if not query or not isinstance(query, str):
        raise ValueError("Each labelled query must have a non-empty string query")
    if not relevant:
        raise ValueError(f"Query '{query}' must have at least one relevant chunk ID")

    options = dict(retrieval_options or {})
    response = retrieve(query, vector_store=vector_store, top_k=k, **options)
    retrieved_ids = [result.chunk_id for result in response.results]
    hits = [chunk_id for chunk_id in retrieved_ids if chunk_id in relevant]

    return {
        "query": query,
        "retrieved_ids": retrieved_ids,
        "relevant_chunk_ids": sorted(relevant),
        "hits": hits,
        "recall": len(set(hits)) / len(relevant),
        "precision": len(hits) / len(retrieved_ids) if retrieved_ids else 0.0,
    }


def evaluate_queries(
    labelled_queries: Iterable[dict],
    vector_store: Optional[VectorStore] = None,
    k: int = 5,
    retrieval_options: Optional[dict[str, Any]] = None,
) -> list[dict]:
    """Evaluate every item in a labelled query set."""
    return [
        evaluate_query(item, vector_store, k, retrieval_options)
        for item in labelled_queries
    ]


def summarize_results(rows: list[dict], k: int = 5) -> dict:
    """Return macro-average recall and precision plus failed queries."""
    if not rows:
        raise ValueError("At least one evaluation row is required")

    return {
        "queries": len(rows),
        "k": k,
        "recall_at_k": sum(row["recall"] for row in rows) / len(rows),
        "precision_at_k": sum(row["precision"] for row in rows) / len(rows),
        "failures": [row for row in rows if row["recall"] < 1.0],
    }


def evaluate(
    labelled_queries: Iterable[dict],
    vector_store: Optional[VectorStore] = None,
    k: int = 5,
    retrieval_options: Optional[dict[str, Any]] = None,
) -> dict:
    """Evaluate a labelled query set and return rows with aggregate metrics."""
    rows = evaluate_queries(labelled_queries, vector_store, k, retrieval_options)
    return {"rows": rows, "summary": summarize_results(rows, k)} if rows else {
        "rows": [],
        "summary": {"queries": 0, "k": k, "recall_at_k": 0.0, "precision_at_k": 0.0, "failures": []},
    }


def print_evaluation_report(report: dict) -> None:
    """Print aggregate metrics and the expected/retrieved IDs for failures."""
    summary = report["summary"]
    print(f"queries: {summary['queries']}")
    print(f"recall@{summary['k']}: {summary['recall_at_k']:.3f}")
    print(f"precision@{summary['k']}: {summary['precision_at_k']:.3f}")

    for failure in summary["failures"]:
        print(f"failed query: {failure['query']}")
        print(f"expected: {failure['relevant_chunk_ids']}")
        print(f"retrieved: {failure['retrieved_ids']}")