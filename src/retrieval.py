"""Similarity search and top-k retrieval from indexed corpus."""

from dataclasses import dataclass
from typing import Optional

try:
    from .embeddings import embed
    from .vector_store import VectorStore
except ImportError:
    from embeddings import embed
    from vector_store import VectorStore


@dataclass
class RetrievalResult:
    """Single search result with score, text, and metadata."""

    rank: int
    score: float  # Distance score (lower is better for cosine)
    similarity: float  # Similarity score (higher is better, 0-1)
    text: str
    metadata: dict
    chunk_id: str


@dataclass
class RetrievalResponse:
    """Complete retrieval response for a query."""

    query: str
    top_k: int
    total_retrieved: int
    results: list[RetrievalResult]
    query_embedding: list[float]


def retrieve(
    query: str,
    vector_store: Optional[VectorStore] = None,
    top_k: int = 3,
    where: Optional[dict] = None,
) -> RetrievalResponse:
    """Retrieve top-k chunks most similar to query.

    Args:
        query: User query string
        vector_store: Initialized VectorStore (creates default if None)
        top_k: Number of results to return
        where: Optional metadata filter dict

    Returns:
        RetrievalResponse with ranked results
    """
    if vector_store is None:
        vector_store = VectorStore()

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    # Embed query with same model as documents
    query_embedding = embed([query])[0]

    # Search vector database
    search_results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        where=where,
    )

    # Format results
    formatted_results = [
        RetrievalResult(
            rank=rank,
            score=result["distance"],
            similarity=result["similarity"],
            text=result["text"],
            metadata=result["metadata"],
            chunk_id=result["id"],
        )
        for rank, result in enumerate(search_results, 1)
    ]

    return RetrievalResponse(
        query=query,
        top_k=top_k,
        total_retrieved=len(formatted_results),
        results=formatted_results,
        query_embedding=query_embedding,
    )


def retrieve_with_context(
    query: str,
    vector_store: Optional[VectorStore] = None,
    top_k: int = 3,
) -> str:
    """Retrieve and format results as context string for LLM.

    Args:
        query: User query
        vector_store: Initialized VectorStore
        top_k: Number of results

    Returns:
        Formatted context string ready for prompt
    """
    response = retrieve(query, vector_store, top_k)

    if not response.results:
        return "(No relevant context found)"

    context_parts = [f"Retrieved {len(response.results)} relevant chunks:"]

    for result in response.results:
        source = result.metadata.get("source", "unknown")
        section = result.metadata.get("section", "")
        section_str = f" / {section}" if section else ""

        context_parts.append(f"\n[{result.rank}] Similarity: {result.similarity:.4f}")
        context_parts.append(f"    Source: {source}{section_str}")
        context_parts.append(f"    Text: {result.text}")

    return "\n".join(context_parts)


def compare_k_values(
    query: str,
    k_values: list[int],
    vector_store: Optional[VectorStore] = None,
) -> dict:
    """Compare retrieval results for different k values.

    Args:
        query: User query
        k_values: List of k values to test (e.g., [1, 3, 5])
        vector_store: Initialized VectorStore

    Returns:
        Dict with results for each k value
    """
    if vector_store is None:
        vector_store = VectorStore()

    results = {}
    for k in sorted(k_values):
        response = retrieve(query, vector_store, top_k=k)
        results[k] = response

    return results


def print_retrieval_results(response: RetrievalResponse) -> None:
    """Pretty-print retrieval results.

    Args:
        response: RetrievalResponse to display
    """
    print("\n" + "=" * 80)
    print("SIMILARITY SEARCH RESULTS")
    print("=" * 80)
    print(f"\nQuery: {response.query}")
    print(f"Top-k: {response.top_k}")
    print(f"Retrieved: {response.total_retrieved} chunks\n")

    if not response.results:
        print("(No results found)")
        return

    for result in response.results:
        print(f"[{result.rank}] Similarity: {result.similarity:.4f} | Distance: {result.score:.4f}")
        print(f"    ID: {result.chunk_id}")
        print(f"    Source: {result.metadata.get('source', 'unknown')}")
        if result.metadata.get("section"):
            print(f"    Section: {result.metadata['section']}")
        print(f"    Chunk Index: {result.metadata.get('chunk_index', 'N/A')}")
        print(f"    Text: {result.text[:100]}...")
        print()


def print_k_comparison(
    query: str,
    comparison_results: dict[int, RetrievalResponse],
) -> None:
    """Print comparison of retrieval results across k values.

    Args:
        query: Original query
        comparison_results: Dict with results for each k from compare_k_values()
    """
    print("\n" + "=" * 80)
    print("TOP-K COMPARISON")
    print("=" * 80)
    print(f"\nQuery: {query}\n")

    for k in sorted(comparison_results.keys()):
        response = comparison_results[k]
        print(f"{'─' * 80}")
        print(f"k = {k} ({response.total_retrieved} results)\n")

        for result in response.results:
            print(f"  [{result.rank}] {result.similarity:.4f} | {result.text[:70]}...")
            print(f"      Source: {result.metadata.get('source')} | "
                  f"Section: {result.metadata.get('section', 'N/A')}")

        print()


def analyze_retrieval_quality(response: RetrievalResponse) -> dict:
    """Analyze quality metrics of retrieval results.

    Args:
        response: RetrievalResponse to analyze

    Returns:
        Dict with quality metrics
    """
    if not response.results:
        return {
            "num_results": 0,
            "avg_similarity": 0.0,
            "min_similarity": 0.0,
            "max_similarity": 0.0,
            "similarity_spread": 0.0,
        }

    similarities = [r.similarity for r in response.results]

    return {
        "num_results": len(response.results),
        "avg_similarity": sum(similarities) / len(similarities),
        "min_similarity": min(similarities),
        "max_similarity": max(similarities),
        "similarity_spread": max(similarities) - min(similarities),
        "top_result_quality": "HIGH" if similarities[0] > 0.7 else "MEDIUM" if similarities[0] > 0.5 else "LOW",
    }


if __name__ == "__main__":
    # Example usage
    import sys

    store = VectorStore()

    # Test retrieval
    test_queries = [
        "How do I reset my password?",
        "What are safety procedures?",
        "Where is the maintenance log?",
    ]

    print("Testing Similarity Search & Top-K Retrieval\n")

    for query in test_queries:
        print(f"\nQuery: {query}")
        
        try:
            # Single retrieval
            response = retrieve(query, store, top_k=3)
            print_retrieval_results(response)

            # Quality analysis
            metrics = analyze_retrieval_quality(response)
            print("Quality Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        except ValueError as e:
            print(f"  Error: {e}")
            if "dimension" in str(e).lower():
                print("  Note: Vector store is empty. Run indexing first.")

    print("\n" + "=" * 80)
