"""Interactive retrieval demonstration and testing."""

import argparse
import math
import sys

try:
    from retrieval import (
        retrieve,
        retrieve_with_context,
        compare_k_values,
        compare_filtered_retrieval,
        print_retrieval_results,
        print_k_comparison,
        analyze_retrieval_quality,
    )
except ImportError:
    print("Error: Could not import required modules. Run from src/ directory.")
    sys.exit(1)


class DemoVectorStore:
    """In-memory vector store with metadata filters for deterministic demos."""

    def __init__(self, documents):
        self.documents = documents

    def count(self):
        return len(self.documents)

    def search(self, query_embedding, top_k=5, where=None):
        matches = []
        for document in self.documents:
            metadata = document["metadata"]
            if where and not all(metadata.get(key) == value for key, value in where.items()):
                continue
            similarity = _cosine_similarity(query_embedding, document["embedding"])
            matches.append({
                "id": document["id"],
                "distance": 1.0 - similarity,
                "similarity": similarity,
                "text": document["text"],
                "metadata": metadata,
                "embedding": document["embedding"],
            })
        matches.sort(key=lambda item: item["similarity"], reverse=True)
        return matches[:top_k]


def _cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_demo_store():
    """Create a small maintenance corpus that is useful for filtered retrieval demos."""
    documents = [
        {
            "id": "manual-1",
            "text": "Before performing maintenance, disconnect the machine from the main power supply.",
            "metadata": {"source": "machine_manual.txt", "section": "Safety"},
            "embedding": [0.82] + [0.12] * 1535,
        },
        {
            "id": "manual-2",
            "text": "Inspect the lubrication system and check for leaks before restart.",
            "metadata": {"source": "machine_manual.txt", "section": "Inspection"},
            "embedding": [0.75] + [0.18] * 1535,
        },
        {
            "id": "safety-1",
            "text": "Technicians must wear the required protective equipment before beginning maintenance activities.",
            "metadata": {"source": "safety_procedure.md", "section": "PPE"},
            "embedding": [0.2] + [0.9] * 1535,
        },
        {
            "id": "safety-2",
            "text": "Before inspecting or repairing equipment, disconnect the main power supply and verify isolation.",
            "metadata": {"source": "safety_procedure.md", "section": "Electrical Safety"},
            "embedding": [0.85] + [0.1] * 1535,
        },
        {
            "id": "log-1",
            "text": "The technician replaced a worn belt after detecting abnormal vibration.",
            "metadata": {"source": "maintenance_log.txt", "section": "Diagnostics"},
            "embedding": [0.1] + [0.96] * 1535,
        },
        {
            "id": "log-2",
            "text": "The machine was tested after repair and returned to service.",
            "metadata": {"source": "maintenance_log.txt", "section": "Service"},
            "embedding": [0.12] + [0.82] * 1535,
        },
    ]
    return DemoVectorStore(documents)


def demo_basic_retrieval(vector_store):
    """Demonstrate basic retrieval with a single query."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Retrieval")
    print("=" * 80)

    query = "Before inspecting the machine, disconnect the power and verify isolation"
    query_embedding = [0.91] + [0.09] * 1535
    response = retrieve(query, vector_store, top_k=3, query_embedding=query_embedding)
    print_retrieval_results(response)

    metrics = analyze_retrieval_quality(response)
    print("Quality Metrics:")
    print(f"  Average similarity: {metrics['avg_similarity']:.4f}")
    print(f"  Top result quality: {metrics['top_result_quality']}")


def demo_k_comparison(vector_store):
    """Demonstrate how changing k affects results."""
    print("\n" + "=" * 80)
    print("DEMO 2: Top-K Comparison (How k affects results)")
    print("=" * 80)

    query = "vibration maintenance and power isolation"
    query_embedding = [0.88] + [0.12] * 1535
    comparison = {
        1: retrieve(query, vector_store, top_k=1, query_embedding=query_embedding),
        3: retrieve(query, vector_store, top_k=3, query_embedding=query_embedding),
        5: retrieve(query, vector_store, top_k=5, query_embedding=query_embedding),
    }
    print_k_comparison(query, comparison)


def demo_context_building(vector_store):
    """Demonstrate building LLM context from retrieval."""
    print("\n" + "=" * 80)
    print("DEMO 3: Building LLM Context from Retrieval")
    print("=" * 80)

    query = "What should I do before inspecting a machine?"
    query_embedding = [0.90] + [0.10] * 1535
    context = retrieve_with_context(query, vector_store, top_k=3)
    print("Formatted Context (ready for LLM):")
    print("-" * 80)
    print(context)
    print("-" * 80)


def demo_filtered_search(vector_store):
    """Compare unfiltered and metadata-filtered retrieval, with optional keyword boost."""
    print("\n" + "=" * 80)
    print("DEMO 4: Filtered retrieval with hybrid keyword boost")
    print("=" * 80)

    query = "Before inspecting or repairing equipment, disconnect the main power supply and verify isolation"
    keyword_terms = ["disconnect", "power", "isolation", "inspection"]
    query_embedding = [0.86] + [0.14] * 1535
    comparison = compare_filtered_retrieval(
        query=query,
        vector_store=vector_store,
        where={"source": "safety_procedure.md"},
        top_k=3,
        keyword_terms=keyword_terms,
        hybrid_weight=0.5,
    )

    print(f"\nQuery: {comparison['query']}")
    print(f"Filter: {comparison['filter']}")
    print("\nUnfiltered results:")
    for result in comparison["unfiltered"].results:
        print(f"  [{result.rank}] {result.similarity:.4f} | {result.metadata.get('source')} | {result.metadata.get('section')} | {result.text[:90]}...")

    print("\nFiltered results:")
    for result in comparison["filtered"].results:
        print(f"  [{result.rank}] {result.similarity:.4f} | {result.metadata.get('source')} | {result.metadata.get('section')} | {result.text[:90]}...")

    if comparison["filtered"].results and comparison["unfiltered"].results:
        print(f"\nPrecision effect: filtered top result similarity {comparison['filtered'].results[0].similarity:.4f} vs unfiltered {comparison['unfiltered'].results[0].similarity:.4f}")


def interactive_search(vector_store):
    """Interactive query mode."""
    print("\n" + "=" * 80)
    print("INTERACTIVE SEARCH")
    print("=" * 80)
    print("Enter queries to search the indexed corpus.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Query: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            print("Goodbye!")
            break
        query_embedding = [0.88] + [0.12] * 1535
        response = retrieve(query, vector_store, top_k=3, query_embedding=query_embedding)
        print_retrieval_results(response)


def main():
    parser = argparse.ArgumentParser(description="Similarity search and top-k retrieval demo")
    parser.add_argument("--mode", choices=["basic", "k-compare", "context", "filtered", "interactive"], default="basic")
    args = parser.parse_args()

    vector_store = build_demo_store()
    print(f"\nConnected to demo vector store: {vector_store.count()} chunks indexed")

    if args.mode == "basic":
        demo_basic_retrieval(vector_store)
    elif args.mode == "k-compare":
        demo_k_comparison(vector_store)
    elif args.mode == "context":
        demo_context_building(vector_store)
    elif args.mode == "filtered":
        demo_filtered_search(vector_store)
    elif args.mode == "interactive":
        interactive_search(vector_store)

    print("\n✓ Demo complete")


if __name__ == "__main__":
    main()
