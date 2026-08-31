"""Interactive retrieval demonstration and testing."""

import sys
import argparse
from pathlib import Path

try:
    from retrieval import (
        retrieve,
        retrieve_with_context,
        compare_k_values,
        print_retrieval_results,
        print_k_comparison,
        analyze_retrieval_quality,
    )
    from vector_store import VectorStore
except ImportError:
    print("Error: Could not import required modules. Run from src/ directory.")
    sys.exit(1)


def demo_basic_retrieval(vector_store: VectorStore):
    """Demonstrate basic retrieval with a single query."""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Retrieval")
    print("=" * 80)

    query = "How do I reset my password?"
    print(f"\nQuery: {query}")

    response = retrieve(query, vector_store, top_k=3)
    print_retrieval_results(response)

    # Quality metrics
    metrics = analyze_retrieval_quality(response)
    print("Quality Metrics:")
    print(f"  Average similarity: {metrics['avg_similarity']:.4f}")
    print(f"  Top result quality: {metrics['top_result_quality']}")
    print(f"  Similarity spread: {metrics['similarity_spread']:.4f}")


def demo_k_comparison(vector_store: VectorStore):
    """Demonstrate how changing k affects results."""
    print("\n" + "=" * 80)
    print("DEMO 2: Top-K Comparison (How k affects results)")
    print("=" * 80)

    query = "maintenance safety procedures"
    k_values = [1, 3, 5]

    comparison = compare_k_values(query, k_values, vector_store)
    print_k_comparison(query, comparison)

    # Analysis
    print("Analysis:")
    for k in sorted(comparison.keys()):
        response = comparison[k]
        metrics = analyze_retrieval_quality(response)
        print(f"  k={k}: {response.total_retrieved} results | "
              f"Avg sim: {metrics['avg_similarity']:.4f} | "
              f"Top: {metrics['top_result_quality']}")


def demo_context_building(vector_store: VectorStore):
    """Demonstrate building LLM context from retrieval."""
    print("\n" + "=" * 80)
    print("DEMO 3: Building LLM Context from Retrieval")
    print("=" * 80)

    query = "What are the procedures?"
    print(f"\nQuery: {query}\n")

    context = retrieve_with_context(query, vector_store, top_k=3)
    print("Formatted Context (ready for LLM):")
    print("-" * 80)
    print(context)
    print("-" * 80)

    # Show how it would be used in a prompt
    llm_prompt = f"""Answer the following question based on the provided context.
If the context does not contain the answer, say so.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

    print("\nComplete LLM Prompt:")
    print("-" * 80)
    print(llm_prompt)
    print("-" * 80)


def interactive_search(vector_store: VectorStore):
    """Interactive query mode."""
    print("\n" + "=" * 80)
    print("INTERACTIVE SEARCH")
    print("=" * 80)
    print("Enter queries to search the indexed corpus.")
    print("Type 'quit' to exit, 'help' for options.\n")

    while True:
        try:
            query = input("Query: ").strip()

            if not query:
                continue

            if query.lower() == "quit":
                print("Goodbye!")
                break

            if query.lower() == "help":
                print("""
Commands:
  quit                        Exit interactive mode
  k=<number> <query>         Set top-k (e.g., k=5 password reset)
  compare <k1>,<k2>,<k3>    Compare k values (e.g., compare 1,3,5)
  help                       Show this help
  <query>                    Search with default k=3
                """)
                continue

            # Parse k parameter
            top_k = 3
            if query.lower().startswith("k="):
                parts = query.split(" ", 1)
                try:
                    top_k = int(parts[0][2:])
                    query = parts[1] if len(parts) > 1 else ""
                    if not query:
                        print("Please enter a query")
                        continue
                except ValueError:
                    print("Invalid k value")
                    continue

            # Compare k values
            if query.lower().startswith("compare "):
                try:
                    k_str = query.split(" ", 1)[1]
                    k_values = [int(k.strip()) for k in k_str.split(",")]
                    
                    query_text = input("Query for comparison: ").strip()
                    if query_text:
                        comparison = compare_k_values(query_text, k_values, vector_store)
                        print_k_comparison(query_text, comparison)
                except (ValueError, IndexError):
                    print("Usage: compare 1,3,5")
                continue

            # Regular search
            response = retrieve(query, vector_store, top_k=top_k)
            print_retrieval_results(response)

            # Show metrics
            metrics = analyze_retrieval_quality(response)
            print(f"Metrics: {response.total_retrieved} results | "
                  f"Avg: {metrics['avg_similarity']:.4f} | "
                  f"Quality: {metrics['top_result_quality']}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_benchmarks(vector_store: VectorStore):
    """Run performance benchmarks."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 80)

    import time

    test_queries = [
        "password reset",
        "safety procedures",
        "maintenance schedule",
        "troubleshooting",
        "account access",
    ]

    k_values = [1, 3, 5, 10]

    print("\nTesting retrieval latency across different k values:\n")
    print(f"{'Query':<30} {'K':<5} {'Latency (ms)':<15} {'Results':<10}")
    print("─" * 70)

    for query in test_queries:
        for k in k_values:
            try:
                start = time.time()
                response = retrieve(query, vector_store, top_k=k)
                elapsed_ms = (time.time() - start) * 1000

                print(f"{query:<30} {k:<5} {elapsed_ms:<15.2f} {response.total_retrieved:<10}")
            except Exception as e:
                print(f"{query:<30} {k:<5} ERROR: {str(e)[:20]}")

    print("\nNote: First query may be slower due to initialization.")


def main():
    parser = argparse.ArgumentParser(description="Similarity search and top-k retrieval demo")
    parser.add_argument("--mode", choices=["basic", "k-compare", "context", "interactive", "benchmark"],
                        default="basic", help="Demo mode to run")
    parser.add_argument("--vector-dir", default="outputs/chroma_db", help="Vector store directory")
    parser.add_argument("--query", help="Query string (for non-interactive mode)")
    parser.add_argument("--k", type=int, default=3, help="Top-k value")
    args = parser.parse_args()

    # Initialize vector store
    vector_store = VectorStore(persist_dir=args.vector_dir)

    if vector_store.count() == 0:
        print("\n⚠ Vector store is empty!")
        print("Run indexing first: python index_corpus.py --data-dir data")
        sys.exit(1)

    print(f"\nConnected to vector store: {vector_store.count()} chunks indexed")

    # Run selected demo
    if args.mode == "basic":
        demo_basic_retrieval(vector_store)

    elif args.mode == "k-compare":
        demo_k_comparison(vector_store)

    elif args.mode == "context":
        demo_context_building(vector_store)

    elif args.mode == "interactive":
        interactive_search(vector_store)

    elif args.mode == "benchmark":
        run_benchmarks(vector_store)

    print("\n✓ Demo complete")


if __name__ == "__main__":
    main()
