"""Test retrieval with sample data when corpus is empty."""

import sys

try:
    from embeddings import EmbeddedChunk
    from vector_store import VectorStore
    from retrieval import retrieve, compare_k_values, print_retrieval_results, print_k_comparison
except ImportError:
    print("Error: Could not import required modules.")
    sys.exit(1)


def create_test_corpus():
    """Create a test corpus for retrieval testing."""
    print("Creating test corpus with sample documents...\n")

    store = VectorStore(persist_dir="outputs/chroma_db_retrieval_test")

    # Sample documents split into chunks
    test_docs = [
        {
            "source": "account-guide.md",
            "chunks": [
                {
                    "text": "How to reset your password: Click on 'Forgot Password' on the login page.",
                    "chunk_index": 0,
                    "section": "Account Access",
                },
                {
                    "text": "Password reset links expire after 24 hours for security.",
                    "chunk_index": 1,
                    "section": "Security",
                },
                {
                    "text": "You can also reset your password from your account settings.",
                    "chunk_index": 2,
                    "section": "Account Access",
                },
            ],
        },
        {
            "source": "safety-guide.md",
            "chunks": [
                {
                    "text": "Safety procedures: Always follow emergency protocols.",
                    "chunk_index": 0,
                    "section": "Emergency",
                },
                {
                    "text": "In case of emergency, use the red emergency button.",
                    "chunk_index": 1,
                    "section": "Emergency",
                },
                {
                    "text": "Regular safety drills are conducted monthly.",
                    "chunk_index": 2,
                    "section": "Training",
                },
            ],
        },
        {
            "source": "maintenance.md",
            "chunks": [
                {
                    "text": "Maintenance schedule: Equipment is serviced every 30 days.",
                    "chunk_index": 0,
                    "section": "Schedule",
                },
                {
                    "text": "Report any equipment issues immediately to the maintenance team.",
                    "chunk_index": 1,
                    "section": "Procedures",
                },
                {
                    "text": "Keep maintenance logs updated for all work performed.",
                    "chunk_index": 2,
                    "section": "Documentation",
                },
            ],
        },
    ]

    # Create test embedded chunks
    embedded_chunks = []
    for doc in test_docs:
        source = doc["source"]
        for chunk_info in doc["chunks"]:
            # Create a simple deterministic embedding based on text
            text = chunk_info["text"]
            base_val = sum(ord(c) for c in text) / len(text) / 100
            embedding = [base_val + (i % 10) * 0.01 for i in range(1536)]

            embedded_chunk = EmbeddedChunk(
                text=text,
                metadata={
                    "source": source,
                    "chunk_index": chunk_info["chunk_index"],
                    "section": chunk_info["section"],
                },
                embedding=embedding,
            )
            embedded_chunks.append(embedded_chunk)

    # Index the test corpus
    print(f"Indexing {len(embedded_chunks)} test chunks...")
    result = store.upsert(embedded_chunks)
    print(f"✓ Indexed {result['inserted']} chunks\n")

    return store


def test_retrieval_with_dummy_embeddings():
    """Test retrieval using dummy embeddings (no API needed)."""
    print("=" * 80)
    print("RETRIEVAL TESTING WITH SAMPLE CORPUS (Dummy Embeddings)")
    print("=" * 80)

    # Create test corpus
    store = create_test_corpus()
    print(f"Total chunks in store: {store.count()}\n")

    # Manually test retrieval with dummy query embeddings
    print("=" * 80)
    print("TEST 1: Search with Dummy Query Embedding")
    print("=" * 80)

    # Create a dummy query embedding
    query_text = "password reset"
    query_embedding = [0.1 + (i % 10) * 0.01 for i in range(1536)]

    print(f"\nQuery: '{query_text}'")
    print(f"Query vector dimension: {len(query_embedding)}")
    print("-" * 80)

    # Search directly
    results = store.search(query_embedding, top_k=3)
    print(f"Found {len(results)} results:\n")

    for rank, result in enumerate(results, 1):
        print(f"[{rank}] Similarity: {result['similarity']:.4f}")
        print(f"    ID: {result['id']}")
        print(f"    Source: {result['metadata'].get('source', 'unknown')}")
        print(f"    Section: {result['metadata'].get('section', 'N/A')}")
        print(f"    Text: {result['text'][:70]}...")
        print()

    # Test k-comparison
    print("\n" + "=" * 80)
    print("TEST 2: Top-K Comparison (Different k values)")
    print("=" * 80)

    for k in [1, 3, 5]:
        results = store.search(query_embedding, top_k=k)
        print(f"\nk = {k} ({len(results)} results):")
        for rank, result in enumerate(results, 1):
            print(f"  [{rank}] {result['similarity']:.4f} | {result['text'][:60]}...")

    print("\n" + "=" * 80)
    print("✓ Retrieval tests complete")
    print("=" * 80)


if __name__ == "__main__":
    test_retrieval_with_dummy_embeddings()
    
    # Note: For full retrieval testing with real embeddings:
    # 1. Ensure OPENAI_API_KEY and OPENAI_BASE_URL are set in .env
    # 2. Run: python index_corpus.py --data-dir data
    # 3. Then use: python retrieval_demo.py
