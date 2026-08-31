"""Complete RAG demonstration: ingest, embed, store, and retrieve."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from embeddings import embed, batch_embed_chunks
    from ingestion import ingest, validate_ingestion
    from vector_store import VectorStore
except ImportError:
    print("Error: Could not import required modules. Run from src/ directory.")
    sys.exit(1)


def demo_complete_pipeline():
    """Demonstrate the complete RAG pipeline end-to-end."""
    print("=" * 70)
    print("RAG PIPELINE DEMONSTRATION")
    print("=" * 70)

    load_dotenv()

    # Configuration
    data_dir = Path("../data")
    vector_dir = Path("../outputs/chroma_db_demo")
    
    if not data_dir.exists():
        print(f"\nError: Data directory not found at {data_dir}")
        print("Please ensure you have documents in the data/ folder")
        return

    print(f"\n[CONFIG]")
    print(f"  Data directory: {data_dir.resolve()}")
    print(f"  Vector store: {vector_dir.resolve()}")

    # Initialize vector store
    print(f"\n[STEP 1] Initialize Vector Store")
    store = VectorStore(persist_dir=vector_dir)
    print(f"  ✓ Collection: {store.collection_name}")
    print(f"  ✓ Dimension: {store.vector_dimension}")
    print(f"  ✓ Metric: cosine")

    # Ingest documents
    print(f"\n[STEP 2] Ingest Documents")
    files, documents, chunks, failures = ingest(data_dir, token_size=64, token_overlap=16)
    validate_ingestion(files, documents, failures)

    print(f"  ✓ Files discovered: {len(files)}")
    print(f"  ✓ Documents processed: {documents}")
    print(f"  ✓ Chunks created: {len(chunks)}")
    if failures:
        print(f"  ⚠ Failures: {len(failures)}")

    if len(chunks) == 0:
        print("\n⚠ No chunks to embed. Check your data directory.")
        return

    # Generate embeddings
    print(f"\n[STEP 3] Generate Embeddings")
    try:
        embedded_chunks, summary = batch_embed_chunks(
            chunks,
            batch_size=50,
            cache_path="../outputs/embedding_cache_demo.json",
        )
        print(f"  ✓ Generated: {summary.embeddings_generated}")
        print(f"  ✓ Cached (reused): {summary.skipped_chunks}")
        print(f"  ✓ Total embeddings: {len(embedded_chunks)}")
        print(f"  ✓ Cost: ${summary.estimated_cost:.4f}")
    except ValueError as e:
        print(f"  ✗ API Error: {e}")
        print("  Ensure OPENAI_API_KEY and OPENAI_BASE_URL are set in .env")
        return

    # Store in vector database
    print(f"\n[STEP 4] Store Embeddings")
    result = store.upsert(embedded_chunks)
    print(f"  ✓ Upserted: {result['inserted']}")
    total = store.count()
    print(f"  ✓ Total in collection: {total}")

    # Demonstrate retrieval
    print(f"\n[STEP 5] Semantic Search Demo")
    
    # Sample queries
    sample_queries = [
        "password reset instructions",
        "maintenance procedures",
        "safety guidelines",
    ]

    for query_text in sample_queries:
        print(f"\n  Query: \"{query_text}\"")
        try:
            query_embedding = embed([query_text])[0]
            results = store.search(query_embedding, top_k=3)

            if not results:
                print(f"    No results found")
                continue

            for rank, result in enumerate(results, 1):
                similarity = result['similarity']
                text_preview = result['text'][:60].replace('\n', ' ')
                print(f"    [{rank}] {similarity:.4f} | {text_preview}...")

        except ValueError as e:
            print(f"    Error: {e}")

    print(f"\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print(f"\nVector store persisted in: {vector_dir.resolve()}")
    print("You can now use this store for retrieval-augmented generation!")


def demo_direct_insert():
    """Demonstrate direct insertion without full pipeline."""
    print("\n" + "=" * 70)
    print("DIRECT INSERT DEMO (Testing only)")
    print("=" * 70)

    from embeddings import EmbeddedChunk

    store = VectorStore(persist_dir="outputs/chroma_db_test")

    # Create test data
    test_chunks = [
        EmbeddedChunk(
            text="How do I reset my account password?",
            metadata={
                "source": "faq.md",
                "chunk_index": 0,
                "section": "Account Management",
            },
            embedding=[0.1] * 1536,
        ),
        EmbeddedChunk(
            text="The password reset link expires after 24 hours.",
            metadata={
                "source": "faq.md",
                "chunk_index": 1,
                "section": "Account Management",
            },
            embedding=[0.2] * 1536,
        ),
    ]

    print(f"\nInserting {len(test_chunks)} test chunks...")
    result = store.upsert(test_chunks)
    print(f"  ✓ Inserted: {result['inserted']}")

    # Retrieve
    print(f"\nRetrieving chunk 'faq.md:0'...")
    chunk = store.get("faq.md:0")
    if chunk:
        print(f"  ✓ ID: {chunk['id']}")
        print(f"  ✓ Text: {chunk['text']}")
        print(f"  ✓ Metadata: {chunk['metadata']}")
        print(f"  ✓ Vector length: {len(chunk['embedding'])}")

    print(f"\nTotal in collection: {store.count()}")


if __name__ == "__main__":
    try:
        # Full pipeline demo (requires documents and API access)
        demo_complete_pipeline()

        # Also show direct insertion
        # Uncomment to test:
        # demo_direct_insert()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
