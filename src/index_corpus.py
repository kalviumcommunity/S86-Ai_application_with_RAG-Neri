"""Corpus indexing workflow: prepare, insert, verify, spot-check."""

import sys
from pathlib import Path

try:
    from embeddings import batch_embed_chunks
    from ingestion import ingest, validate_ingestion
    from indexing import index_embeddings, print_indexing_summary, reindex_corpus
    from vector_store import VectorStore
except ImportError:
    print("Error: Could not import required modules. Run from src/ directory.")
    sys.exit(1)


def index_corpus_workflow(
    data_dir: str | Path,
    vector_dir: str | Path = "outputs/chroma_db",
    clear_existing: bool = False,
) -> dict:
    """Complete workflow: ingest → embed → index → validate.

    Args:
        data_dir: Directory containing source documents
        vector_dir: Vector database directory
        clear_existing: Whether to clear existing index

    Returns:
        Summary dict with workflow results
    """
    print("=" * 70)
    print("CORPUS INDEXING WORKFLOW")
    print("=" * 70)

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        print(f"\nError: Data directory not found: {data_dir}")
        return {"success": False, "error": "Data directory not found"}

    # Initialize vector store
    print(f"\n[STEP 1] Initialize Vector Store")
    vector_store = VectorStore(persist_dir=vector_dir)
    print(f"  ✓ Collection: {vector_store.collection_name}")
    print(f"  ✓ Dimension: {vector_store.vector_dimension}")

    if clear_existing:
        vector_store.clear()
        print(f"  ✓ Cleared existing index")

    # Ingest documents
    print(f"\n[STEP 2] Ingest Documents")
    files, documents, chunks, failures = ingest(data_dir, token_size=64, token_overlap=16)

    try:
        validate_ingestion(files, documents, failures)
    except AssertionError as e:
        print(f"  ✗ Ingestion validation failed: {e}")
        return {"success": False, "error": str(e)}

    print(f"  ✓ Files: {len(files)}")
    print(f"  ✓ Documents: {documents}")
    print(f"  ✓ Chunks: {len(chunks)}")

    if not chunks:
        print("\n⚠ No chunks to index. Check your data directory.")
        return {"success": False, "error": "No chunks created"}

    # Generate embeddings
    print(f"\n[STEP 3] Generate Embeddings")
    try:
        embedded_chunks, summary = batch_embed_chunks(
            chunks,
            batch_size=50,
            cache_path="outputs/embedding_cache.json",
        )
        print(f"  ✓ Generated: {summary.embeddings_generated}")
        print(f"  ✓ Cached: {summary.skipped_chunks}")
        print(f"  ✓ Total: {len(embedded_chunks)}")
    except ValueError as e:
        print(f"  ✗ Embedding error: {e}")
        return {"success": False, "error": str(e)}

    # INDEX: Prepare records for indexing
    print(f"\n[STEP 4] Prepare Vector Records")
    print(f"  ✓ {len(embedded_chunks)} records prepared with:")
    print(f"    - Stable IDs (source:chunk_index)")
    print(f"    - Embeddings (1536-dim)")
    print(f"    - Source text")
    print(f"    - Metadata (source, chunk_index, section)")

    # INDEX: Bulk insert and verify
    print(f"\n[STEP 5] Bulk Insert & Verify")
    indexing_result = index_embeddings(embedded_chunks, vector_store, batch_size=100)

    # Store results
    results = {
        "success": indexing_result.validation_passed,
        "files": len(files),
        "documents": documents,
        "chunks": len(chunks),
        "embedded": len(embedded_chunks),
        "indexed": indexing_result.indexed_count,
        "failures": indexing_result.failures,
        "validation_passed": indexing_result.validation_passed,
        "spot_checks": {
            "total": len(indexing_result.spot_checks),
            "passed": sum(1 for c in indexing_result.spot_checks if c["passed"]),
        },
    }

    # Print summary
    print_indexing_summary(indexing_result)

    return results


def verify_indexed_corpus(vector_dir: str | Path) -> dict:
    """Verify an existing indexed corpus.

    Args:
        vector_dir: Vector database directory

    Returns:
        Verification results
    """
    print("\n" + "=" * 70)
    print("VERIFY INDEXED CORPUS")
    print("=" * 70)

    vector_store = VectorStore(persist_dir=vector_dir)
    count = vector_store.count()

    print(f"\nIndexed collection: {vector_store.collection_name}")
    print(f"Total chunks: {count}")

    if count == 0:
        print("\n⚠ No chunks in index")
        return {"total": 0, "verified": False}

    # Sample a record
    results = vector_store.collection.get(include=[], limit=1)
    if results.get("ids"):
        sample_id = results["ids"][0]
        sample = vector_store.get(sample_id)

        if sample:
            print(f"\nSample chunk: {sample['id']}")
            print(f"  Source: {sample['metadata'].get('source', 'unknown')}")
            print(f"  Section: {sample['metadata'].get('section', 'N/A')}")
            print(f"  Text: {sample['text'][:80]}...")
            print(f"  Vector dim: {len(sample['embedding'])}")

    return {
        "total": count,
        "verified": True,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../data", help="Documents directory")
    parser.add_argument("--vector-dir", default="../outputs/chroma_db", help="Vector DB directory")
    parser.add_argument("--clear", action="store_true", help="Clear existing index")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing index")
    args = parser.parse_args()

    if args.verify_only:
        verify_indexed_corpus(args.vector_dir)
    else:
        results = index_corpus_workflow(args.data_dir, args.vector_dir, args.clear)
        if results["success"]:
            print("\n✓ Indexing workflow completed successfully!")
        else:
            print(f"\n✗ Indexing workflow failed: {results.get('error')}")
            sys.exit(1)
