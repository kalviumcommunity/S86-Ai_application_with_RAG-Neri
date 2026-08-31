"""Indexing embeddings and metadata into vector database."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

try:
    from .embeddings import EmbeddedChunk
    from .vector_store import VectorStore
except ImportError:
    from embeddings import EmbeddedChunk
    from vector_store import VectorStore


@dataclass
class IndexingResult:
    """Summary from an indexing operation."""

    total_records: int
    inserted: int
    updated: int
    failures: int
    error_details: list[dict]
    indexed_count: int
    validation_passed: bool
    spot_checks: list[dict]


def batches(items: list, batch_size: int):
    """Yield successive batches from a list."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def to_vector_record(chunk: EmbeddedChunk) -> dict:
    """Convert an EmbeddedChunk to a vector record for storage.

    Args:
        chunk: EmbeddedChunk with text, metadata, and embedding

    Returns:
        Dict with id, vector, text, metadata for vector database
    """
    # Generate stable ID from source + chunk index
    source = chunk.metadata.get("source", "unknown")
    chunk_idx = chunk.metadata.get("chunk_index", 0)
    chunk_id = f"{source}:{chunk_idx}"

    return {
        "id": chunk_id,
        "vector": chunk.embedding,
        "text": chunk.text,
        "metadata": {
            "source": chunk.metadata.get("source", ""),
            "chunk_index": chunk.metadata.get("chunk_index", 0),
            "section": chunk.metadata.get("section", ""),
        },
    }


def index_embeddings(
    embedded_chunks: list[EmbeddedChunk],
    vector_store: VectorStore,
    batch_size: int = 100,
) -> IndexingResult:
    """Index all embedded chunks into vector database with validation.

    Args:
        embedded_chunks: List of EmbeddedChunk objects
        vector_store: Initialized VectorStore
        batch_size: Size of batches for bulk insert

    Returns:
        IndexingResult with counts, failures, and validation status
    """
    print("=" * 70)
    print("INDEXING EMBEDDINGS & METADATA STORAGE")
    print("=" * 70)

    if not embedded_chunks:
        return IndexingResult(
            total_records=0,
            inserted=0,
            updated=0,
            failures=0,
            error_details=[],
            indexed_count=0,
            validation_passed=False,
            spot_checks=[],
        )

    # Convert chunks to records
    print(f"\n[1/4] Preparing {len(embedded_chunks)} records for indexing...")
    records = [to_vector_record(chunk) for chunk in embedded_chunks]
    print(f"  ✓ {len(records)} records prepared")
    print(f"  Sample ID: {records[0]['id']}")
    print(f"  Metadata keys: {list(records[0]['metadata'].keys())}")

    # Bulk insert in batches
    print(f"\n[2/4] Bulk inserting in batches of {batch_size}...")
    inserted = 0
    failures = 0
    error_details = []

    for batch_num, batch in enumerate(batches(records, batch_size), 1):
        try:
            # Convert to EmbeddedChunk format for upsert
            batch_chunks = [
                EmbeddedChunk(
                    text=record["text"],
                    metadata=record["metadata"],
                    embedding=record["vector"],
                )
                for record in batch
            ]
            result = vector_store.upsert(batch_chunks)
            inserted += result["inserted"]
            print(f"  Batch {batch_num}: {result['inserted']} records")

        except Exception as error:
            failures += len(batch)
            error_details.append(
                {
                    "batch": batch_num,
                    "size": len(batch),
                    "error": str(error),
                    "first_id": batch[0]["id"],
                }
            )
            print(f"  Batch {batch_num}: FAILED - {str(error)[:60]}")

    print(f"  ✓ Inserted this run: {inserted}")
    if error_details:
        print(f"  ⚠ Failures: {failures}")

    # Verify counts
    print(f"\n[3/4] Verifying indexed count...")
    indexed_count = vector_store.count()
    expected_count = len(embedded_chunks)

    print(f"  Expected chunks: {expected_count}")
    print(f"  Indexed count: {indexed_count}")
    print(f"  Inserted this run: {inserted}")

    validation_passed = indexed_count >= expected_count
    if validation_passed:
        print(f"  ✓ Validation PASSED")
    else:
        print(f"  ✗ Validation FAILED: indexed count does not match")

    # Spot-check integrity
    print(f"\n[4/4] Spot-checking stored records...")
    spot_checks = spot_check_records(embedded_chunks, vector_store)

    passed = sum(1 for check in spot_checks if check["passed"])
    print(f"  ✓ {passed}/{len(spot_checks)} spot checks passed")

    if not all(check["passed"] for check in spot_checks):
        print(f"  ✗ Some spot checks failed")
        for check in spot_checks:
            if not check["passed"]:
                print(f"    - {check['id']}: {check['error']}")

    print("\n" + "=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)

    return IndexingResult(
        total_records=len(records),
        inserted=inserted,
        updated=0,
        failures=failures,
        error_details=error_details,
        indexed_count=indexed_count,
        validation_passed=validation_passed,
        spot_checks=spot_checks,
    )


def spot_check_records(
    embedded_chunks: list[EmbeddedChunk],
    vector_store: VectorStore,
    sample_size: int = 3,
) -> list[dict]:
    """Verify stored records match original chunks.

    Args:
        embedded_chunks: Original EmbeddedChunk list
        vector_store: VectorStore to check
        sample_size: Number of chunks to spot-check

    Returns:
        List of check results with pass/fail status
    """
    if not embedded_chunks:
        return []

    # Sample uniformly across the corpus
    step = max(1, len(embedded_chunks) // sample_size)
    samples = embedded_chunks[::step][:sample_size]

    checks = []
    for sample in samples:
        record = to_vector_record(sample)
        chunk_id = record["id"]

        try:
            stored = vector_store.get(chunk_id)

            if stored is None:
                checks.append(
                    {
                        "id": chunk_id,
                        "passed": False,
                        "error": "Record not found in database",
                    }
                )
                continue

            # Verify text
            text_match = stored["text"] == record["text"]

            # Verify metadata
            metadata_match = all(
                stored["metadata"].get(key) == value
                for key, value in record["metadata"].items()
                if value  # Only check non-empty metadata
            )

            # Verify vector dimension
            vector_dim_match = len(stored["embedding"]) == len(record["vector"])

            # All checks must pass
            passed = text_match and metadata_match and vector_dim_match

            checks.append(
                {
                    "id": chunk_id,
                    "passed": passed,
                    "text_match": text_match,
                    "metadata_match": metadata_match,
                    "vector_dim_match": vector_dim_match,
                    "source": stored["metadata"].get("source", ""),
                    "text_preview": stored["text"][:80],
                    "error": None if passed else "One or more fields don't match",
                }
            )

        except Exception as error:
            checks.append(
                {
                    "id": chunk_id,
                    "passed": False,
                    "error": str(error),
                }
            )

    return checks


def reindex_corpus(
    embedded_chunks: list[EmbeddedChunk],
    vector_store: VectorStore,
    batch_size: int = 100,
) -> dict:
    """Re-index with document version tracking for incremental updates.

    Args:
        embedded_chunks: Updated corpus
        vector_store: VectorStore with previous version
        batch_size: Batch size for insertion

    Returns:
        Summary of changes
    """
    print("\n" + "=" * 70)
    print("RE-INDEXING CORPUS (Incremental Update)")
    print("=" * 70)

    # For a full reindex, we'll clear and rebuild
    # In production, you'd track document versions and do selective updates
    print("\nRe-indexing strategy: Full rebuild")
    print("  (In production, implement version tracking for incremental updates)")

    vector_store.clear()
    print("  ✓ Cleared existing index")

    result = index_embeddings(embedded_chunks, vector_store, batch_size)

    return {
        "total": result.total_records,
        "inserted": result.inserted,
        "validation_passed": result.validation_passed,
        "spot_checks_passed": sum(1 for c in result.spot_checks if c["passed"]),
    }


def print_indexing_summary(result: IndexingResult) -> None:
    """Print formatted indexing results."""
    print("\n" + "=" * 70)
    print("INDEXING SUMMARY")
    print("=" * 70)
    print(f"\nRecords:")
    print(f"  Total prepared: {result.total_records}")
    print(f"  Inserted: {result.inserted}")
    print(f"  Failures: {result.failures}")

    print(f"\nVerification:")
    print(f"  Indexed count: {result.indexed_count}")
    print(f"  Validation: {'PASSED ✓' if result.validation_passed else 'FAILED ✗'}")

    print(f"\nSpot Checks:")
    if result.spot_checks:
        passed = sum(1 for c in result.spot_checks if c["passed"])
        print(f"  {passed}/{len(result.spot_checks)} checks passed")

        for check in result.spot_checks:
            status = "✓" if check["passed"] else "✗"
            print(f"  {status} {check['id']}")
            if not check["passed"] and check.get("error"):
                print(f"     Error: {check['error']}")

    if result.error_details:
        print(f"\nFailure Details:")
        for error in result.error_details:
            print(
                f"  Batch {error['batch']}: {error['first_id']} - {error['error'][:60]}"
            )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    from embeddings import EmbeddedChunk

    store = VectorStore(persist_dir="outputs/chroma_db_index_test")

    # Create sample data
    sample_chunks = [
        EmbeddedChunk(
            text="Account password reset instructions here.",
            metadata={
                "source": "account-guide.md",
                "chunk_index": i,
                "section": "Account Access",
            },
            embedding=[0.1 * (i + 1)] * 1536,
        )
        for i in range(5)
    ]

    # Index them
    result = index_embeddings(sample_chunks, store)
    print_indexing_summary(result)
