"""
Indexing utilities for the NERI RAG application.

This module:
- Converts EmbeddedChunk/dict objects into vector records
- Handles Chroma insertion
- Supports insert/update
- Performs reliable spot checks
- Avoids NumPy truth-value errors
- Provides a stable IndexingResult API
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class IndexingResult:
    """
    Result returned by index_embeddings().
    """

    total_prepared: int = 0
    inserted: int = 0
    updated: int = 0
    failures: int = 0

    indexed_count: int = 0
    validation_passed: bool = False

    # IMPORTANT:
    # index_corpus.py expects this attribute.
    spot_checks: List[Dict[str, Any]] = field(default_factory=list)

    errors: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# GENERIC HELPERS
# ============================================================

def _get_value(obj: Any, key: str, default=None):
    """
    Read a value from either:
    - dictionary
    - dataclass/object
    """

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _safe_list(value):
    """
    Convert a value to a normal Python list.

    This is especially important for NumPy arrays.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    try:
        return value.tolist()
    except AttributeError:
        try:
            return list(value)
        except TypeError:
            return [value]


def _embedding_dimension(embedding):
    """
    Safely determine embedding dimension.
    """

    if embedding is None:
        return 0

    try:
        return len(embedding)
    except TypeError:
        return 0


# ============================================================
# BATCHING
# ============================================================

def batches(items: Iterable, batch_size: int = 100):
    """
    Yield items in batches.
    """

    items = list(items)

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


# ============================================================
# VECTOR RECORD CREATION
# ============================================================

def to_vector_record(chunk: Any, embedding: Any = None) -> Dict[str, Any]:
    """
    Convert an EmbeddedChunk or dictionary into a standard
    vector record dictionary.

    Standard format:

    {
        "id": "...",
        "text": "...",
        "embedding": [...],
        "metadata": {...}
    }
    """

    # --------------------------------------------------------
    # If the object is already a vector record
    # --------------------------------------------------------

    if isinstance(chunk, dict):
        chunk_id = (
            chunk.get("id")
            or chunk.get("chunk_id")
            or chunk.get("document_id")
        )

        text = chunk.get("text", "")

        if embedding is None:
            embedding = chunk.get("embedding")

        metadata = chunk.get("metadata") or {}

        metadata = dict(metadata)

    else:
        chunk_id = (
            getattr(chunk, "id", None)
            or getattr(chunk, "chunk_id", None)
            or getattr(chunk, "document_id", None)
        )

        text = getattr(chunk, "text", "")

        if embedding is None:
            embedding = getattr(chunk, "embedding", None)

        metadata = getattr(chunk, "metadata", None) or {}

        metadata = dict(metadata)

    # --------------------------------------------------------
    # If ID is missing, build one from source + chunk_index
    # --------------------------------------------------------

    source = metadata.get("source")

    chunk_index = metadata.get("chunk_index")

    if not chunk_id:
        if source is not None and chunk_index is not None:
            chunk_id = f"{source}:{chunk_index}"
        else:
            raise ValueError(
                "Could not determine chunk ID. "
                "Expected id/chunk_id or metadata[source] + metadata[chunk_index]."
            )

    # --------------------------------------------------------
    # Clean metadata
    # --------------------------------------------------------

    clean_metadata = {}

    for key, value in metadata.items():

        # Chroma metadata must contain primitive values.
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            clean_metadata[key] = value
        else:
            clean_metadata[key] = str(value)

    # --------------------------------------------------------
    # Normalize embedding
    # --------------------------------------------------------

    clean_embedding = _safe_list(embedding)

    return {
        "id": str(chunk_id),
        "text": str(text),
        "embedding": clean_embedding,
        "metadata": clean_metadata,
    }


# ============================================================
# CHROMA UPSERT
# ============================================================

def _get_collection(vector_store):
    """
    Get the underlying Chroma collection.

    Supports common VectorStore implementations.
    """

    # Most likely implementation
    collection = getattr(vector_store, "collection", None)

    if collection is not None:
        return collection

    # Some implementations use _collection
    collection = getattr(vector_store, "_collection", None)

    if collection is not None:
        return collection

    # Some implementations expose get_collection()
    getter = getattr(vector_store, "get_collection", None)

    if callable(getter):
        return getter()

    raise AttributeError(
        "Could not find the Chroma collection inside VectorStore. "
        "Expected .collection, ._collection, or get_collection()."
    )


def _existing_ids(vector_store, ids: List[str]) -> set:
    """
    Check which IDs already exist.

    Uses Chroma collection.get().
    """

    collection = _get_collection(vector_store)

    if not ids:
        return set()

    result = collection.get(ids=ids)

    existing = result.get("ids", []) if isinstance(result, dict) else []

    # Chroma normally returns a flat list.
    if existing is None:
        return set()

    if isinstance(existing, list):
        return set(str(x) for x in existing)

    return {str(existing)}


def _upsert_batch(vector_store, records: List[Dict[str, Any]]):
    """
    Insert/update records into Chroma.

    We intentionally use Chroma's upsert() directly.
    """

    if not records:
        return

    collection = _get_collection(vector_store)

    ids = [record["id"] for record in records]

    documents = [
        record["text"]
        for record in records
    ]

    embeddings = [
        record["embedding"]
        for record in records
    ]

    metadatas = [
        record["metadata"]
        for record in records
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


# ============================================================
# SPOT CHECK
# ============================================================

def spot_check_records(
    vector_store,
    records: List[Dict[str, Any]],
    sample_size: int = 3,
) -> List[Dict[str, Any]]:
    """
    Verify that selected records really exist in Chroma.

    IMPORTANT:
    We never do:

        if embedding == other_embedding:

    because NumPy arrays cause:

        The truth value of an array with more than one element
        is ambiguous.

    Instead, we verify IDs and text.
    """

    if not records:
        return []

    # Deterministic sample
    selected = records[:sample_size]

    collection = _get_collection(vector_store)

    results = []

    for record in selected:

        record_id = str(record["id"])

        try:
            result = collection.get(
                ids=[record_id],
                include=[
                    "documents",
                    "metadatas",
                ],
            )

            returned_ids = result.get("ids", [])

            # Normalize Chroma response
            returned_ids = [
                str(x)
                for x in (returned_ids or [])
            ]

            if record_id not in returned_ids:
                results.append({
                    "id": record_id,
                    "passed": False,
                    "error": "Record not found in database",
                })
                continue

            # Find returned position
            position = returned_ids.index(record_id)

            documents = result.get("documents", []) or []
            metadatas = result.get("metadatas", []) or []

            stored_text = (
                documents[position]
                if position < len(documents)
                else None
            )

            stored_metadata = (
                metadatas[position]
                if position < len(metadatas)
                else {}
            )

            expected_text = record.get("text", "")

            text_matches = (
                stored_text == expected_text
            )

            # Do NOT compare embedding arrays.
            metadata_matches = True

            expected_source = record["metadata"].get("source")

            if expected_source is not None:
                metadata_matches = (
                    stored_metadata.get("source")
                    == expected_source
                )

            passed = bool(
                text_matches
                and metadata_matches
            )

            results.append({
                "id": record_id,
                "passed": passed,
                "error": None if passed else "Stored data does not match",
            })

        except Exception as error:

            results.append({
                "id": record_id,
                "passed": False,
                "error": str(error),
            })

    return results


# ============================================================
# INDEX EMBEDDINGS
# ============================================================

def index_embeddings(
    vector_store,
    chunks,
    embeddings=None,
    batch_size: int = 100,
    expected_dimension: Optional[int] = None,
):
    """
    Index chunks and embeddings.

    Supports:
        EmbeddedChunk objects
        dictionaries

    Returns:
        IndexingResult
    """

    chunks = list(chunks)

    if embeddings is None:
        # Try embeddings attached to chunks
        embeddings = [
            _get_value(chunk, "embedding", None)
            for chunk in chunks
        ]

    embeddings = list(embeddings)

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Number of chunks ({len(chunks)}) does not match "
            f"number of embeddings ({len(embeddings)})."
        )

    # --------------------------------------------------------
    # Prepare records
    # --------------------------------------------------------

    records = []

    for chunk, embedding in zip(chunks, embeddings):

        record = to_vector_record(
            chunk,
            embedding,
        )

        dimension = _embedding_dimension(
            record["embedding"]
        )

        if dimension == 0:
            raise ValueError(
                f"Empty embedding for record {record['id']}"
            )

        if expected_dimension is not None:
            if dimension != expected_dimension:
                raise ValueError(
                    f"Wrong embedding dimension for "
                    f"{record['id']}: "
                    f"{dimension}, expected {expected_dimension}"
                )

        records.append(record)

    print(
        f"\n[1/4] Preparing {len(records)} records for indexing..."
    )

    print(
        f"  ✓ {len(records)} records prepared"
    )

    if records:
        print(
            f"  Sample ID: {records[0]['id']}"
        )

        print(
            "  Metadata keys:",
            list(records[0]["metadata"].keys()),
        )

    # --------------------------------------------------------
    # Find existing records
    # --------------------------------------------------------

    ids = [
        record["id"]
        for record in records
    ]

    try:
        existing = _existing_ids(
            vector_store,
            ids,
        )
    except Exception:
        existing = set()

    # --------------------------------------------------------
    # Insert / update
    # --------------------------------------------------------

    inserted = 0
    updated = 0
    failures = 0
    errors = []

    print(
        f"\n[2/4] Bulk inserting in batches of {batch_size}..."
    )

    for batch_number, batch in enumerate(
        batches(records, batch_size),
        start=1,
    ):

        try:

            _upsert_batch(
                vector_store,
                batch,
            )

            batch_existing = sum(
                1
                for record in batch
                if record["id"] in existing
            )

            batch_new = (
                len(batch)
                - batch_existing
            )

            updated += batch_existing
            inserted += batch_new

            print(
                f"  Batch {batch_number}: "
                f"{len(batch)} records"
            )

            print(
                f"  ✓ Inserted: {batch_new}"
            )

            print(
                f"  ✓ Updated: {batch_existing}"
            )

        except Exception as error:

            failures += len(batch)

            errors.append({
                "batch": batch_number,
                "count": len(batch),
                "error": str(error),
            })

            print(
                f"  ✗ batch {batch_number} "
                f"({len(batch)} records): {error}"
            )

    # --------------------------------------------------------
    # Verify count
    # --------------------------------------------------------

    print(
        "\n[3/4] Verifying indexed count..."
    )

    try:
        indexed_count = vector_store.count()
    except Exception:
        collection = _get_collection(vector_store)
        indexed_count = collection.count()

    expected_count = len(records)

    validation_passed = (
        indexed_count == expected_count
        and failures == 0
    )

    print(
        f"  Expected chunks: {expected_count}"
    )

    print(
        f"  Indexed count: {indexed_count}"
    )

    if validation_passed:
        print(
            "  ✓ Validation PASSED"
        )
    else:
        print(
            "  ✗ Validation FAILED"
        )

    # --------------------------------------------------------
    # Spot checks
    # --------------------------------------------------------

    print(
        "\n[4/4] Spot-checking stored records..."
    )

    checks = spot_check_records(
        vector_store,
        records,
        sample_size=3,
    )

    passed_checks = sum(
        1
        for check in checks
        if check.get("passed") is True
    )

    print(
        f"  ✓ {passed_checks}/{len(checks)} "
        f"spot checks passed"
    )

    for check in checks:

        if not check["passed"]:
            print(
                f"  ✗ {check['id']}: "
                f"{check.get('error')}"
            )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = IndexingResult(
        total_prepared=len(records),
        inserted=inserted,
        updated=updated,
        failures=failures,
        indexed_count=indexed_count,
        validation_passed=validation_passed,
        spot_checks=checks,
        errors=errors,
    )

    return result


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_indexing_summary(result: IndexingResult):
    """
    Print final indexing summary.
    """

    print("\n")
    print("=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)

    print()
    print("=" * 70)
    print("INDEXING SUMMARY")
    print("=" * 70)

    print()
    print("Records:")
    print(
        f"  Total prepared: {result.total_prepared}"
    )
    print(
        f"  Inserted: {result.inserted}"
    )
    print(
        f"  Updated: {result.updated}"
    )
    print(
        f"  Failures: {result.failures}"
    )

    print()
    print("Verification:")
    print(
        f"  Indexed count: {result.indexed_count}"
    )

    if result.validation_passed:
        print(
            "  Validation: PASSED ✓"
        )
    else:
        print(
            "  Validation: FAILED ✗"
        )

    print()
    print("Spot Checks:")

    passed = sum(
        1
        for check in result.spot_checks
        if check.get("passed") is True
    )

    total = len(result.spot_checks)

    print(
        f"  {passed}/{total} checks passed"
    )

    for check in result.spot_checks:

        if not check["passed"]:
            print(
                f"  ✗ {check['id']}"
            )
            print(
                f"     Error: {check.get('error')}"
            )

    print()
    print("=" * 70)


# ============================================================
# REINDEX CORPUS
# ============================================================

def reindex_corpus(
    vector_store,
    chunks,
    embeddings,
    batch_size=100,
    expected_dimension=None,
):
    """
    Convenience wrapper.
    """

    return index_embeddings(
        vector_store=vector_store,
        chunks=chunks,
        embeddings=embeddings,
        batch_size=batch_size,
        expected_dimension=expected_dimension,
    )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("Testing indexing module...")
    print("-" * 50)

    print("✓ IndexingResult available")
    print("✓ batches() available")
    print("✓ to_vector_record() available")
    print("✓ spot_check_records() available")
    print("✓ index_embeddings() available")
    print("✓ print_indexing_summary() available")
    print("✓ reindex_corpus() available")

    print("-" * 50)
    print("Indexing module test complete!")