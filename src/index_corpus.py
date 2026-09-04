"""
NERI RAG Corpus Indexing Workflow

Usage:

python src/index_corpus.py --data-dir data --vector-dir outputs/chroma_db --clear
"""

import argparse
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# IMPORTS
# ============================================================

from src.ingestion import ingest
from src.embeddings import embed
from src.vector_store import VectorStore

from src.indexing import (
    index_embeddings,
    print_indexing_summary,
)


# ============================================================
# CONFIG
# ============================================================

DEFAULT_TOKEN_SIZE = 64
DEFAULT_TOKEN_OVERLAP = 16
DEFAULT_BATCH_SIZE = 100
EXPECTED_EMBEDDING_DIMENSION = 3072


# ============================================================
# CLEAR VECTOR STORE
# ============================================================

def recreate_collection(vector_store):
    """
    Recreate the Chroma collection.

    We use the underlying Chroma collection directly when
    available.
    """

    collection = getattr(
        vector_store,
        "collection",
        None,
    )

    if collection is None:
        collection = getattr(
            vector_store,
            "_collection",
            None,
        )

    if collection is None:
        print(
            "  ⚠ Could not access underlying collection."
        )
        print(
            "  Continuing without explicit recreation."
        )
        return

    # --------------------------------------------------------
    # Chroma collection name
    # --------------------------------------------------------

    name = getattr(
        collection,
        "name",
        "rag_chunks",
    )

    # --------------------------------------------------------
    # Get Chroma client
    # --------------------------------------------------------

    client = getattr(
        vector_store,
        "client",
        None,
    )

    if client is None:
        client = getattr(
            vector_store,
            "_client",
            None,
        )

    if client is None:

        print(
            "  ⚠ Could not access Chroma client."
        )

        return

    # --------------------------------------------------------
    # Delete collection
    # --------------------------------------------------------

    try:

        client.delete_collection(
            name=name
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # Recreate collection
    # --------------------------------------------------------

    try:

        new_collection = client.get_or_create_collection(
            name=name
        )

        if hasattr(
            vector_store,
            "collection"
        ):
            vector_store.collection = new_collection

        elif hasattr(
            vector_store,
            "_collection"
        ):
            vector_store._collection = new_collection

        print(
            "  ✓ Collection recreated"
        )

    except Exception as error:

        print(
            f"  ✗ Could not recreate collection: {error}"
        )

        raise


# ============================================================
# GET CHUNK TEXT
# ============================================================

def get_chunk_text(chunk):
    """
    Get text from EmbeddedChunk or dict.
    """

    if isinstance(chunk, dict):
        return chunk.get("text", "")

    return getattr(
        chunk,
        "text",
        "",
    )


# ============================================================
# MAIN WORKFLOW
# ============================================================

def index_corpus_workflow(
    data_dir,
    vector_dir,
    clear=False,
):
    print("=" * 70)
    print("CORPUS INDEXING WORKFLOW")
    print("=" * 70)

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print("[STEP 1] Initialize Vector Store")

    vector_store = VectorStore(
        persist_dir=str(vector_dir),
        vector_dimension=EXPECTED_EMBEDDING_DIMENSION,
    )

    print(
        f"  ✓ Collection: rag_chunks"
    )

    print(
        f"  ✓ Expected dimension: "
        f"{EXPECTED_EMBEDDING_DIMENSION}"
    )

    existing_count = vector_store.count()

    print(
        f"  ✓ Existing records: "
        f"{existing_count}"
    )

    if clear:

        print()
        print(
            "  Recreating collection..."
        )

        recreate_collection(
            vector_store
        )

        print(
            "  ✓ Collection count: "
            f"{vector_store.count()}"
        )

    # ========================================================
    # STEP 2
    # ========================================================

    print()
    print("[STEP 2] Ingest Documents")

    result = ingest(
        str(data_dir),
        token_size=DEFAULT_TOKEN_SIZE,
        token_overlap=DEFAULT_TOKEN_OVERLAP,
    )

    # Your existing ingest() returns:
    #
    # _, _, chunks, _
    #
    # based on your previous successful command.

    if not isinstance(result, tuple):
        raise TypeError(
            "Expected ingest() to return a tuple."
        )

    if len(result) < 3:
        raise ValueError(
            "ingest() returned fewer than 3 values."
        )

    files_info = result[0]
    documents = result[1]
    chunks = result[2]

    print(
        f"  ✓ Files: {len(files_info) if hasattr(files_info, '__len__') else files_info}"
    )

    print(
        f"  ✓ Documents: "
        f"{len(documents) if hasattr(documents, '__len__') else documents}"
    )

    print(
        f"  ✓ Chunks: {len(chunks)}"
    )

    if not chunks:
        raise ValueError(
            "No chunks were created from the data directory."
        )

    # ========================================================
    # STEP 3
    # ========================================================

    print()
    print("[STEP 3] Generate Embeddings")

    texts = [
        get_chunk_text(chunk)
        for chunk in chunks
    ]

    embeddings = embed(texts)

    print(
        f"  ✓ Generated: {len(embeddings)}"
    )

    print(
        f"  ✓ Cached: 0"
    )

    print(
        f"  ✓ Total: {len(embeddings)}"
    )

    # ========================================================
    # STEP 3.5
    # ========================================================

    print()
    print(
        "[STEP 3.5] Validate Embedding Dimensions"
    )

    dimensions = [
        len(vector)
        for vector in embeddings
    ]

    invalid_dimensions = [
        dimension
        for dimension in dimensions
        if dimension != EXPECTED_EMBEDDING_DIMENSION
    ]

    if invalid_dimensions:

        raise ValueError(
            "Invalid embedding dimensions: "
            f"{invalid_dimensions}"
        )

    print(
        f"  ✓ All embeddings are "
        f"{EXPECTED_EMBEDDING_DIMENSION}-dimensional"
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print()
    print(
        "[STEP 4] Prepare Vector Records"
    )

    print(
        f"  ✓ {len(chunks)} records prepared"
    )

    print(
        "    - Stable IDs (source:chunk_index)"
    )

    print(
        f"    - Embeddings "
        f"({EXPECTED_EMBEDDING_DIMENSION}-dim)"
    )

    print(
        "    - Source text"
    )

    print(
        "    - Metadata (source, chunk_index, section)"
    )

    # ========================================================
    # STEP 5
    # ========================================================

    print()
    print(
        "[STEP 5] Bulk Insert & Verify"
    )

    print("=" * 70)
    print(
        "INDEXING EMBEDDINGS & METADATA STORAGE"
    )
    print("=" * 70)

    indexing_result = index_embeddings(
        vector_store=vector_store,
        chunks=chunks,
        embeddings=embeddings,
        batch_size=DEFAULT_BATCH_SIZE,
        expected_dimension=EXPECTED_EMBEDDING_DIMENSION,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print_indexing_summary(
        indexing_result
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    if (
        indexing_result.validation_passed
        and indexing_result.failures == 0
    ):

        print()
        print(
            "✓ Indexing workflow completed successfully!"
        )

        return indexing_result

    print()
    print(
        "✗ Indexing workflow completed with errors."
    )

    return indexing_result


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Index NERI RAG corpus into Chroma."
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing source documents.",
    )

    parser.add_argument(
        "--vector-dir",
        default="outputs/chroma_db",
        help="Directory used for Chroma persistence.",
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear and recreate the vector collection.",
    )

    args = parser.parse_args()

    try:

        index_corpus_workflow(
            data_dir=args.data_dir,
            vector_dir=args.vector_dir,
            clear=args.clear,
        )

    except Exception as error:

        print()
        print(
            f"✗ Indexing failed: {error}"
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()