"""
Runtime document upload and indexing pipeline.

Flow:
    uploaded file
        ↓
    ingestion
        ↓
    chunking
        ↓
    embeddings
        ↓
    ChromaDB
"""

from pathlib import Path
import shutil

from .ingestion import load_text, clean_text, token_chunks, DocumentRecord
from .embeddings import batch_embed_chunks
from .indexing import index_embeddings
from .vector_store import VectorStore


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

UPLOAD_DIR = Path("uploads")

VECTOR_DIR = Path("outputs/chroma_db")

COLLECTION_NAME = "rag_chunks"

VECTOR_DIMENSION = 3072


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
}


MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


# ---------------------------------------------------------
# Validate upload
# ---------------------------------------------------------

def validate_file(filename: str, content: bytes) -> None:
    """
    Validate uploaded document.

    Raises:
        ValueError: If file is invalid.
    """

    if not filename:
        raise ValueError("Filename is required.")

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if not content:
        raise ValueError("Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise ValueError(
            f"File is too large. Maximum size is "
            f"{MAX_FILE_SIZE // (1024 * 1024)} MB."
        )


# ---------------------------------------------------------
# Save upload
# ---------------------------------------------------------

def save_uploaded_file(filename: str, content: bytes) -> Path:
    """
    Safely save uploaded file into uploads/.
    """

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Only keep the filename, preventing directory traversal.
    safe_name = Path(filename).name

    destination = UPLOAD_DIR / safe_name

    destination.write_bytes(content)

    return destination


# ---------------------------------------------------------
# Process document
# ---------------------------------------------------------

def process_document(path: Path) -> dict:
    """
    Process one uploaded document through:

        load
        ↓
        clean
        ↓
        chunk
        ↓
        embed
        ↓
        index
    """

    # -----------------------------------------------------
    # 1. Load document
    # -----------------------------------------------------

    raw_text = load_text(path)

    if not raw_text.strip():
        raise ValueError("Uploaded document contains no readable text.")

    # -----------------------------------------------------
    # 2. Clean text
    # -----------------------------------------------------

    cleaned_text = clean_text(raw_text)

    if not cleaned_text.strip():
        raise ValueError("Document contains no usable text after cleaning.")

    # -----------------------------------------------------
    # 3. Create DocumentRecord
    # -----------------------------------------------------

    document = DocumentRecord(
        source=path.name,
        path=str(path),
        text=cleaned_text,
    )

    # -----------------------------------------------------
    # 4. Chunk document
    # -----------------------------------------------------

    chunks = token_chunks(
        document,
        size=64,
        overlap=16,
    )

    if not chunks:
        raise ValueError("No chunks were generated from the document.")

    # -----------------------------------------------------
    # 5. Generate embeddings
    # -----------------------------------------------------

    embedded_chunks, embedding_summary = batch_embed_chunks(
        chunks
    )

    if not embedded_chunks:
        raise RuntimeError("No embeddings were generated.")

    # -----------------------------------------------------
    # 6. Connect to existing vector store
    # -----------------------------------------------------

    vector_store = VectorStore(
        persist_dir=VECTOR_DIR,
        collection_name=COLLECTION_NAME,
        vector_dimension=VECTOR_DIMENSION,
    )

    # -----------------------------------------------------
    # 7. Index embeddings
    # -----------------------------------------------------

    indexing_result = index_embeddings(
        vector_store,
        embedded_chunks,
        expected_dimension=VECTOR_DIMENSION,
    )

    # -----------------------------------------------------
    # 8. Return summary
    # -----------------------------------------------------

    return {
        "document": str(path),
        "filename": path.name,
        "characters": len(cleaned_text),
        "chunks": len(chunks),
        "embeddings_generated": embedding_summary.embeddings_generated,
        "embedding_failures": len(embedding_summary.failures),
        "indexed": indexing_result.indexed_count,
        "inserted": indexing_result.inserted,
        "updated": indexing_result.updated,
        "indexing_failures": indexing_result.failures,
        "validation_passed": indexing_result.validation_passed,
        "vector_store_count": vector_store.count(),
    }


# ---------------------------------------------------------
# Complete upload pipeline
# ---------------------------------------------------------

def upload_and_index(filename: str, content: bytes) -> dict:
    """
    Complete runtime upload and indexing operation.
    """

    # Validate
    validate_file(filename, content)

    # Save
    path = save_uploaded_file(filename, content)

    try:
        # Process
        summary = process_document(path)

        return {
            "status": "indexed",
            "filename": filename,
            "summary": summary,
        }

    except Exception:
        # Remove failed upload so broken documents
        # don't remain in the uploads directory.
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

        raise


# ---------------------------------------------------------
# Manual test
# ---------------------------------------------------------

def main():
    print("=" * 60)
    print("RUNTIME DOCUMENT UPLOAD & INDEXING TEST")
    print("=" * 60)

    test_file = Path("test_runtime_document.txt")

    test_content = b"""
Runtime indexing test document.

This document was uploaded after the RAG API started.

Technicians must inspect equipment before restarting
a machine after detecting abnormal vibration.

This content is being used to verify runtime indexing.
"""

    test_file.write_bytes(test_content)

    try:

        result = upload_and_index(
            test_file.name,
            test_content,
        )

        print("\nUpload result:")
        print(result)

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)

    finally:

        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    main()