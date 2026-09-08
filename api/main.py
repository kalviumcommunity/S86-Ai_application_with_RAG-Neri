"""
FastAPI backend for the RAG service.

Endpoints:
    GET  /
    GET  /health
    POST /query
    POST /documents

The /documents endpoint:
    1. Validates the uploaded file
    2. Stores it in uploads/
    3. Loads and cleans the document
    4. Chunks the document
    5. Generates embeddings
    6. Indexes the chunks in Chroma
    7. Returns an indexing summary
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.grounded_generation import guarded_answer
from src.embeddings import batch_embed_chunks
from src.ingestion import (
    DocumentRecord,
    clean_text,
    token_chunks,
)
from src.indexing import index_embeddings
from src.vector_store import VectorStore


# ============================================================
# Configuration
# ============================================================

load_dotenv()

VECTOR_DIR = Path(
    os.getenv(
        "VECTOR_DIR",
        "outputs/chroma_db",
    )
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "rag_chunks",
)

VECTOR_DIMENSION = int(
    os.getenv(
        "VECTOR_DIMENSION",
        "3072",
    )
)

UPLOAD_DIR = Path(
    os.getenv(
        "UPLOAD_DIR",
        "uploads",
    )
)

MAX_UPLOAD_SIZE = int(
    os.getenv(
        "MAX_UPLOAD_SIZE",
        str(5 * 1024 * 1024),
    )
)

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
}


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="RAG Service API",
    description="Backend API for the grounded RAG pipeline.",
    version="1.0.0",
)


# ============================================================
# Request / Response Models
# ============================================================


class QueryRequest(BaseModel):
    """
    Request body for /query.
    """

    question: str = Field(
        min_length=3,
        max_length=1000,
        description="Question to ask the RAG system.",
    )


class Source(BaseModel):
    """
    Source information returned to the client.
    """

    source: str | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    section: str | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    """
    Structured response from the RAG pipeline.
    """

    answer: str
    sources: list[Source]
    status: str
    reason: str | None = None


class DocumentSummary(BaseModel):
    """
    Summary of the document indexing operation.
    """

    document: str
    chunks: int
    indexed: int


class UploadResponse(BaseModel):
    """
    Response returned after successful document indexing.
    """

    status: str
    filename: str
    summary: DocumentSummary


# ============================================================
# Utility Functions
# ============================================================


def get_vector_store() -> VectorStore:
    """
    Create the project's vector store.
    """

    return VectorStore(
        persist_dir=VECTOR_DIR,
        collection_name=COLLECTION_NAME,
        vector_dimension=VECTOR_DIMENSION,
    )


def validate_upload(filename: str | None) -> str:
    """
    Validate uploaded filename and return its extension.
    """

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported file type. "
                "Supported formats: .txt, .md, .pdf"
            ),
        )

    return suffix


async def store_upload(file: UploadFile) -> Path:
    """
    Validate and safely store an uploaded document.
    """

    validate_upload(file.filename)

    # Read uploaded file.
    content = await file.read()

    # Reject empty files.
    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # Reject oversized files.
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is too large. Maximum allowed size is "
                f"{MAX_UPLOAD_SIZE} bytes."
            ),
        )

    # Create upload directory.
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Only use the filename itself.
    # This prevents paths such as ../../file.txt.
    safe_filename = Path(
        file.filename
    ).name

    path = UPLOAD_DIR / safe_filename

    path.write_bytes(content)

    return path


# ============================================================
# Document Processing
# ============================================================


def process_uploaded_document(
    path: Path,
) -> dict:
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

    # --------------------------------------------------------
    # Step 1: Load document
    # --------------------------------------------------------

    try:
        # The existing ingestion loader handles
        # the project's supported document formats.
        from src.ingestion import load_text

        raw_text = load_text(path)

    except Exception as error:
        raise ValueError(
            f"Could not read document: {error}"
        ) from error

    # --------------------------------------------------------
    # Step 2: Clean text
    # --------------------------------------------------------

    cleaned_text = clean_text(
        raw_text
    )

    if not cleaned_text.strip():
        raise ValueError(
            "Document contains no readable text."
        )

    # --------------------------------------------------------
    # Step 3: Create DocumentRecord
    # --------------------------------------------------------

    document = DocumentRecord(
        source=path.name,
        path=str(path),
        text=cleaned_text,
    )

    # --------------------------------------------------------
    # Step 4: Chunk document
    # --------------------------------------------------------

    chunks = token_chunks(
        document,
        size=64,
        overlap=16,
    )

    if not chunks:
        raise ValueError(
            "No chunks were generated from the document."
        )

    # --------------------------------------------------------
    # Step 5: Generate embeddings
    # --------------------------------------------------------

    embedded_chunks, embedding_summary = batch_embed_chunks(
        chunks
    )

    if not embedded_chunks:
        raise ValueError(
            "No embeddings were generated."
        )

    # --------------------------------------------------------
    # Step 6: Index embeddings
    # --------------------------------------------------------

    vector_store = get_vector_store()

    indexing_result = index_embeddings(
        vector_store,
        embedded_chunks,
        embeddings=None,
        batch_size=100,
        expected_dimension=VECTOR_DIMENSION,
    )

    # --------------------------------------------------------
    # Step 7: Return summary
    # --------------------------------------------------------

    indexed_count = getattr(
        indexing_result,
        "inserted",
        0,
    ) + getattr(
        indexing_result,
        "updated",
        0,
    )

    return {
        "document": str(path),
        "chunks": len(chunks),
        "indexed": indexed_count,
        "embedding_summary": {
            "total_chunks": embedding_summary.total_chunks,
            "embeddings_generated": (
                embedding_summary.embeddings_generated
            ),
            "skipped_chunks": (
                embedding_summary.skipped_chunks
            ),
            "failures": embedding_summary.failures,
        },
    }


# ============================================================
# Root Endpoint
# ============================================================


@app.get("/")
def root():
    """
    Basic API information.
    """

    return {
        "service": "RAG Service API",
        "status": "running",
        "endpoints": {
            "query": "POST /query",
            "upload": "POST /documents",
            "health": "GET /health",
        },
    }


# ============================================================
# Health Endpoint
# ============================================================


@app.get("/health")
def health():
    """
    Health check endpoint.
    """

    try:
        vector_store = get_vector_store()

        return {
            "status": "healthy",
            "vector_store": "connected",
            "indexed_chunks": vector_store.count(),
        }

    except Exception as error:
        return {
            "status": "unhealthy",
            "error": str(error),
        }


# ============================================================
# Query Endpoint
# ============================================================


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_rag(
    request: QueryRequest,
):
    """
    Ask a question and receive a grounded RAG answer.
    """

    try:

        result = guarded_answer(
            request.question,
            candidate_k=10,
            final_k=3,
        )

        sources = []

        for chunk in result.get(
            "retrieved_chunks",
            [],
        ):

            metadata = chunk.get(
                "metadata",
                {},
            )

            sources.append(
                Source(
                    source=metadata.get(
                        "source"
                    ),
                    chunk_id=chunk.get(
                        "id"
                    ),
                    chunk_index=metadata.get(
                        "chunk_index"
                    ),
                    section=metadata.get(
                        "section"
                    ),
                    score=chunk.get(
                        "similarity"
                    ),
                )
            )

        return QueryResponse(
            answer=result.get(
                "answer",
                "",
            ),
            sources=sources,
            status=result.get(
                "status",
                "answered",
            ),
            reason=result.get(
                "reason"
            ),
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"RAG service failed: {error}",
        ) from error


# ============================================================
# Document Upload Endpoint
# ============================================================


@app.post(
    "/documents",
    response_model=UploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload, process, embed, and index a document.

    Supported:
        .txt
        .md
        .pdf
    """

    try:

        # ----------------------------------------------------
        # Step 1: Store upload
        # ----------------------------------------------------

        path = await store_upload(
            file
        )

        # ----------------------------------------------------
        # Step 2: Process and index
        # ----------------------------------------------------

        summary = process_uploaded_document(
            path
        )

        # ----------------------------------------------------
        # Step 3: Return response
        # ----------------------------------------------------

        return UploadResponse(
            status="indexed",
            filename=file.filename,
            summary=DocumentSummary(
                document=summary["document"],
                chunks=summary["chunks"],
                indexed=summary["indexed"],
            ),
        )

    except HTTPException:
        raise

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Document indexing failed: {error}"
            ),
        ) from error


# ============================================================
# Run directly
# ============================================================


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )