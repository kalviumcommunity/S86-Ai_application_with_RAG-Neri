from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

from src.grounded_generation import guarded_answer


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VECTOR_DIR = os.getenv(
    "VECTOR_DIR",
    "outputs/chroma_db"
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "rag_chunks"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    os.getenv("EMBED_MODEL", "gemini-embedding-001")
)

CHAT_MODEL = os.getenv(
    "CHAT_MODEL",
    "gemini-3.1-flash-lite"
)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="NERI RAG API",
    description="Backend API for the NERI grounded RAG assistant",
    version="1.0.0",
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Question to ask the RAG system"
    )


# --------------------------------------------------
# Source response model
# --------------------------------------------------

class Source(BaseModel):
    source: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    section: Optional[str] = None
    score: Optional[float] = None


# --------------------------------------------------
# Response model
# --------------------------------------------------

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    status: str
    reason: Optional[str] = None


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "NERI RAG API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "rag-api"
    }


# --------------------------------------------------
# Query endpoint
# --------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):

    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        # Call the existing RAG pipeline
        result = guarded_answer(
            question,
            candidate_k=10,
            final_k=3,
        )

        # --------------------------------------------------
        # Convert retrieved chunks into API sources
        # --------------------------------------------------

        sources = []

        for chunk in result.get("retrieved_chunks", []):

            metadata = chunk.get("metadata", {})

            sources.append(
                Source(
                    source=metadata.get("source"),
                    chunk_id=chunk.get("id"),
                    chunk_index=metadata.get("chunk_index"),
                    section=metadata.get("section"),
                    score=chunk.get(
                        "similarity",
                        chunk.get("score")
                    ),
                )
            )

        # --------------------------------------------------
        # Return structured response
        # --------------------------------------------------

        return QueryResponse(
            answer=result.get(
                "answer",
                "I don't have enough reliable context to answer that question."
            ),
            sources=sources,
            status=result.get(
                "status",
                "answered"
            ),
            reason=result.get("reason")
        )

    except HTTPException:
        raise

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        print(f"RAG API error: {error}")

        raise HTTPException(
            status_code=500,
            detail="RAG service failed while processing the question."
        )