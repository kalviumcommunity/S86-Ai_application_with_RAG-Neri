import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.grounded_generation import guarded_answer


# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))

INPUT_COST_PER_1K = float(
    os.getenv("MODEL_INPUT_COST_PER_1K", "0.00015")
)

OUTPUT_COST_PER_1K = float(
    os.getenv("MODEL_OUTPUT_COST_PER_1K", "0.00060")
)

CHAT_MODEL = os.getenv("CHAT_MODEL", "unknown")

LOG_FILE = Path(
    os.getenv("RAG_LOG_FILE", "outputs/rag_requests.jsonl")
)


# ============================================================
# LOGGING SETUP
# ============================================================

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(console_handler)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="NERI RAG API",
    description="Backend API for the grounded NERI RAG service.",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        description="Question to ask the RAG system."
    )


class Source(BaseModel):
    source: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    section: Optional[str] = None
    score: Optional[float] = None


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    cache_hit: bool
    latency_ms: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    status: str
    reason: Optional[str] = None
    usage: Usage


# ============================================================
# IN-MEMORY QUERY CACHE
# ============================================================

query_cache = {}


def cache_key(question: str) -> str:
    """
    Create a stable cache key for an identical question.
    """

    normalized_question = " ".join(
        question.strip().lower().split()
    )

    return hashlib.sha256(
        normalized_question.encode("utf-8")
    ).hexdigest()


def get_cached_answer(question: str):
    """
    Return cached response if it exists and has not expired.
    """

    key = cache_key(question)

    cached = query_cache.get(key)

    if cached is None:
        return None

    age = time.time() - cached["created_at"]

    if age > CACHE_TTL_SECONDS:
        query_cache.pop(key, None)

        logger.info(
            "CACHE EXPIRED | key=%s",
            key[:12]
        )

        return None

    logger.info(
        "CACHE HIT | key=%s",
        key[:12]
    )

    return cached["response"]


def save_cached_answer(
    question: str,
    response: dict
):
    """
    Save an API response in the cache.
    """

    key = cache_key(question)

    query_cache[key] = {
        "created_at": time.time(),
        "response": response,
    }

    logger.info(
        "CACHE SAVE | key=%s",
        key[:12]
    )


# ============================================================
# TOKEN / COST HELPERS
# ============================================================

def estimate_tokens(text: str) -> int:
    """
    Simple token estimate.

    Approximation:
    1 token ~= 4 characters.
    """

    if not text:
        return 0

    return max(1, len(text) // 4)


def estimate_cost(
    input_tokens: int,
    output_tokens: int
) -> float:

    input_cost = (
        input_tokens / 1000
    ) * INPUT_COST_PER_1K

    output_cost = (
        output_tokens / 1000
    ) * OUTPUT_COST_PER_1K

    return round(
        input_cost + output_cost,
        6
    )


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def extract_sources(result: dict) -> list[dict]:
    """
    Convert RAG source information into API-friendly metadata.
    """

    sources = []

    for source in result.get("sources", []):
        if isinstance(source, str):

            sources.append(
                {
                    "source": source,
                    "chunk_id": None,
                    "chunk_index": None,
                    "section": None,
                    "score": None,
                }
            )

        elif isinstance(source, dict):

            sources.append(
                {
                    "source": source.get("source"),
                    "chunk_id": source.get("chunk_id"),
                    "chunk_index": source.get("chunk_index"),
                    "section": source.get("section"),
                    "score": source.get("score"),
                }
            )

    return sources


# ============================================================
# STRUCTURED LOGGING
# ============================================================

def write_usage_log(record: dict):
    """
    Write one JSON record per request.
    """

    with LOG_FILE.open(
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


def log_rag_request(
    request_id: str,
    question: str,
    answer: str,
    sources: list,
    cache_hit: bool,
    input_tokens: int,
    output_tokens: int,
    estimated_cost: float,
    latency_ms: float,
    status: str,
):
    """
    Log useful information for debugging and monitoring.
    """

    record = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "request_id": request_id,

        "question": question,

        "answer_preview": answer[:180],

        "sources": sources,

        "cache_hit": cache_hit,

        "input_tokens": input_tokens,

        "output_tokens": output_tokens,

        "estimated_cost": estimated_cost,

        "latency_ms": latency_ms,

        "status": status,

        "model": CHAT_MODEL,
    }

    logger.info(
        "RAG REQUEST | %s",
        json.dumps(record)
    )

    write_usage_log(record)


# ============================================================
# USAGE REPORT
# ============================================================

def load_usage_records() -> list[dict]:
    """
    Load all stored request logs.
    """

    if not LOG_FILE.exists():
        return []

    records = []

    with LOG_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                continue

    return records


def summarize_usage(
    records: list[dict]
) -> dict:

    total_requests = len(records)

    cache_hits = sum(
        1
        for record in records
        if record.get("cache_hit") is True
    )

    total_cost = sum(
        record.get(
            "estimated_cost",
            0
        )
        for record in records
    )

    total_input_tokens = sum(
        record.get(
            "input_tokens",
            0
        )
        for record in records
    )

    total_output_tokens = sum(
        record.get(
            "output_tokens",
            0
        )
        for record in records
    )

    total_latency = sum(
        record.get(
            "latency_ms",
            0
        )
        for record in records
    )

    average_latency = (
        total_latency / total_requests
        if total_requests
        else 0
    )

    cache_hit_rate = (
        cache_hits / total_requests
        if total_requests
        else 0
    )

    return {
        "total_requests": total_requests,

        "cache_hits": cache_hits,

        "cache_misses": (
            total_requests - cache_hits
        ),

        "cache_hit_rate": round(
            cache_hit_rate,
            2
        ),

        "total_input_tokens": (
            total_input_tokens
        ),

        "total_output_tokens": (
            total_output_tokens
        ),

        "total_estimated_cost": round(
            total_cost,
            6
        ),

        "average_latency_ms": round(
            average_latency,
            2
        ),
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "NERI RAG API",
        "status": "running",
        "endpoints": [
            "/query",
            "/usage",
            "/health",
            "/docs",
        ],
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "cache_entries": len(query_cache),
        "log_file": str(LOG_FILE),
    }


# ============================================================
# QUERY ENDPOINT
# ============================================================

@app.post(
    "/query",
    response_model=QueryResponse
)
def query_rag(
    request: QueryRequest
):

    request_id = str(uuid.uuid4())

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # CHECK CACHE
    # --------------------------------------------------------

    cached_response = get_cached_answer(
        question
    )

    if cached_response is not None:

        latency_ms = round(
            (
                time.perf_counter()
                - start_time
            ) * 1000,
            2
        )

        cached_response = dict(
            cached_response
        )

        cached_response["usage"] = dict(
            cached_response["usage"]
        )

        cached_response["usage"][
            "cache_hit"
        ] = True

        cached_response["usage"][
            "latency_ms"
        ] = latency_ms


        log_rag_request(
            request_id=request_id,
            question=question,
            answer=cached_response["answer"],
            sources=cached_response["sources"],
            cache_hit=True,
            input_tokens=cached_response[
                "usage"
            ]["input_tokens"],
            output_tokens=cached_response[
                "usage"
            ]["output_tokens"],
            estimated_cost=cached_response[
                "usage"
            ]["estimated_cost"],
            latency_ms=latency_ms,
            status=cached_response[
                "status"
            ],
        )

        return cached_response


    # --------------------------------------------------------
    # CACHE MISS
    # --------------------------------------------------------

    logger.info(
        "CACHE MISS | request_id=%s",
        request_id
    )


    try:

        # ----------------------------------------------------
        # CALL EXISTING RAG PIPELINE
        # ----------------------------------------------------

        result = guarded_answer(
            question,
            candidate_k=10,
            final_k=3,
        )


        answer = result.get(
            "answer",
            ""
        )

        sources = extract_sources(
            result
        )

        status = result.get(
            "status",
            "answered"
        )

        reason = result.get(
            "reason"
        )


        # ----------------------------------------------------
        # TOKEN / COST ESTIMATION
        # ----------------------------------------------------

        input_tokens = estimate_tokens(
            question
        )

        output_tokens = estimate_tokens(
            answer
        )

        estimated_cost = estimate_cost(
            input_tokens,
            output_tokens
        )


        # ----------------------------------------------------
        # LATENCY
        # ----------------------------------------------------

        latency_ms = round(
            (
                time.perf_counter()
                - start_time
            ) * 1000,
            2
        )


        response = {
            "answer": answer,

            "sources": sources,

            "status": status,

            "reason": reason,

            "usage": {
                "input_tokens": input_tokens,

                "output_tokens": output_tokens,

                "estimated_cost": estimated_cost,

                "cache_hit": False,

                "latency_ms": latency_ms,
            },
        }


        # ----------------------------------------------------
        # SAVE CACHE
        # ----------------------------------------------------

        save_cached_answer(
            question,
            response
        )


        # ----------------------------------------------------
        # WRITE LOG
        # ----------------------------------------------------

        log_rag_request(
            request_id=request_id,
            question=question,
            answer=answer,
            sources=sources,
            cache_hit=False,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            latency_ms=latency_ms,
            status=status,
        )


        return response


    except ValueError as error:

        logger.error(
            "BAD REQUEST | request_id=%s | error=%s",
            request_id,
            error
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    except Exception as error:

        logger.exception(
            "RAG FAILURE | request_id=%s | error=%s",
            request_id,
            error
        )

        raise HTTPException(
            status_code=500,
            detail="RAG service failed."
        )


# ============================================================
# USAGE REPORT ENDPOINT
# ============================================================

@app.get("/usage")
def usage_report():

    records = load_usage_records()

    summary = summarize_usage(
        records
    )

    return {
        "status": "ok",

        "summary": summary,

        "model": CHAT_MODEL,

        "cache_ttl_seconds": (
            CACHE_TTL_SECONDS
        ),

        "log_file": str(LOG_FILE),
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )