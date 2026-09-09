import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "rag_requests.jsonl"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Logging configuration
# --------------------------------------------------

logger = logging.getLogger("rag_app")

if not logger.handlers:

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


# --------------------------------------------------
# Cost configuration
# --------------------------------------------------

MODEL_INPUT_COST_PER_1K = 0.00015
MODEL_OUTPUT_COST_PER_1K = 0.00060


def estimate_cost(
    input_tokens,
    output_tokens,
):
    """
    Estimate generation cost using the configured
    per-1K-token rates.
    """

    input_cost = (
        input_tokens / 1000
    ) * MODEL_INPUT_COST_PER_1K

    output_cost = (
        output_tokens / 1000
    ) * MODEL_OUTPUT_COST_PER_1K

    return round(
        input_cost + output_cost,
        6,
    )


def approximate_tokens(text):
    """
    Simple token estimate.

    This is intentionally approximate.
    """

    if not text:
        return 0

    # Rough estimate:
    # approximately 4 characters per token.
    return max(
        1,
        len(text) // 4,
    )


def generate_request_id():
    return str(uuid.uuid4())


def create_usage_record(
    question,
    answer="",
    sources=None,
    cache_hit=False,
    latency_ms=0,
    input_tokens=None,
    output_tokens=None,
    status="answered",
    error=None,
):
    """
    Create a structured usage record.
    """

    sources = sources or []

    if input_tokens is None:
        input_tokens = approximate_tokens(
            question
        )

    if output_tokens is None:
        output_tokens = approximate_tokens(
            answer
        )

    estimated_cost = estimate_cost(
        input_tokens,
        output_tokens,
    )

    return {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "request_id": generate_request_id(),

        "question": question,

        "answer_preview": answer[:180],

        "sources": sources,

        "cache_hit": cache_hit,

        "input_tokens": input_tokens,

        "output_tokens": output_tokens,

        "estimated_cost": estimated_cost,

        "latency_ms": round(
            latency_ms,
            2,
        ),

        "status": status,

        "error": error,
    }


def log_rag_request(record):
    """
    Write structured JSON log.
    """

    # Console log
    logger.info(
        json.dumps(
            record,
            ensure_ascii=False,
        )
    )

    # JSONL file
    with LOG_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def summarize_usage():
    """
    Read the JSONL logs and create
    a usage summary.
    """

    if not LOG_FILE.exists():

        return {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_hit_rate": 0,
            "total_estimated_cost": 0,
            "average_latency_ms": 0,
            "answered_requests": 0,
            "refused_requests": 0,
            "error_requests": 0,
        }

    records = []

    with LOG_FILE.open(
        "r",
        encoding="utf-8",
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

    total_requests = len(records)

    cache_hits = sum(
        1
        for item in records
        if item.get("cache_hit")
    )

    total_cost = sum(
        item.get(
            "estimated_cost",
            0,
        )
        for item in records
    )

    total_latency = sum(
        item.get(
            "latency_ms",
            0,
        )
        for item in records
    )

    answered = sum(
        1
        for item in records
        if item.get("status")
        == "answered"
    )

    refused = sum(
        1
        for item in records
        if item.get("status")
        in {
            "refused",
            "refused_weak_context",
        }
    )

    errors = sum(
        1
        for item in records
        if item.get("status")
        == "error"
    )

    return {
        "total_requests": total_requests,

        "cache_hits": cache_hits,

        "cache_hit_rate": round(
            cache_hits
            / max(total_requests, 1),
            2,
        ),

        "total_estimated_cost": round(
            total_cost,
            6,
        ),

        "average_latency_ms": round(
            total_latency
            / max(total_requests, 1),
            2,
        ),

        "answered_requests": answered,

        "refused_requests": refused,

        "error_requests": errors,
    }


def save_usage_report(
    output_path="reports/usage_summary.json",
):
    """
    Generate and save the usage report.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = summarize_usage()

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    return summary


def print_usage_summary():
    summary = summarize_usage()

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":

    print("=" * 60)
    print("RAG USAGE MONITORING")
    print("=" * 60)

    summary = save_usage_report()

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print(
        f"Log file: {LOG_FILE}"
    )

    print(
        "Usage report generated."
    )