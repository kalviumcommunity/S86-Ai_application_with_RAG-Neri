"""Generate sample embeddings and compare their semantic similarity."""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

try:
    from .chunking import Chunk, token_count
    from .ingestion import ingest
except ImportError:
    from chunking import Chunk, token_count
    from ingestion import ingest


SAMPLE_TEXTS = [
    "How do I reset my account password?",
    "Steps to recover access to my login",
    "The cafeteria menu has pasta today",
]


@dataclass(frozen=True)
class EmbeddedChunk:
    """A searchable vector kept together with its source chunk."""

    text: str
    metadata: dict[str, str | int]
    embedding: list[float]


@dataclass(frozen=True)
class BatchEmbeddingSummary:
    """Counters from one cache-aware embedding run."""

    total_chunks: int
    embeddings_generated: int
    skipped_chunks: int
    failures: list[str]
    batches: int
    retries: int
    input_tokens: int
    estimated_cost: float


def rank_chunks(
    query_embedding: list[float],
    records: list[EmbeddedChunk],
    top_k: int | None = None,
) -> list[tuple[float, EmbeddedChunk]]:
    """Return chunks ordered from most to least similar to a query vector."""
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    ranked = sorted(
        ((cosine(query_embedding, record.embedding), record) for record in records),
        key=lambda result: result[0],
        reverse=True,
    )
    return ranked[:top_k] if top_k is not None else ranked


def embed(
    texts: list[str],
    client=None,
    model: str | None = None,
) -> list[list[float]]:
    """Return one embedding vector for each input text."""
    if not texts:
        return []

    if client is None:
        from openai import OpenAI

        load_dotenv()
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not base_url or not api_key:
            raise ValueError(
                "OPENAI_BASE_URL and OPENAI_API_KEY are required for embeddings"
            )
        client = OpenAI(base_url=base_url, api_key=api_key)

    embedding_model = model or os.getenv("EMBEDDING_MODEL") or os.getenv("EMBED_MODEL")
    if not embedding_model:
        raise ValueError("EMBED_MODEL is missing from .env")

    response = client.embeddings.create(model=embedding_model, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def embed_chunks(
    chunks: list[Chunk],
    client=None,
    model: str | None = None,
    batch_size: int = 100,
) -> list[EmbeddedChunk]:
    """Embed chunks in batches while preserving text and metadata."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    records: list[EmbeddedChunk] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = embed([chunk.text for chunk in batch], client, model)
        records.extend(
            EmbeddedChunk(chunk.text, chunk.metadata, vector)
            for chunk, vector in zip(batch, vectors)
        )
    return records


def _cache_key(chunk: Chunk, model: str) -> str:
    payload = json.dumps(
        {"model": model, "text": chunk.text, "metadata": chunk.metadata}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("records", {})


def _write_cache(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": 1, "records": records}, indent=2) + "\n",
        encoding="utf-8",
    )


def _is_transient(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    return status_code in {408, 409, 429, 500, 502, 503, 504} or (
        error.__class__.__name__ in {"RateLimitError", "APITimeoutError"}
    )


def batch_embed_chunks(
    chunks: list[Chunk],
    client=None,
    model: str | None = None,
    batch_size: int = 100,
    cache_path: str | Path = "outputs/embedding_cache.json",
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    sleep=time.sleep,
    cost_per_1k_tokens: float | None = None,
) -> tuple[list[EmbeddedChunk], BatchEmbeddingSummary]:
    """Embed uncached chunks in batches, retry transient failures, and persist results."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if max_retries < 0 or backoff_seconds < 0:
        raise ValueError("retry settings must be non-negative")

    cache_file = Path(cache_path)
    cache_model = model or os.getenv("EMBEDDING_MODEL") or os.getenv("EMBED_MODEL") or ""
    cached = _load_cache(cache_file)
    pending: list[tuple[Chunk, str]] = []
    records: list[EmbeddedChunk] = []
    skipped = 0
    for chunk in chunks:
        key = _cache_key(chunk, cache_model)
        saved = cached.get(key)
        if saved:
            records.append(EmbeddedChunk(saved["text"], saved["metadata"], saved["embedding"]))
            skipped += 1
        else:
            pending.append((chunk, key))

    failures: list[str] = []
    retries = 0
    generated = 0
    for batch_number, start in enumerate(range(0, len(pending), batch_size), start=1):
        batch = pending[start:start + batch_size]
        attempt = 0
        while True:
            try:
                vectors = embed([chunk.text for chunk, _ in batch], client, model)
                if len(vectors) != len(batch):
                    raise ValueError("embedding API returned an unexpected number of vectors")
                for (chunk, key), vector in zip(batch, vectors):
                    saved = {"text": chunk.text, "metadata": chunk.metadata, "embedding": vector}
                    cached[key] = saved
                    records.append(EmbeddedChunk(chunk.text, chunk.metadata, vector))
                _write_cache(cache_file, cached)
                generated += len(batch)
                break
            except Exception as error:
                if not _is_transient(error) or attempt >= max_retries:
                    failures.append(f"batch {batch_number} ({len(batch)} chunks): {error}")
                    break
                attempt += 1
                retries += 1
                sleep(backoff_seconds * (2 ** (attempt - 1)))

    input_tokens = sum(token_count(chunk.text) for chunk, _ in pending)
    rate = cost_per_1k_tokens
    if rate is None:
        rate = float(os.getenv("EMBEDDING_COST_PER_1K", "0.00002"))
    summary = BatchEmbeddingSummary(
        total_chunks=len(chunks),
        embeddings_generated=generated,
        skipped_chunks=skipped,
        failures=failures,
        batches=(len(pending) + batch_size - 1) // batch_size,
        retries=retries,
        input_tokens=input_tokens,
        estimated_cost=input_tokens / 1000 * rate,
    )
    return records, summary


def cosine(a: list[float], b: list[float]) -> float:
    """Compare vector direction using cosine similarity."""
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    denominator = math.sqrt(sum(value * value for value in a)) * math.sqrt(
        sum(value * value for value in b)
    )
    if denominator == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return sum(left * right for left, right in zip(a, b)) / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Override EMBED_MODEL")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    parser.add_argument("--cache", default="outputs/embedding_cache.json")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--query",
        default="What should I do before inspecting the machine?",
        help="Question to compare against the embedded chunks",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        default="outputs/similarity_ranking.md",
        help="Markdown file for the ranked sample results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    model = args.model or os.getenv("EMBEDDING_MODEL") or os.getenv("EMBED_MODEL")
    if not model:
        raise ValueError("EMBEDDING_MODEL or EMBED_MODEL is missing from .env")

    data_dir = Path(args.data_dir).resolve()
    _, _, chunks, failures = ingest(data_dir)
    records, summary = batch_embed_chunks(
        chunks,
        model=model,
        batch_size=args.batch_size,
        cache_path=args.cache,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
    )
    if not records:
        raise SystemExit("No chunks were available for embedding.")

    query_embedding = embed([args.query], model=model)[0]
    ranked = rank_chunks(query_embedding, records, args.top_k)

    print(f"model: {model}")
    print(f"records: {len(records)}")
    print(f"vector length: {len(records[0].embedding)}")
    print(f"sample values: {records[0].embedding[:5]}")
    print(f"batches: {summary.batches}")
    print(f"embeddings generated: {summary.embeddings_generated}")
    print(f"skipped chunks: {summary.skipped_chunks}")
    print(f"retries: {summary.retries}")
    print(f"failures: {len(summary.failures)}")
    print(f"input tokens: {summary.input_tokens}")
    print(f"approximate embedding cost: ${summary.estimated_cost:.6f}")
    for failure in summary.failures:
        print(f"FAILED: {failure}")
    print(f"query: {args.query}")
    print("rank | score | source | chunk | text")
    for rank, (score, record) in enumerate(ranked, start=1):
        print(
            f"{rank} | {score:.6f} | {record.metadata['source']} | "
            f"{record.metadata['chunk_index']} | {record.text.replace(chr(10), ' ')}"
        )
    if ranked:
        print(f"most similar: {ranked[0][1].text}")
        print(f"least similar: {ranked[-1][1].text}")

    report_lines = [
        "# Embedding Similarity Ranking",
        "",
        "## Metric justification",
        "",
        "This demo uses cosine similarity. It compares the direction of embedding vectors, "
        "which is useful for semantic text matching when vector magnitude should not dominate. "
        "Higher scores indicate greater similarity; distance metrics reverse that interpretation, "
        "where lower scores are better.",
        "",
        f"## Query: {args.query}",
        "",
        "| Rank | Cosine score | Source | Chunk | Section | Text |",
        "| ---: | ---: | --- | ---: | --- | --- |",
    ]
    for rank, (score, record) in enumerate(ranked, start=1):
        text = record.text.replace("\n", " ").replace("|", "\\|")
        report_lines.append(
            f"| {rank} | {score:.6f} | {record.metadata['source']} | "
            f"{record.metadata['chunk_index']} | {record.metadata['section']} | {text} |"
        )
    if ranked:
        report_lines.extend([
            "",
            f"**Most similar:** {ranked[0][1].text}",
            "",
            f"**Least similar:** {ranked[-1][1].text}",
            "",
            "A high similarity score identifies likely relevant context. It does not guarantee "
            "that the chunk is factually correct, current, complete, or safe to use without "
            "metadata, citations, freshness checks, and answer validation.",
        ])
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"ranking report: {output}")
    for name, error in failures:
        print(f"FAILED: {name}: {error}")


if __name__ == "__main__":
    main()