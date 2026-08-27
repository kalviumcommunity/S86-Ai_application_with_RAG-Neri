"""Generate sample embeddings and compare their semantic similarity."""

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

try:
    from .chunking import Chunk
    from .ingestion import ingest
except ImportError:
    from chunking import Chunk
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
    parser.add_argument("--samples", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    model = args.model or os.getenv("EMBEDDING_MODEL") or os.getenv("EMBED_MODEL")
    if not model:
        raise ValueError("EMBEDDING_MODEL or EMBED_MODEL is missing from .env")

    data_dir = Path(args.data_dir).resolve()
    _, _, chunks, failures = ingest(data_dir)
    records = embed_chunks(chunks, model=model, batch_size=args.batch_size)
    if not records:
        raise SystemExit("No chunks were available for embedding.")

    print(f"model: {model}")
    print(f"records: {len(records)}")
    print(f"vector length: {len(records[0].embedding)}")
    print(f"sample values: {records[0].embedding[:5]}")
    for name, error in failures:
        print(f"FAILED: {name}: {error}")


if __name__ == "__main__":
    main()