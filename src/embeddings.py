"""Generate sample embeddings and compare their semantic similarity."""

import argparse
import math
import os

from dotenv import load_dotenv


SAMPLE_TEXTS = [
    "How do I reset my account password?",
    "Steps to recover access to my login",
    "The cafeteria menu has pasta today",
]


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

    embedding_model = model or os.getenv("EMBED_MODEL")
    if not embedding_model:
        raise ValueError("EMBED_MODEL is missing from .env")

    response = client.embeddings.create(model=embedding_model, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vectors = embed(SAMPLE_TEXTS, model=args.model)
    print(f"dimension: {len(vectors[0])}")
    print(f"first 8 values: {vectors[0][:8]}")
    print(f"password vs login recovery: {cosine(vectors[0], vectors[1]):.6f}")
    print(f"password vs cafeteria menu: {cosine(vectors[0], vectors[2]):.6f}")


if __name__ == "__main__":
    main()