"""
Embedding utilities for the RAG application.

Supports:
- OpenAI-compatible embedding APIs
- Google Gemini OpenAI-compatible endpoint
- Batch embedding
- Persistent embedding cache
- Stable cache keys
- Embedding ranking
- Retry handling
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

# Load .env from project root
load_dotenv(ROOT_DIR / ".env")


# ============================================================
# CONFIGURATION
# ============================================================

# Gemini's OpenAI-compatible endpoint can return 3072-dimensional
# embeddings when using gemini-embedding-001.
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

DEFAULT_CACHE_PATH = ROOT_DIR / "outputs" / "embedding_cache.json"

DEFAULT_BATCH_SIZE = 100

DEFAULT_MAX_RETRIES = 3

DEFAULT_BACKOFF_SECONDS = 1.0

DEFAULT_COST_PER_1K = 0.00002


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class EmbeddedChunk:
    """
    Represents a text chunk together with its embedding.
    """

    text: str
    metadata: dict
    embedding: list[float]


@dataclass
class BatchEmbeddingSummary:
    """
    Summary of a batch embedding operation.
    """

    total_chunks: int
    embeddings_generated: int
    skipped_chunks: int
    failures: list[str]
    batches: int
    retries: int
    input_tokens: int
    estimated_cost: float


# ============================================================
# CLIENT
# ============================================================

def _get_client(client=None):
    """
    Return an OpenAI-compatible client.

    If a client is passed, use it.

    Otherwise create a client using:
        OPENAI_API_KEY
        OPENAI_BASE_URL
    """

    if client is not None:
        return client

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing from environment variables. "
            "Make sure your .env file exists in the project root."
        )

    # base_url is optional for standard OpenAI.
    # It is required for Gemini's OpenAI-compatible endpoint.
    if base_url:
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    return OpenAI(
        api_key=api_key,
    )


# ============================================================
# MODEL
# ============================================================

def _get_model(model: Optional[str] = None) -> str:
    """
    Resolve embedding model.

    Priority:
        1. Explicit model argument
        2. EMBEDDING_MODEL
        3. EMBED_MODEL
        4. Default Gemini embedding model
    """

    resolved_model = (
        model
        or os.getenv("EMBEDDING_MODEL")
        or os.getenv("EMBED_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )

    return resolved_model


# ============================================================
# TOKEN COUNT
# ============================================================

def token_count(text: str) -> int:
    """
    Approximate token count.

    We intentionally avoid requiring tiktoken because the
    embedding endpoint may use a different tokenizer.

    A simple approximation of 1 token ~= 4 characters is
    sufficient for cost estimation.
    """

    if not text:
        return 0

    return max(1, len(text) // 4)


# ============================================================
# EMBEDDING API
# ============================================================

def embed(
    texts: list[str],
    client=None,
    model: Optional[str] = None,
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts:
            List of strings to embed.

        client:
            Optional OpenAI-compatible client.

        model:
            Optional embedding model.

    Returns:
        List of embedding vectors.

    Raises:
        ValueError:
            If input is invalid or API response is invalid.
    """

    if not texts:
        return []

    if not isinstance(texts, list):
        raise ValueError("texts must be a list of strings")

    for text in texts:
        if not isinstance(text, str):
            raise ValueError("Every item in texts must be a string")

    client = _get_client(client)
    model = _get_model(model)

    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    if response is None:
        raise ValueError("Embedding API returned no response")

    data = getattr(response, "data", None)

    if not data:
        raise ValueError("Embedding API returned no embedding data")

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # Some OpenAI-compatible APIs may return:
    #
    #     index=None
    #
    # Therefore:
    #
    #     sorted(response.data, key=lambda item: item.index)
    #
    # can crash with:
    #
    # TypeError:
    # '<' not supported between instances of 'int' and 'NoneType'
    #
    # We handle missing indexes safely.
    # --------------------------------------------------------

    vectors = [getattr(item, "embedding", None) for item in data]

    if any(vector is None for vector in vectors):
        raise ValueError(
            "Embedding API returned an item without an embedding vector"
        )

    # If indexes are present and valid, restore original order.
    indexes = [getattr(item, "index", None) for item in data]

    if all(isinstance(index, int) for index in indexes):
        ordered_items = sorted(
            zip(indexes, vectors),
            key=lambda pair: pair[0],
        )

        vectors = [vector for _, vector in ordered_items]

    else:
        # Gemini's OpenAI-compatible API may return None indexes.
        # In that case the response order is used.
        vectors = list(vectors)

    # Verify number of vectors.
    if len(vectors) != len(texts):
        raise ValueError(
            f"Embedding API returned {len(vectors)} vectors "
            f"for {len(texts)} input texts"
        )

    # Convert values to normal Python floats.
    normalized_vectors = []

    for vector in vectors:
        normalized_vector = [float(value) for value in vector]

        if not normalized_vector:
            raise ValueError(
                "Embedding API returned an empty embedding vector"
            )

        normalized_vectors.append(normalized_vector)

    return normalized_vectors


# ============================================================
# CACHE KEY
# ============================================================

def _cache_key(chunk, model: str) -> str:
    """
    Generate a stable SHA256 cache key.

    The key depends on:
        - model
        - chunk text
        - chunk metadata
    """

    metadata = getattr(chunk, "metadata", {})

    payload = {
        "model": model,
        "text": chunk.text,
        "metadata": metadata,
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


# ============================================================
# LOAD CACHE
# ============================================================

def _load_cache(cache_path: Path) -> dict:
    """
    Load embedding cache.

    Returns:
        Dictionary of cached records.
    """

    if not cache_path.exists():
        return {}

    try:
        with cache_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(data, dict):
        return {}

    records = data.get("records", {})

    if not isinstance(records, dict):
        return {}

    return records


# ============================================================
# WRITE CACHE
# ============================================================

def _write_cache(
    cache_path: Path,
    records: dict,
) -> None:
    """
    Persist embedding cache to disk.
    """

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "version": 1,
        "records": records,
    }

    temporary_path = cache_path.with_suffix(
        cache_path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
        )

    temporary_path.replace(cache_path)


# ============================================================
# TRANSIENT ERROR CHECK
# ============================================================

def _is_transient(error: Exception) -> bool:
    """
    Determine whether an error is likely temporary.

    Retry:
        - rate limits
        - server errors
        - connection errors
        - timeout errors

    Do not retry:
        - programming errors
        - invalid input
        - authentication problems
    """

    error_text = str(error).lower()

    transient_keywords = [
        "rate limit",
        "rate_limit",
        "429",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "server error",
    ]

    return any(
        keyword in error_text
        for keyword in transient_keywords
    )


# ============================================================
# BATCH EMBEDDING
# ============================================================

def batch_embed_chunks(
    chunks: list,
    client=None,
    model: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable = time.sleep,
    cost_per_1k_tokens: Optional[float] = None,
) -> tuple[list[EmbeddedChunk], BatchEmbeddingSummary]:
    """
    Embed chunks in batches with caching and retry handling.

    Args:
        chunks:
            List of Chunk objects.

        client:
            Optional OpenAI-compatible client.

        model:
            Embedding model.

        batch_size:
            Number of chunks sent per API request.

        cache_path:
            Location of persistent cache.

        max_retries:
            Maximum retries for transient errors.

        backoff_seconds:
            Initial retry delay.

        sleep:
            Sleep function, injectable for tests.

        cost_per_1k_tokens:
            Optional cost estimation rate.

    Returns:
        (
            list of EmbeddedChunk,
            BatchEmbeddingSummary
        )
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero"
        )

    if max_retries < 0:
        raise ValueError(
            "max_retries must be non-negative"
        )

    if backoff_seconds < 0:
        raise ValueError(
            "backoff_seconds must be non-negative"
        )

    if not chunks:
        return (
            [],
            BatchEmbeddingSummary(
                total_chunks=0,
                embeddings_generated=0,
                skipped_chunks=0,
                failures=[],
                batches=0,
                retries=0,
                input_tokens=0,
                estimated_cost=0.0,
            ),
        )

    # --------------------------------------------------------
    # Resolve model.
    # --------------------------------------------------------

    resolved_model = _get_model(model)

    # --------------------------------------------------------
    # Cache.
    # --------------------------------------------------------

    cache_file = Path(cache_path)

    cached = _load_cache(cache_file)

    pending = []

    records = []

    skipped = 0

    # --------------------------------------------------------
    # Check cache.
    # --------------------------------------------------------

    for chunk in chunks:

        key = _cache_key(
            chunk,
            resolved_model,
        )

        saved = cached.get(key)

        if saved:

            embedding = saved.get("embedding")

            text = saved.get("text")

            metadata = saved.get("metadata", {})

            if (
                isinstance(embedding, list)
                and embedding
                and isinstance(text, str)
                and isinstance(metadata, dict)
            ):
                records.append(
                    EmbeddedChunk(
                        text=text,
                        metadata=metadata,
                        embedding=[
                            float(value)
                            for value in embedding
                        ],
                    )
                )

                skipped += 1

                continue

        pending.append(
            (chunk, key)
        )

    # --------------------------------------------------------
    # API client.
    #
    # IMPORTANT:
    # Only create the client if there are uncached chunks.
    # This makes cached runs work without unnecessary API
    # initialization.
    # --------------------------------------------------------

    api_client = None

    if pending:
        api_client = _get_client(client)

    # --------------------------------------------------------
    # Process batches.
    # --------------------------------------------------------

    failures = []

    retries = 0

    generated = 0

    for batch_number, start in enumerate(
        range(
            0,
            len(pending),
            batch_size,
        ),
        start=1,
    ):

        batch = pending[
            start:start + batch_size
        ]

        attempt = 0

        while True:

            try:

                texts = [
                    chunk.text
                    for chunk, _ in batch
                ]

                vectors = embed(
                    texts,
                    client=api_client,
                    model=resolved_model,
                )

                if len(vectors) != len(batch):
                    raise ValueError(
                        "Embedding API returned an "
                        "unexpected number of vectors"
                    )

                # --------------------------------------------
                # Store results.
                # --------------------------------------------

                for (
                    (chunk, key),
                    vector,
                ) in zip(
                    batch,
                    vectors,
                ):

                    saved = {
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "embedding": vector,
                    }

                    cached[key] = saved

                    records.append(
                        EmbeddedChunk(
                            text=chunk.text,
                            metadata=chunk.metadata,
                            embedding=vector,
                        )
                    )

                # --------------------------------------------
                # Save cache after every successful batch.
                # --------------------------------------------

                _write_cache(
                    cache_file,
                    cached,
                )

                generated += len(batch)

                break

            except Exception as error:

                if (
                    not _is_transient(error)
                    or attempt >= max_retries
                ):

                    failures.append(
                        f"batch {batch_number} "
                        f"({len(batch)} chunks): {error}"
                    )

                    break

                attempt += 1

                retries += 1

                delay = (
                    backoff_seconds
                    * (2 ** (attempt - 1))
                )

                sleep(delay)

    # --------------------------------------------------------
    # Cost estimation.
    # --------------------------------------------------------

    input_tokens = sum(
        token_count(chunk.text)
        for chunk, _ in pending
    )

    if cost_per_1k_tokens is None:

        cost_per_1k_tokens = float(
            os.getenv(
                "EMBEDDING_COST_PER_1K",
                str(DEFAULT_COST_PER_1K),
            )
        )

    estimated_cost = (
        input_tokens / 1000
    ) * cost_per_1k_tokens

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------

    summary = BatchEmbeddingSummary(
        total_chunks=len(chunks),
        embeddings_generated=generated,
        skipped_chunks=skipped,
        failures=failures,
        batches=(
            len(pending) + batch_size - 1
        ) // batch_size,
        retries=retries,
        input_tokens=input_tokens,
        estimated_cost=estimated_cost,
    )

    return records, summary


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a: list[float],
    b: list[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.
    """

    if not a or not b:
        return 0.0

    if len(a) != len(b):
        raise ValueError(
            "Vectors must have the same dimension"
        )

    dot_product = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = sum(
        x * x
        for x in a
    ) ** 0.5

    norm_b = sum(
        x * x
        for x in b
    ) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (
        norm_a * norm_b
    )


# ============================================================
# RANK CHUNKS
# ============================================================

def rank_chunks(
    query_embedding: list[float],
    chunks: list,
    top_k: int = 5,
) -> list:
    """
    Rank chunks according to cosine similarity.

    This function is required by reranking.py.

    Args:
        query_embedding:
            Embedding of the query.

        chunks:
            Chunks containing an `embedding` attribute or
            dictionary key.

        top_k:
            Number of chunks to return.

    Returns:
        Chunks ordered by descending similarity.
    """

    if not chunks:
        return []

    if top_k <= 0:
        return []

    scored = []

    for chunk in chunks:

        # Support dictionaries.
        if isinstance(chunk, dict):
            embedding = chunk.get(
                "embedding",
                chunk.get("vector"),
            )

        # Support EmbeddedChunk objects.
        else:
            embedding = getattr(
                chunk,
                "embedding",
                None,
            )

        if embedding is None:
            continue

        score = cosine_similarity(
            query_embedding,
            embedding,
        )

        scored.append(
            (
                score,
                chunk,
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        chunk
        for _, chunk in scored[:top_k]
    ]


# ============================================================
# TEST
# ============================================================

def test_embedding():
    """
    Simple embedding test.
    """

    print("=" * 60)
    print("EMBEDDING TEST")
    print("=" * 60)

    print(
        "Model:",
        _get_model(),
    )

    vectors = embed(
        [
            "Machine vibration inspection",
            "Preventive maintenance procedure",
        ]
    )

    print(
        "Vectors:",
        len(vectors),
    )

    print(
        "Dimensions:",
        [
            len(vector)
            for vector in vectors
        ],
    )

    print(
        "First 5 values:",
        vectors[0][:5],
    )

    print("=" * 60)
    print("Embedding test passed")
    print("=" * 60)


if __name__ == "__main__":
    test_embedding()