"""Chunk re-ranking experiment."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

try:
    from .embeddings import embed, rank_chunks
    from .ingestion import ingest
    from .embeddings import batch_embed_chunks
except ImportError:
    from embeddings import embed, rank_chunks
    from ingestion import ingest
    from embeddings import batch_embed_chunks


CANDIDATE_K = 10
FINAL_K = 3


def normalize(text: str) -> str:
    """Normalize text for keyword matching."""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def get_query_terms(query: str) -> list[str]:
    """Extract meaningful query terms."""
    stop_words = {
        "what",
        "is",
        "the",
        "a",
        "an",
        "are",
        "to",
        "of",
        "and",
        "in",
        "on",
        "for",
        "do",
        "does",
        "should",
        "can",
        "i",
        "before",
        "if",
    }

    return [
        word
        for word in normalize(query).split()
        if word not in stop_words and len(word) > 2
    ]


def keyword_score(query: str, text: str) -> float:
    """Measure how many query terms appear in the chunk."""
    terms = get_query_terms(query)

    if not terms:
        return 0.0

    words = set(normalize(text).split())

    matches = sum(1 for term in terms if term in words)

    return matches / len(terms)


def phrase_score(query: str, text: str) -> float:
    """Give an additional score when related phrases appear."""
    query_normalized = normalize(query)
    text_normalized = normalize(text)

    if query_normalized in text_normalized:
        return 1.0

    terms = get_query_terms(query)

    if len(terms) < 2:
        return 0.0

    matched = 0

    for index in range(len(terms) - 1):
        phrase = f"{terms[index]} {terms[index + 1]}"

        if phrase in text_normalized:
            matched += 1

    return matched / (len(terms) - 1)


def rerank_score(query: str, similarity: float, text: str):
    """Calculate a second-stage relevance score."""

    keywords = keyword_score(query, text)
    phrases = phrase_score(query, text)

    score = (
        0.70 * similarity
        + 0.20 * keywords
        + 0.10 * phrases
    )

    return score, keywords, phrases


def print_initial_candidates(candidates):
    """Print the initial vector retrieval ordering."""

    print("\n" + "=" * 80)
    print("INITIAL RETRIEVAL - TOP 10 CANDIDATES")
    print("=" * 80)

    for rank, (similarity, record) in enumerate(candidates, start=1):
        print(f"\nRank: {rank}")
        print(f"Vector score: {similarity:.6f}")
        print(f"Source: {record.metadata['source']}")
        print(f"Section: {record.metadata['section']}")
        print(f"Chunk: {record.metadata['chunk_index']}")
        print(f"Text: {record.text.replace(chr(10), ' ')[:200]}")


def rerank_candidates(query: str, candidates):
    """Apply the second-stage ranking."""

    results = []

    for original_rank, (similarity, record) in enumerate(
        candidates,
        start=1,
    ):
        score, keywords, phrases = rerank_score(
            query,
            similarity,
            record.text,
        )

        results.append(
            {
                "original_rank": original_rank,
                "vector_score": similarity,
                "keyword_score": keywords,
                "phrase_score": phrases,
                "rerank_score": score,
                "source": record.metadata["source"],
                "section": record.metadata["section"],
                "chunk_index": record.metadata["chunk_index"],
                "text": record.text,
                "metadata": record.metadata,
            }
        )

    results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    for rank, item in enumerate(results, start=1):
        item["rerank_rank"] = rank

    return results


def print_reranked_results(results):
    """Print the final re-ranked ordering."""

    print("\n" + "=" * 80)
    print("AFTER RE-RANKING")
    print("=" * 80)

    for item in results:
        print(f"\nFinal rank: {item['rerank_rank']}")
        print(f"Original rank: {item['original_rank']}")
        print(f"Vector score: {item['vector_score']:.6f}")
        print(f"Keyword score: {item['keyword_score']:.6f}")
        print(f"Phrase score: {item['phrase_score']:.6f}")
        print(f"Re-rank score: {item['rerank_score']:.6f}")
        print(f"Source: {item['source']}")
        print(f"Section: {item['section']}")
        print(f"Chunk: {item['chunk_index']}")
        print(f"Text: {item['text'].replace(chr(10), ' ')[:200]}")


def write_report(
    query: str,
    candidates,
    reranked,
    output_path: Path,
):
    """Save before-and-after results as Markdown."""

    lines = [
        "# Chunk Re-Ranking Experiment",
        "",
        f"**Query:** {query}",
        "",
        f"**Candidate set:** {len(candidates)}",
        "",
        f"**Final k:** {FINAL_K}",
        "",
        "## Before Re-Ranking",
        "",
        "| Rank | Vector Score | Source | Section | Chunk |",
        "| ---: | ---: | --- | --- | ---: |",
    ]

    for rank, (similarity, record) in enumerate(candidates, start=1):
        lines.append(
            f"| {rank} | {similarity:.6f} | "
            f"{record.metadata['source']} | "
            f"{record.metadata['section']} | "
            f"{record.metadata['chunk_index']} |"
        )

    lines.extend(
        [
            "",
            "## After Re-Ranking",
            "",
            "| Final Rank | Original Rank | Vector Score | "
            "Keyword Score | Phrase Score | Re-Rank Score | Source |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for item in reranked:
        lines.append(
            f"| {item['rerank_rank']} | "
            f"{item['original_rank']} | "
            f"{item['vector_score']:.6f} | "
            f"{item['keyword_score']:.6f} | "
            f"{item['phrase_score']:.6f} | "
            f"{item['rerank_score']:.6f} | "
            f"{item['source']} |"
        )

    lines.extend(
        [
            "",
            "## Final Selected Chunks",
            "",
        ]
    )

    for item in reranked[:FINAL_K]:
        lines.extend(
            [
                f"### Rank {item['rerank_rank']}",
                "",
                f"**Source:** {item['source']}",
                "",
                f"**Section:** {item['section']}",
                "",
                f"**Original rank:** {item['original_rank']}",
                "",
                f"**Re-rank score:** {item['rerank_score']:.6f}",
                "",
                item["text"].replace("\n", " "),
                "",
            ]
        )

    lines.extend(
        [
            "## Trade-off",
            "",
            "Initial vector retrieval is efficient because it searches "
            "the embedding space.",
            "",
            "Re-ranking adds a second scoring stage over only the "
            "candidate set. This can improve precision, but it adds "
            "computation and latency.",
            "",
            "A practical approach is to retrieve a larger candidate set "
            "and then keep only the highest-scoring final chunks.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    load_dotenv()

    query = "What should a technician do if abnormal vibration is detected?"

    print("=" * 80)
    print("CHUNK RE-RANKING FOR PRECISION")
    print("=" * 80)

    print(f"\nQuery: {query}")
    print(f"Candidate k: {CANDIDATE_K}")
    print(f"Final k: {FINAL_K}")

    data_dir = Path("data").resolve()

    _, _, chunks, failures = ingest(data_dir)

    model = (
        os.getenv("EMBEDDING_MODEL")
        or os.getenv("EMBED_MODEL")
    )

    records, summary = batch_embed_chunks(
        chunks,
        model=model,
        batch_size=1,
        cache_path="outputs/embedding_cache.json",
        max_retries=0,
    )

    if not records:
        raise SystemExit("No embedded records available.")

    print(f"\nEmbedded records available: {len(records)}")

    query_vector = embed(
        [query],
        model=model,
    )[0]

    ranked = rank_chunks(
        query_vector,
        records,
        top_k=CANDIDATE_K,
    )

    print_initial_candidates(ranked)

    reranked = rerank_candidates(
        query,
        ranked,
    )

    print_reranked_results(reranked)

    final_results = reranked[:FINAL_K]

    print("\n" + "=" * 80)
    print("FINAL TOP 3")
    print("=" * 80)

    for item in final_results:
        print(
            f"\n{item['rerank_rank']}. "
            f"{item['source']} | "
            f"re-rank score={item['rerank_score']:.6f}"
        )

    output_path = Path(
        "outputs/reranking_results.md"
    )

    write_report(
        query,
        ranked,
        reranked,
        output_path,
    )

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()