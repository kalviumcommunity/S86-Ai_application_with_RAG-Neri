"""Complete RAG pipeline: ingest, retrieve, assemble, and generate."""

import argparse
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

try:
    from .chunking import token_count
    from .embeddings import embed, batch_embed_chunks
    from .ingestion import ingest, validate_ingestion
    from .vector_store import VectorStore
except ImportError:
    from chunking import token_count
    from embeddings import embed, batch_embed_chunks
    from ingestion import ingest, validate_ingestion
    from vector_store import VectorStore


NO_CONTEXT_ANSWER = "I could not find relevant context for that question."
GROUNDING_INSTRUCTIONS = (
    "Answer only from the provided context. Cite evidence with its [number]. "
    "If the context is insufficient, say what information is missing."
)


def embed_query(query: str, embedder: Callable[[list[str]], list[list[float]]] = embed) -> list[float]:
    """Embed one user query using the same model used for document chunks."""
    if not query.strip():
        raise ValueError("query must not be empty")
    vectors = embedder([query])
    if not vectors:
        raise ValueError("the embedder returned no vector")
    return vectors[0]


def retrieve_context(
    query_vector: list[float],
    vector_store: VectorStore,
    k: int = 4,
) -> list[dict]:
    """Retrieve the top-k evidence chunks for a query vector."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
    return vector_store.search(query_vector, top_k=k)


def _format_chunk(index: int, chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "unknown")
    section = metadata.get("section")
    label = f"{source} ({section})" if section else source
    return f"[{index}] Source: {label}\n{chunk.get('text', '')}"


def assemble_context(chunks: list[dict], max_tokens: int | None = None) -> str:
    """Format retrieved chunks with stable citation numbers for the prompt."""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        candidate = _format_chunk(index, chunk)
        proposed = "\n\n".join(parts + [candidate])
        if max_tokens is not None and token_count(proposed) > max_tokens:
            break
        parts.append(candidate)
    return "\n\n".join(parts)


def build_augmented_prompt(
    query: str,
    chunks: list[dict],
    model_token_budget: int = 2048,
    answer_token_reserve: int = 256,
) -> dict:
    """Build a grounded prompt while reserving tokens for the answer."""
    if model_token_budget <= 0 or answer_token_reserve < 0:
        raise ValueError("token budget must be positive and answer reserve non-negative")

    question_header = f"Question: {query}"
    fixed_prompt = f"{GROUNDING_INSTRUCTIONS}\n\nContext:\n\nQuestion:"
    available_context_tokens = (
        model_token_budget
        - answer_token_reserve
        - token_count(fixed_prompt)
        - token_count(question_header.removeprefix("Question: "))
    )
    if available_context_tokens < 0:
        raise ValueError("token budget is too small for instructions and question")

    context = assemble_context(chunks, max_tokens=available_context_tokens)
    prompt = f"{GROUNDING_INSTRUCTIONS}\n\nContext:\n{context}\n\n{question_header}"
    return {
        "prompt": prompt,
        "context": context,
        "included_chunks": sum(
            1 for line in context.splitlines() if line.startswith("[") and "] Source:" in line
        ),
        "prompt_tokens": token_count(prompt),
        "answer_token_reserve": answer_token_reserve,
        "model_token_budget": model_token_budget,
        "total_reserved_tokens": token_count(prompt) + answer_token_reserve,
    }


def generate_answer(
    query: str,
    context: str,
    client=None,
    model: str | None = None,
) -> str:
    """Generate an answer grounded only in the assembled context."""
    if client is None:
        from openai import OpenAI

        load_dotenv()
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not base_url or not api_key:
            raise ValueError("OPENAI_BASE_URL and OPENAI_API_KEY are required")
        client = OpenAI(base_url=base_url, api_key=api_key)

    chat_model = model or os.getenv("CHAT_MODEL")
    if not chat_model:
        raise ValueError("CHAT_MODEL is missing from .env")

    response = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    GROUNDING_INSTRUCTIONS
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
    )
    return response.choices[0].message.content or ""


def answer_query(
    query: str,
    vector_store: VectorStore,
    k: int = 4,
    embedder: Callable[[list[str]], list[list[float]]] = embed,
    generator: Callable[[str, str], str] = generate_answer,
) -> dict:
    """Run query embedding, retrieval, context assembly, and generation."""
    query_vector = embed_query(query, embedder)
    chunks = retrieve_context(query_vector, vector_store, k=k)
    if not chunks:
        return {"answer": NO_CONTEXT_ANSWER, "sources": []}

    prompt = build_augmented_prompt(query, chunks)
    answer = generator(query, prompt["context"])
    return {
        "answer": answer,
        "sources": [chunk.get("metadata", {}) for chunk in chunks],
    }


def build_rag_index(
    data_dir: str | Path,
    vector_store: VectorStore,
    token_size: int = 64,
    token_overlap: int = 16,
    batch_size: int = 100,
    clear_existing: bool = False,
) -> dict:
    """Build complete vector index from documents.

    Args:
        data_dir: Directory containing documents to ingest
        vector_store: Initialized VectorStore instance
        token_size: Tokens per chunk
        token_overlap: Overlap between chunks
        batch_size: Batch size for embedding API
        clear_existing: Whether to clear vector store before rebuild

    Returns:
        Summary dict with pipeline statistics
    """
    print("=" * 60)
    print("RAG PIPELINE: INGEST → EMBED → STORE")
    print("=" * 60)

    # Clear if requested
    if clear_existing:
        print("Clearing existing vector store...")
        vector_store.clear()

    # Step 1: Ingest and chunk documents
    print("\n[1/3] Ingesting documents...")
    files, documents, chunks, failures = ingest(data_dir, token_size, token_overlap)
    validate_ingestion(files, documents, failures)

    print(f"  Files discovered: {len(files)}")
    print(f"  Documents ingested: {documents}")
    print(f"  Chunks created: {len(chunks)}")
    if failures:
        print(f"  Failures: {len(failures)}")
        for path, error in failures:
            print(f"    - {path}: {error}")

    # Step 2: Generate embeddings
    print("\n[2/3] Generating embeddings...")
    embedded_chunks, summary = batch_embed_chunks(
        chunks,
        batch_size=batch_size,
        cache_path="outputs/embedding_cache.json",
    )

    print(f"  Embeddings generated: {summary.embeddings_generated}")
    print(f"  Cached embeddings reused: {summary.skipped_chunks}")
    print(f"  Total embeddings: {len(embedded_chunks)}")
    print(f"  Batches processed: {summary.batches}")
    if summary.failures:
        print(f"  Failures: {len(summary.failures)}")
        for failure in summary.failures:
            print(f"    - {failure}")
    print(f"  Estimated cost: ${summary.estimated_cost:.4f}")

    # Step 3: Store in vector database
    print("\n[3/3] Storing in vector database...")
    store_result = vector_store.upsert(embedded_chunks)
    print(f"  Upserted: {store_result['inserted']}")
    if store_result.get('error'):
        print(f"  Error: {store_result['error']}")

    total_in_store = vector_store.count()
    print(f"  Total chunks in store: {total_in_store}")

    print("\n" + "=" * 60)
    print("RAG PIPELINE COMPLETE")
    print("=" * 60)

    return {
        "files": len(files),
        "documents": documents,
        "chunks": len(chunks),
        "embeddings_generated": summary.embeddings_generated,
        "embeddings_cached": summary.skipped_chunks,
        "vector_store_total": total_in_store,
        "failures": len(failures) + len(summary.failures),
    }


def retrieve_similar(
    query_text: str,
    query_embedding: list[float],
    vector_store: VectorStore,
    top_k: int = 5,
) -> None:
    """Retrieve and display similar chunks."""
    print("\n" + "=" * 60)
    print(f"SEMANTIC SEARCH: {query_text}")
    print("=" * 60)

    results = vector_store.search(query_embedding, top_k=top_k)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, 1):
        similarity = result['similarity']
        print(f"\n[{rank}] Similarity: {similarity:.4f}")
        print(f"    ID: {result['id']}")
        print(f"    Source: {result['metadata'].get('source', 'unknown')}")
        print(f"    Section: {result['metadata'].get('section', 'N/A')}")
        print(f"    Text: {result['text'][:80]}...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Directory with documents")
    parser.add_argument("--token-size", type=int, default=64, help="Tokens per chunk")
    parser.add_argument("--token-overlap", type=int, default=16, help="Chunk overlap")
    parser.add_argument("--batch-size", type=int, default=100, help="Embedding batch size")
    parser.add_argument("--vector-dir", default="outputs/chroma_db", help="Vector store directory")
    parser.add_argument("--clear", action="store_true", help="Clear vector store before rebuild")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    # Initialize vector store
    vector_store = VectorStore(persist_dir=args.vector_dir)

    # Build RAG index
    summary = build_rag_index(
        data_dir,
        vector_store,
        token_size=args.token_size,
        token_overlap=args.token_overlap,
        batch_size=args.batch_size,
        clear_existing=args.clear,
    )


if __name__ == "__main__":
    main()
