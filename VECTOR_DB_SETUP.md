# Vector Database Setup & Integration Guide

This guide covers the vector database implementation for the RAG application, following the design principles from the vector database concept.

## Architecture Overview

Your RAG pipeline now consists of three integrated layers:

```
Documents
    ↓
[ingestion.py] → Chunks with metadata
    ↓
[embeddings.py] → Embedded chunks (text + metadata + vectors)
    ↓
[vector_store.py] → Persistent vector collection
    ↓
[rag_pipeline.py] → Search and retrieval
```

## Collection Design

The implementation follows the standard schema for RAG:

```python
{
    "id": "stable chunk id",
    "vector": "1536-dim embedding from OpenAI",
    "text": "original chunk text",
    "metadata": {
        "source": "document filename",
        "chunk_index": "position in document",
        "section": "optional section heading"
    }
}
```

**Key properties:**
- **Dimension**: 1536 (OpenAI text-embedding-3-small)
- **Metric**: cosine similarity
- **Persistence**: DuckDB + Parquet in `outputs/chroma_db/`
- **Text & metadata stored**: Enables grounded retrieval, not just anonymous vectors

## Quick Start

### 1. Test the Vector Store

Verify basic insert/read-back functionality:

```bash
cd src
python vector_store.py
```

Expected output:
```
✓ Connected to vector database
✓ Inserted test record
✓ Read back verification
✓ Collection now contains: 1 chunk(s)
```

### 2. Build a RAG Index from Your Documents

Ingest, embed, and store all documents:

```bash
cd src
python rag_pipeline.py --data-dir ../data
```

This runs the full pipeline:
- **Step 1**: Ingests documents from `data/`, chunks them (64 tokens, 16 token overlap)
- **Step 2**: Generates embeddings (cached for efficiency)
- **Step 3**: Stores embeddings in vector database

Output shows:
- Files discovered and processed
- Chunks created
- Embeddings generated vs. cached
- Total cost estimate

### 3. Use the Vector Store in Code

```python
from vector_store import VectorStore
from embeddings import embed

# Initialize
store = VectorStore(persist_dir="outputs/chroma_db")

# Add embedded chunks
store.upsert(embedded_chunks_list)

# Search for similar chunks
query_embedding = embed(["Your query here"])[0]
results = store.search(query_embedding, top_k=5)

for result in results:
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Text: {result['text']}")
    print(f"Source: {result['metadata']['source']}")
```

## Key Methods

### VectorStore Class

#### `upsert(embedded_chunks: list[EmbeddedChunk]) -> dict`
Insert or update chunks in the vector store.
- Takes `EmbeddedChunk` objects (text, metadata, embedding)
- Automatically generates stable IDs from source + chunk_index
- Returns summary of inserted/updated/failed counts

#### `search(query_embedding: list[float], top_k: int = 5, where: dict = None) -> list[dict]`
Retrieve semantically similar chunks.
- Input: Query embedding vector (must be 1536-dimensional)
- Output: List of results ordered by cosine similarity
- Each result includes: id, text, metadata, similarity score
- Optional metadata filtering with `where` parameter

#### `get(chunk_id: str) -> Optional[dict]`
Retrieve a specific chunk by ID.
- Useful for follow-up operations after search
- Returns full chunk data if found

#### `count() -> int`
Get total number of chunks in collection.

#### `clear() -> None`
Delete all chunks (useful for full rebuilds).

## Configuration

Edit in `vector_store.py`:

```python
VECTOR_DIMENSION = 1536  # Match your embedding model
COLLECTION_NAME = "rag_chunks"  # Collection name
SIMILARITY_METRIC = "cosine"  # Vector similarity metric
```

## Persistence

Vector data is stored in `outputs/chroma_db/`:
- Uses DuckDB + Parquet for efficient storage
- Persistent across sessions
- Survives application restarts
- Can be backed up as a directory

## Troubleshooting

**"Vector dimension mismatch"**
- Ensure your embedding model outputs 1536 dimensions
- Check EMBED_MODEL in `.env`

**"No results found"**
- Verify chunks were inserted successfully
- Check that query embedding has correct dimension
- Try increasing `top_k` parameter

**"Chroma connection failed"**
- Ensure `outputs/chroma_db/` directory is writable
- Check disk space
- Delete `outputs/chroma_db/` to reset (loses data)

## Next Steps

1. **Add retrieval to LLM chain**: Use top-k results as context for generation
2. **Implement re-ranking**: Score results with a cross-encoder model
3. **Add filtering**: Use metadata filters for document-specific retrieval
4. **Monitor performance**: Track query latency and relevance metrics
5. **Tune chunking**: Experiment with chunk size and overlap for your domain

## References

- Chroma Documentation: https://docs.trychroma.com/
- Vector Database Concepts: https://qdrant.tech/documentation/concepts/collections/
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
