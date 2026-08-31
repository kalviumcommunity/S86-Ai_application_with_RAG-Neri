# Indexing Embeddings & Metadata Storage

This guide covers the corpus indexing workflow for your RAG application, implementing the concepts from the indexing assignment.

## What Is Indexing?

Indexing is the process of inserting your prepared embeddings into the vector database **with their source text and metadata attached**. This enables:
- ✓ Fast semantic search across the corpus
- ✓ Citation tracking (knowing where each result came from)
- ✓ Metadata filtering (by source, section, date, etc.)
- ✓ Integrity verification (spot-checking stored records)

## Architecture

```
Embedded Chunks (text + metadata + embeddings)
           ↓
    [indexing.py]
           ↓
    [1] Prepare records (stable IDs)
    [2] Bulk insert in batches
    [3] Verify indexed count
    [4] Spot-check integrity
           ↓
    Vector Store (persistent)
```

## Record Schema

Each indexed record follows this structure:

```python
{
    "id": "source:chunk_index",           # Stable ID for tracking
    "vector": [0.1, 0.2, ...],           # 1536-dim embedding
    "text": "Original chunk text...",     # Source text
    "metadata": {
        "source": "document.md",         # Source filename
        "chunk_index": 0,                # Position in document
        "section": "Introduction"        # Optional section header
    }
}
```

## Quick Start Commands

### 1. Index Your Corpus

Full workflow: ingest → embed → index → verify

```bash
cd src
python index_corpus.py --data-dir ../data
```

This runs:
- Ingests documents
- Generates embeddings (cached)
- Prepares vector records
- Bulk inserts in batches
- Verifies indexed count matches chunk count
- Spot-checks 3 random records

### 2. Index with Clear (Full Rebuild)

```bash
python index_corpus.py --data-dir ../data --clear
```

Clears existing index and rebuilds from scratch.

### 3. Verify Existing Index

Check what's already indexed:

```bash
python index_corpus.py --verify-only
```

Shows total chunk count and a sample record.

### 4. Test Indexing Module

Run unit tests with sample data:

```bash
python indexing.py
```

Expected output:
```
✓ 5 records prepared
✓ Batch inserted
✓ Validation PASSED
✓ 3/3 spot checks passed
```

## Key Functions

### `index_embeddings(embedded_chunks, vector_store, batch_size=100)`

**Bulk insert with validation**

```python
from indexing import index_embeddings
from vector_store import VectorStore

store = VectorStore()
result = index_embeddings(embedded_chunks, store)

print(f"Inserted: {result.inserted}")
print(f"Validation: {'PASSED' if result.validation_passed else 'FAILED'}")
```

Returns:
- `IndexingResult` with counts, failures, indexed_count, validation status
- Performs 4 steps:
  1. Prepare records with stable IDs
  2. Bulk insert in batches
  3. Verify indexed count = chunk count
  4. Spot-check sample records

### `spot_check_records(embedded_chunks, vector_store, sample_size=3)`

**Verify stored records match originals**

```python
from indexing import spot_check_records

checks = spot_check_records(embedded_chunks, store)

for check in checks:
    if check["passed"]:
        print(f"✓ {check['id']}")
    else:
        print(f"✗ {check['id']}: {check['error']}")
```

For each sampled record, verifies:
- ✓ Text matches original
- ✓ Metadata matches
- ✓ Vector dimension correct

### `reindex_corpus(embedded_chunks, vector_store)`

**Re-index after corpus changes**

```python
from indexing import reindex_corpus

# When documents change
result = reindex_corpus(new_embedded_chunks, store)
```

In production, this would implement incremental updates:
- Track document versions
- Delete removed chunks
- Update changed chunks
- Leave unchanged chunks alone

Currently does full rebuild for correctness.

## Validation & Verification

### Count Verification

After indexing, automatically checks:

```
Expected chunks: 10
Inserted this run: 10
Indexed count: 10
Validation: PASSED ✓
```

**Assertion**: `indexed_count >= expected_count`

If this fails, investigate:
- API errors during insertion
- Batch processing failures
- Database connection issues

### Spot-Check Results

Samples 3 records uniformly across corpus:

```
✓ document.md:0 (Text match, Metadata match, Vector dim match)
✓ document.md:5
✓ document.md:9
```

Each spot check verifies:
- `text_match`: Stored text = original text
- `metadata_match`: All metadata fields present and correct
- `vector_dim_match`: Vector has correct dimension (1536)

If a spot check fails:
- Record was corrupted during storage
- Metadata wasn't preserved
- Database had retrieval issues

## Workflow Output

Complete indexing run shows:

```
INDEXING EMBEDDINGS & METADATA STORAGE
======================================================================

[1/4] Preparing records for indexing...
  ✓ 100 records prepared
  Sample ID: guide.md:0
  Metadata keys: ['source', 'chunk_index', 'section']

[2/4] Bulk inserting in batches...
  Batch 1: 100 records
  Batch 2: 50 records
  ✓ Inserted this run: 150

[3/4] Verifying indexed count...
  Expected chunks: 150
  Indexed count: 150
  ✓ Validation PASSED

[4/4] Spot-checking stored records...
  ✓ 3/3 spot checks passed
```

## Metadata Filtering (Advanced)

With metadata stored, you can filter retrieval:

```python
# Retrieve only from specific source
results = store.search(
    query_embedding,
    where={"metadata": {"source": "guide.md"}}
)

# Or filter by section
results = store.search(
    query_embedding,
    where={"metadata": {"section": "Troubleshooting"}}
)
```

## Troubleshooting

**"Indexed count does not match chunk count"**
- Check for errors in batch insertion details
- Verify API connectivity during embedding
- Try clearing and re-running: `python index_corpus.py --clear`

**"Spot checks failed"**
- Indicates data corruption or retrieval issue
- Check vector database logs
- Consider deleting `outputs/chroma_db/` and starting fresh

**"No chunks created"**
- Ensure documents exist in data directory
- Check document format (TXT, MD, etc.)
- Verify `ingestion.py` can read files

**"Embedding API timeout"**
- Reduce batch size: `--batch-size 50`
- Check OPENAI_API_KEY and OPENAI_BASE_URL in .env

## Next Steps

1. **Index your corpus**: `python index_corpus.py --data-dir ../data --clear`
2. **Verify success**: Check spot checks all passed
3. **Search indexed data**: Use `vector_store.search()` with query embeddings
4. **Monitor performance**: Track indexing time and success rate
5. **Incremental updates**: When documents change, re-index only changed chunks

## File Structure

```
src/
├── indexing.py         ← Bulk insert, verify, spot-check
├── index_corpus.py     ← Full workflow script
├── vector_store.py     ← Vector database interface
├── embeddings.py       ← Embedding generation
├── ingestion.py        ← Document ingestion
└── ...

outputs/
├── chroma_db/          ← Persistent indexed data
├── embedding_cache.json
└── ...
```

## References

- Qdrant Points: https://qdrant.tech/documentation/concepts/points/
- Chroma Add Data: https://docs.trychroma.com/docs/collections/add-data
- Pinecone Upsert: https://docs.pinecone.io/guides/data/upsert-data
