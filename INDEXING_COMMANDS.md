# Indexing & Metadata Storage - Commands Reference

## Quick Commands

```powershell
# Navigate to project
cd "c:\Users\Hasini reddy\Desktop\p1\S86-Ai_application_with_RAG-Neri"

# Test indexing module with sample data
python src/indexing.py

# Full corpus indexing workflow
python src/index_corpus.py --data-dir data

# Rebuild index (clear existing)
python src/index_corpus.py --data-dir data --clear

# Verify existing index
python src/index_corpus.py --verify-only

# Check indexed count
python -c "from src.vector_store import VectorStore; print(f'Indexed chunks: {VectorStore().count()}')"
```

## Step-by-Step Workflow

### 1. Prepare Your Documents
Ensure documents are in the `data/` folder:
```
data/
├── machine_manual.txt
├── maintenance_log.txt
└── safety_procedure.md
```

### 2. Test Indexing Setup
Verify the indexing module works with sample data:
```powershell
python src/indexing.py
```

Expected output:
```
✓ 5 records prepared
✓ Batch 1: 5 records
✓ Validation PASSED
✓ 3/3 spot checks passed
```

### 3. Index Your Corpus
Full workflow: ingest → embed → prepare → insert → verify:
```powershell
python src/index_corpus.py --data-dir data
```

Shows:
- Documents ingested and chunked
- Embeddings generated
- Records prepared with stable IDs
- Batch insertion progress
- Count verification (expected vs. actual)
- Spot-check results

### 4. Index with Full Rebuild
Clear existing index and start fresh:
```powershell
python src/index_corpus.py --data-dir data --clear
```

### 5. Verify Indexed Data
Check what's in the database:
```powershell
python src/index_corpus.py --verify-only
```

Shows:
- Total chunks in collection
- Sample record preview
- Source and metadata

## Detailed Commands

### Indexing Commands

```powershell
# Full workflow from documents to index
python src/index_corpus.py `
  --data-dir data `
  --vector-dir outputs/chroma_db

# Clear and rebuild
python src/index_corpus.py `
  --data-dir data `
  --clear

# Only verify existing index (no re-indexing)
python src/index_corpus.py --verify-only

# Custom vector database location
python src/index_corpus.py `
  --data-dir data `
  --vector-dir custom/path/to/db
```

### Python API Commands

```powershell
# Test indexing with sample data
python src/indexing.py

# Check total indexed chunks
python -c "from src.vector_store import VectorStore; s = VectorStore(); print(f'Total indexed: {s.count()}')"

# Get specific chunk
python -c "
from src.vector_store import VectorStore
s = VectorStore()
chunk = s.get('document.md:0')
if chunk:
    print(f'Found: {chunk[\"id\"]}')
    print(f'Text: {chunk[\"text\"][:80]}')
"

# List all chunks (first 5)
python -c "
from src.vector_store import VectorStore
s = VectorStore()
results = s.collection.get(limit=5, include=['documents', 'metadatas'])
for i, (doc_id, text) in enumerate(zip(results['ids'], results['documents']), 1):
    print(f'{i}. {doc_id}: {text[:60]}...')
"
```

### Monitoring Commands

```powershell
# Monitor indexing progress (run in second terminal)
while($true) {
    python -c "from src.vector_store import VectorStore; print(Get-Date; VectorStore().count()); Start-Sleep -Seconds 5"
}

# Get indexing statistics
python -c "
from src.vector_store import VectorStore
s = VectorStore()
print(f'Collection: {s.collection_name}')
print(f'Total chunks: {s.count()}')
print(f'Vector dimension: {s.vector_dimension}')
"

# Test batch insertion performance
python -c "
import time
from src.vector_store import VectorStore
from src.embeddings import EmbeddedChunk

s = VectorStore()
chunks = [
    EmbeddedChunk('test', {'source': 'test.md', 'chunk_index': i}, [0.1]*1536)
    for i in range(100)
]

start = time.time()
s.upsert(chunks)
elapsed = time.time() - start
print(f'Inserted 100 chunks in {elapsed:.2f}s')
"
```

## Troubleshooting Commands

```powershell
# Test vector database connection
python -c "
from src.vector_store import VectorStore
try:
    s = VectorStore()
    print(f'✓ Connected to {s.collection_name}')
    print(f'  Chunks: {s.count()}')
except Exception as e:
    print(f'✗ Connection failed: {e}')
"

# Verify embeddings were cached
python -c "
import json
from pathlib import Path
cache_file = Path('outputs/embedding_cache.json')
if cache_file.exists():
    data = json.loads(cache_file.read_text())
    print(f'Cached embeddings: {len(data.get(\"records\", {}))}')
else:
    print('No cache file found')
"

# Clear vector store
python -c "
from src.vector_store import VectorStore
s = VectorStore()
s.clear()
print(f'Store cleared. Chunks now: {s.count()}')
"

# Rebuild from scratch
python -c "
from src.vector_store import VectorStore
s = VectorStore()
s.clear()
print(f'✓ Cleared')
print(f'✓ Ready for re-indexing')
"
```

## Advanced Workflows

### Re-index with Incremental Updates

```powershell
# Python script to handle document changes
$script = @'
from src.ingestion import ingest
from src.embeddings import batch_embed_chunks
from src.indexing import reindex_corpus
from src.vector_store import VectorStore

data_dir = "data"
vector_store = VectorStore()

# Ingest updated documents
_, _, chunks, _ = ingest(data_dir)

# Generate embeddings
embedded_chunks, _ = batch_embed_chunks(chunks)

# Re-index (will clear and rebuild in current implementation)
result = reindex_corpus(embedded_chunks, vector_store)
print(f"Re-indexed: {result['total']} records")
'@

$script | python
```

### Batch Indexing with Progress

```powershell
# Show progress during large indexing jobs
$script = @'
from src.indexing import batches, to_vector_record
from src.embeddings import batch_embed_chunks
from src.ingestion import ingest
from src.vector_store import VectorStore

data_dir = "data"
vector_store = VectorStore()
batch_size = 50

# Ingest and embed
_, _, chunks, _ = ingest(data_dir)
embedded_chunks, _ = batch_embed_chunks(chunks, batch_size=batch_size)

# Manual batch insertion with progress
records = [to_vector_record(c) for c in embedded_chunks]
total = len(records)

for batch_num, batch in enumerate(batches(records, batch_size), 1):
    # Convert to EmbeddedChunk format
    from src.embeddings import EmbeddedChunk
    batch_chunks = [
        EmbeddedChunk(r["text"], r["metadata"], r["vector"])
        for r in batch
    ]
    result = vector_store.upsert(batch_chunks)
    
    progress = (batch_num * batch_size) / total * 100
    print(f"[{progress:.0f}%] Batch {batch_num}: {result['inserted']} records")

print(f"Complete: {vector_store.count()} total chunks indexed")
'@

$script | python
```

## Integration Examples

### With LLM Retrieval

```powershell
# Use indexed data for retrieval-augmented generation
$script = @'
from src.vector_store import VectorStore
from src.embeddings import embed
from src.rag_example import retrieve_context, build_rag_prompt

# Initialize
store = VectorStore()

# Query
question = "How do I reset my password?"
context = retrieve_context(question, store, top_k=3)

# Build prompt
prompt = build_rag_prompt(question, context)
print(prompt)

# Send to LLM...
'@

$script | python
```

## File Locations

```
Project Root/
├── src/
│   ├── indexing.py              ← Core indexing logic
│   ├── index_corpus.py          ← Full workflow script
│   ├── vector_store.py
│   ├── embeddings.py
│   ├── ingestion.py
│   └── ...
│
├── data/                        ← Input documents
│   ├── machine_manual.txt
│   ├── maintenance_log.txt
│   └── safety_procedure.md
│
├── outputs/
│   ├── chroma_db/              ← Indexed data (persistent)
│   ├── embedding_cache.json    ← Cached embeddings
│   └── ...
│
├── INDEXING_GUIDE.md           ← Full documentation
└── VECTOR_DB_SETUP.md
```

## Common Patterns

```powershell
# Ingest and index in one go
python src/index_corpus.py --data-dir data --clear

# Check if indexing completed successfully
if ($LASTEXITCODE -eq 0) {
    echo "✓ Indexing successful"
    python src/index_corpus.py --verify-only
} else {
    echo "✗ Indexing failed"
}

# Export indexed chunks for inspection
python -c "
import json
from src.vector_store import VectorStore

s = VectorStore()
results = s.collection.get(include=['documents', 'metadatas', 'distances'])

chunks = []
for i, id_ in enumerate(results['ids']):
    chunks.append({
        'id': id_,
        'text': results['documents'][i] if results['documents'] else '',
        'metadata': results['metadatas'][i] if results['metadatas'] else {},
    })

with open('indexed_chunks.json', 'w') as f:
    json.dump(chunks, f, indent=2)
    
print(f'Exported {len(chunks)} chunks')
"
```

---

**Pro Tip**: Combine `--data-dir`, `--vector-dir`, and `--clear` flags based on your needs:
- Full rebuild: `--data-dir data --clear`
- Incremental: `--data-dir data` (keeps existing)
- Custom location: `--data-dir data --vector-dir custom/db`
