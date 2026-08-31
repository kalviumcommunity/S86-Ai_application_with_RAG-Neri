# All Assignments: Essential Commands

## Navigate to Project
```powershell
cd "c:\Users\Hasini reddy\Desktop\p1\S86-Ai_application_with_RAG-Neri"
```

---

## ASSIGNMENT 1: Vector Database Setup ✓

### Test Vector Store
```powershell
python src/vector_store.py
```
Expected: ✓ Connection | ✓ Insert | ✓ Read-back | ✓ Count

### Verify Vector Store
```powershell
python -c "from src.vector_store import VectorStore; print(f'✓ Store OK, {VectorStore().count()} chunks')"
```

---

## ASSIGNMENT 2: Indexing ✓

### Test Indexing Module
```powershell
python src/indexing.py
```
Expected: ✓ Prepare | ✓ Insert | ✓ Verify Count | ✓ Spot-check

### Index Your Corpus
```powershell
python src/index_corpus.py --data-dir data
```
Full workflow: ingest → embed → prepare → insert → verify

### Rebuild Index (Clear Existing)
```powershell
python src/index_corpus.py --data-dir data --clear
```

### Verify Indexed Data
```powershell
python src/index_corpus.py --verify-only
```

### Check Indexed Count
```powershell
python -c "from src.vector_store import VectorStore; print(f'{VectorStore().count()} chunks indexed')"
```

---

## ASSIGNMENT 3: Retrieval ✓

### Test Retrieval (No API Key Needed)
```powershell
python src/retrieval_test.py
```
Expected: ✓ Create corpus | ✓ Search | ✓ Compare k=1,3,5

### Run Basic Retrieval Demo
```powershell
python src/retrieval_demo.py --mode basic
```
Single query with top-3 results

### Compare k Values (1, 3, 5)
```powershell
python src/retrieval_demo.py --mode k-compare
```
See how results change with different k

### Build LLM Context
```powershell
python src/retrieval_demo.py --mode context
```
Shows formatted context and complete LLM prompt

### Interactive Search Mode
```powershell
python src/retrieval_demo.py --mode interactive
```
Try multiple queries with commands:
- `<query>` - Search with k=3
- `k=5 <query>` - Custom k value
- `compare 1,3,5` - Compare k values
- `help` - Show commands
- `quit` - Exit

### Performance Benchmarks
```powershell
python src/retrieval_demo.py --mode benchmark
```
Test latency across different k values

---

## Complete Workflow: All Three Assignments

### Step 1: Test All Modules
```powershell
python src/vector_store.py      # Test 1: Vector DB setup
python src/indexing.py           # Test 2: Indexing
python src/retrieval_test.py     # Test 3: Retrieval
```

### Step 2: Index Your Documents
```powershell
python src/index_corpus.py --data-dir data --clear
```

### Step 3: Test Retrieval with Real Data
```powershell
python src/retrieval_demo.py --mode basic
```

### Step 4: Explore and Tune
```powershell
# Compare k values
python src/retrieval_demo.py --mode k-compare

# Interactive testing
python src/retrieval_demo.py --mode interactive

# Check performance
python src/retrieval_demo.py --mode benchmark
```

---

## Python API Usage

### Quick Retrieval
```powershell
python -c "
from src.retrieval import retrieve
from src.vector_store import VectorStore

store = VectorStore()
response = retrieve('password reset', store, top_k=3)
for r in response.results:
    print(f'{r.rank}. {r.similarity:.4f} | {r.text[:60]}...')
"
```

### Build LLM Context
```powershell
python -c "
from src.retrieval import retrieve_with_context

context = retrieve_with_context('your query', top_k=3)
print(context)
"
```

### Get Quality Metrics
```powershell
python -c "
from src.retrieval import retrieve, analyze_retrieval_quality
from src.vector_store import VectorStore

response = retrieve('query', VectorStore(), top_k=5)
metrics = analyze_retrieval_quality(response)
print(f'Avg similarity: {metrics[\"avg_similarity\"]:.4f}')
"
```

### Compare k Values
```powershell
python -c "
from src.retrieval import compare_k_values
from src.vector_store import VectorStore

results = compare_k_values('query', [1, 3, 5, 10])
for k in sorted(results.keys()):
    print(f'k={k}: {results[k].total_retrieved} results')
"
```

---

## Useful Checks

### Check Vector Store Status
```powershell
python -c "
from src.vector_store import VectorStore
s = VectorStore()
print(f'Collection: {s.collection_name}')
print(f'Chunks: {s.count()}')
print(f'Dimension: {s.vector_dimension}')
"
```

### Check Embedding Cache
```powershell
python -c "
import json
from pathlib import Path
cache = Path('outputs/embedding_cache.json')
if cache.exists():
    data = json.loads(cache.read_text())
    print(f'Cached embeddings: {len(data.get(\"records\", {}))}')
"
```

### List Sample Chunks
```powershell
python -c "
from src.vector_store import VectorStore
s = VectorStore()
results = s.collection.get(include=['documents', 'metadatas'], limit=3)
for i, doc_id in enumerate(results['ids'], 1):
    text = results['documents'][i-1] if results['documents'] else ''
    print(f'{i}. {doc_id}: {text[:60]}...')
"
```

### Export Retrieval Results
```powershell
python -c "
import json
from src.retrieval import retrieve
from src.vector_store import VectorStore

response = retrieve('query', VectorStore(), top_k=3)
results = [
    {'rank': r.rank, 'similarity': r.similarity, 'text': r.text, 'source': r.metadata.get('source')}
    for r in response.results
]
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Exported to results.json')
"
```

---

## Troubleshooting

### Vector Store is Empty
```powershell
# Check count
python -c "from src.vector_store import VectorStore; print(VectorStore().count())"

# Index if empty
python src/index_corpus.py --data-dir data --clear
```

### No Results from Retrieval
```powershell
# Verify corpus is indexed
python src/index_corpus.py --verify-only

# Try larger k
python src/retrieval_demo.py --mode basic  # Uses k=3
```

### API Key Issues
```powershell
# Test without API (uses dummy embeddings)
python src/retrieval_test.py

# For real queries, set .env:
# OPENAI_API_KEY=your_key
# OPENAI_BASE_URL=https://api.openai.com/v1
```

### Slow Queries
```powershell
# Check corpus size
python -c "from src.vector_store import VectorStore; print(f'Size: {VectorStore().count()} chunks')"

# Run benchmark
python src/retrieval_demo.py --mode benchmark
```

---

## File Locations

```
S86-Ai_application_with_RAG-Neri/
├── COMPLETE_PIPELINE.md
├── VECTOR_DB_SETUP.md
├── INDEXING_GUIDE.md
├── RETRIEVAL_GUIDE.md
├── INDEXING_COMMANDS.md
├── RETRIEVAL_COMMANDS.md
│
├── src/
│   ├── vector_store.py
│   ├── indexing.py
│   ├── index_corpus.py
│   ├── retrieval.py
│   ├── retrieval_demo.py
│   ├── retrieval_test.py
│   └── ...
│
├── data/
│   ├── machine_manual.txt
│   ├── maintenance_log.txt
│   └── safety_procedure.md
│
└── outputs/
    ├── chroma_db/
    └── embedding_cache.json
```

---

## Summary

✅ **Assignment 1: Vector Database Setup**
- Command: `python src/vector_store.py`
- Status: Connected, insert/read verified

✅ **Assignment 2: Indexing**
- Command: `python src/index_corpus.py --data-dir data --clear`
- Status: Indexed, verified, spot-checked

✅ **Assignment 3: Retrieval**
- Command: `python src/retrieval_demo.py --mode basic`
- Status: Search working, k-tunable, LLM-ready

---

**All assignments complete! RAG pipeline ready for production.** 🚀
