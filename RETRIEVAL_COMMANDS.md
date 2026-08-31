# Similarity Search & Top-K Retrieval - Commands Reference

## Quick Start Commands

```powershell
# Navigate to project
cd "c:\Users\Hasini reddy\Desktop\p1\S86-Ai_application_with_RAG-Neri"

# Test retrieval with sample data (no API key needed)
python src/retrieval_test.py

# Run interactive retrieval demo (requires indexed corpus)
python src/retrieval_demo.py --mode basic

# Compare different k values
python src/retrieval_demo.py --mode k-compare

# Build LLM context from retrieval
python src/retrieval_demo.py --mode context

# Performance benchmarks
python src/retrieval_demo.py --mode benchmark

# Interactive search mode
python src/retrieval_demo.py --mode interactive
```

## Complete Workflow

### 1. Index Your Corpus First

```powershell
cd src
python index_corpus.py --data-dir ../data --clear
```

Wait for indexing to complete. You should see:
```
✓ Validation PASSED
✓ 3/3 spot checks passed
```

### 2. Test Retrieval

Basic retrieval test (no API needed):
```powershell
python retrieval_test.py
```

Shows:
- k=1, 3, 5 comparison
- Result ranking by similarity
- Metadata preservation

### 3. Run Interactive Demo

```powershell
python retrieval_demo.py --mode basic
```

Choose demo mode:
- `basic` - Single query retrieval
- `k-compare` - Compare k values (1, 3, 5)
- `context` - Build LLM prompt context
- `interactive` - Try multiple queries
- `benchmark` - Performance testing

## Detailed Commands

### Basic Retrieval

```powershell
# Default: top-3 results
python src/retrieval_demo.py --mode basic

# Custom vector directory
python src/retrieval_demo.py --mode basic --vector-dir outputs/chroma_db
```

### Top-K Comparison

See how results change with different k values:

```powershell
python src/retrieval_demo.py --mode k-compare
```

Tests k=1, 3, 5 and shows:
- Retrieved chunks for each k
- Quality metrics
- Trade-offs between focused vs. comprehensive

### Context Building

Generate formatted context for LLM:

```powershell
python src/retrieval_demo.py --mode context
```

Shows:
- Formatted context string
- Complete LLM prompt template
- How to integrate with LLM call

### Interactive Search

```powershell
python src/retrieval_demo.py --mode interactive
```

Commands:
```
<query>              Search with default k=3
k=5 <query>         Search with custom k value
compare 1,3,5       Compare k values
help               Show help
quit               Exit
```

Example session:
```
Query: password reset
[displays top 3 results]

Query: k=5 account access
[displays top 5 results]

Query: compare 1,3,5
[enters comparison mode]
Query for comparison: security procedures
[shows results for k=1, 3, 5]
```

### Performance Benchmarks

```powershell
python src/retrieval_demo.py --mode benchmark
```

Tests latency across:
- Multiple queries
- k values (1, 3, 5, 10)
- Reports in milliseconds

## Python API Usage

### Basic Retrieval

```powershell
python -c "
from src.retrieval import retrieve
from src.vector_store import VectorStore

store = VectorStore()
response = retrieve('password reset', store, top_k=3)

for result in response.results:
    print(f'{result.rank}. {result.similarity:.4f} | {result.text[:60]}...')
"
```

### Compare k Values

```powershell
python -c "
from src.retrieval import compare_k_values
from src.vector_store import VectorStore

store = VectorStore()
results = compare_k_values('password reset', [1, 3, 5], store)

for k in sorted(results.keys()):
    r = results[k]
    print(f'k={k}: {r.total_retrieved} results, avg sim: {sum(x.similarity for x in r.results)/len(r.results):.4f}')
"
```

### Get Quality Metrics

```powershell
python -c "
from src.retrieval import retrieve, analyze_retrieval_quality
from src.vector_store import VectorStore

store = VectorStore()
response = retrieve('query', store, top_k=5)
metrics = analyze_retrieval_quality(response)

for key, value in metrics.items():
    print(f'{key}: {value}')
"
```

### Build LLM Context

```powershell
python -c "
from src.retrieval import retrieve_with_context
from src.vector_store import VectorStore

store = VectorStore()
context = retrieve_with_context('password reset', store, top_k=3)
print(context)
"
```

## Advanced Usage

### Custom Vector Directory

```powershell
# Use different database
python src/retrieval_demo.py --mode basic --vector-dir outputs/chroma_db_custom
```

### Filter by Metadata

```powershell
python -c "
from src.vector_store import VectorStore

store = VectorStore()

# Filter to specific source
results = store.search(
    query_embedding,
    top_k=5,
    where={'source': 'guide.md'}
)
"
```

### Batch Queries

```powershell
python -c "
from src.retrieval import retrieve
from src.vector_store import VectorStore

store = VectorStore()
queries = ['password reset', 'safety procedure', 'maintenance']

for query in queries:
    response = retrieve(query, store, top_k=3)
    print(f'{query}: {response.total_retrieved} results')
"
```

### Export Results

```powershell
python -c "
import json
from src.retrieval import retrieve
from src.vector_store import VectorStore

store = VectorStore()
response = retrieve('query', store, top_k=5)

results = []
for r in response.results:
    results.append({
        'rank': r.rank,
        'similarity': r.similarity,
        'text': r.text,
        'source': r.metadata.get('source'),
    })

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
"
```

## Testing & Validation

### Unit Test

```powershell
python src/retrieval_test.py
```

Creates sample corpus and tests:
- Top-k retrieval with different k
- Result ranking
- Metadata preservation

### Test with Real Corpus

```powershell
# 1. Index your documents
python src/index_corpus.py --data-dir data --clear

# 2. Test retrieval
python src/retrieval_demo.py --mode basic
```

## Integration with LLM

### Step 1: Retrieve Context

```powershell
python -c "
from src.retrieval import retrieve_with_context

query = 'Your question here'
context = retrieve_with_context(query, top_k=3)
"
```

### Step 2: Build Prompt

```powershell
python -c "
prompt = '''Answer the question based on context.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:'''
"
```

### Step 3: Call LLM

```powershell
python -c "
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model='gpt-4',
    messages=[{'role': 'user', 'content': prompt}]
)
print(response.choices[0].message.content)
"
```

## Troubleshooting

### Vector Store Empty

```powershell
# Check count
python -c "from src.vector_store import VectorStore; print(VectorStore().count())"

# Index if empty
python src/index_corpus.py --data-dir data --clear
```

### No Results Found

```powershell
# Try larger k
python src/retrieval_demo.py --mode basic  # Uses default k=3

# Try interactive with different query
python src/retrieval_demo.py --mode interactive
```

### Slow Queries

```powershell
# Check corpus size
python -c "from src.vector_store import VectorStore; print(f'Indexed: {VectorStore().count()}')"

# Run benchmark
python src/retrieval_demo.py --mode benchmark
```

### API Key Issues

```powershell
# Test without API (uses dummy embeddings)
python src/retrieval_test.py

# For real retrieval with queries, set .env:
# OPENAI_API_KEY=your_key
# OPENAI_BASE_URL=https://api.openai.com/v1
```

## Performance Tips

1. **Optimize k value**
   - Start with k=3
   - Increase only if needed for better recall
   - Higher k = more cost and latency

2. **Use metadata filters**
   - Filter to specific document types
   - Reduces search space
   - Faster retrieval

3. **Batch queries when possible**
   - Multiple queries at once
   - Reduce initialization overhead

4. **Monitor similarity scores**
   - Scores < 0.5 may be low quality
   - Consider increasing k or changing query

## File Locations

```
Project Root
├── src/
│   ├── retrieval.py           ← Core retrieval engine
│   ├── retrieval_demo.py      ← Interactive demo
│   ├── retrieval_test.py      ← Unit tests
│   ├── vector_store.py
│   ├── embeddings.py
│   ├── indexing.py
│   └── index_corpus.py
│
├── outputs/
│   ├── chroma_db/             ← Indexed corpus
│   └── embedding_cache.json
│
├── data/                      ← Source documents
│   ├── machine_manual.txt
│   ├── maintenance_log.txt
│   └── safety_procedure.md
│
├── RETRIEVAL_GUIDE.md         ← Full documentation
└── RETRIEVAL_COMMANDS.md      ← This file
```

---

**Pro Tips:**
1. Test with `retrieval_test.py` first (no API needed)
2. Use `--mode k-compare` to tune k value
3. Save `.env` with API keys for real embedding queries
4. Monitor first queries (slower due to initialization)
5. Use `--mode benchmark` to understand latency
