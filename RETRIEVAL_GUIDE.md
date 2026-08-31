# Similarity Search & Top-K Retrieval

This guide covers the similarity search and retrieval functionality for your RAG application, implementing the concepts from the retrieval assignment.

## What is Similarity Search?

Similarity search finds the chunks in your indexed corpus that are most semantically similar to a user's query. The process:

1. **Embed the query** - Convert user query to a vector using the same embedding model as documents
2. **Search vector database** - Find chunks with vectors closest to the query vector
3. **Return top-k results** - Return the k highest-scoring matches ranked by similarity
4. **Format for LLM** - Include text, metadata, and similarity scores for grounding

## Query to Retrieved Chunks

```
User Query
    ↓
[embed with same model]
    ↓
Query Vector (1536-dim)
    ↓
[cosine similarity search]
    ↓
Vector Database (indexed corpus)
    ↓
Top-k ranked results
    ↓
[format with text, metadata, scores]
    ↓
Retrieved Context for LLM
```

## Quick Start

### 1. Retrieve with Default k=3

```bash
cd src
python retrieval_demo.py --mode basic
```

Returns top 3 most similar chunks with scores.

### 2. Compare Different k Values

See how results change with different k:

```bash
python retrieval_demo.py --mode k-compare
```

Tests k=1, 3, 5 to show the trade-off between focused vs. comprehensive context.

### 3. Build LLM Context

Generate formatted context ready for LLM prompt:

```bash
python retrieval_demo.py --mode context
```

Shows context formatted for language model and complete prompt template.

### 4. Interactive Search

Try multiple queries interactively:

```bash
python retrieval_demo.py --mode interactive
```

Commands:
- `<query>` - Search with default k=3
- `k=5 <query>` - Search with custom k value
- `compare 1,3,5` - Compare k values
- `help` - Show commands
- `quit` - Exit

### 5. Performance Benchmarks

Test retrieval latency across k values:

```bash
python retrieval_demo.py --mode benchmark
```

## Top-K Parameter

**Top-k means "return the k highest-scoring matches."**

```python
# k=1: Most focused, lowest cost
retrieve(query, k=1)  # Just the best match

# k=3: Balanced (default)
retrieve(query, k=3)  # Three good results

# k=5: More context, higher cost
retrieve(query, k=5)  # Five results to choose from
```

### Trade-offs

| k | Pros | Cons |
|---|------|------|
| 1 | Fast, focused, lowest cost | May miss context, risky |
| 3 | Good balance | Still selective |
| 5+ | More context, better recall | Higher cost, potential noise |

**Key insight**: Larger k can improve recall (including more potentially useful chunks), but increases latency, cost, and context window usage. Tune k with real queries, not guesses.

## Retrieval Result Structure

Each retrieved chunk contains:

```python
{
    "rank": 1,                    # Position in results (1-indexed)
    "score": 0.15,                # Distance (lower better for cosine)
    "similarity": 0.85,           # Similarity 0-1 (higher better)
    "chunk_id": "guide.md:0",    # Stable chunk ID
    "text": "Password reset...",  # Source text
    "metadata": {
        "source": "guide.md",     # Document source
        "chunk_index": 0,         # Position in document
        "section": "Account"      # Optional section
    }
}
```

## API Reference

### `retrieve(query, vector_store=None, top_k=3, where=None)`

Basic similarity search.

```python
from retrieval import retrieve
from vector_store import VectorStore

store = VectorStore()
response = retrieve("password reset", store, top_k=5)

for result in response.results:
    print(f"{result.rank}. Similarity: {result.similarity:.4f}")
    print(f"   {result.text[:80]}...")
```

**Parameters:**
- `query` - User query string
- `vector_store` - VectorStore instance (optional)
- `top_k` - Number of results (default: 3)
- `where` - Metadata filter dict (optional)

**Returns:** `RetrievalResponse` with ranked results

### `retrieve_with_context(query, vector_store=None, top_k=3)`

Retrieve and format as context string for LLM.

```python
from retrieval import retrieve_with_context

context = retrieve_with_context("password reset", top_k=3)
print(context)  # Formatted context ready for prompt
```

### `compare_k_values(query, k_values, vector_store=None)`

Compare results for different k values.

```python
from retrieval import compare_k_values, print_k_comparison

results = compare_k_values("password", [1, 3, 5, 10])
print_k_comparison("password", results)
```

Shows how retrieved chunks change as k increases.

### `analyze_retrieval_quality(response)`

Get quality metrics for a retrieval response.

```python
from retrieval import retrieve, analyze_retrieval_quality

response = retrieve("query", top_k=3)
metrics = analyze_retrieval_quality(response)

print(f"Avg similarity: {metrics['avg_similarity']:.4f}")
print(f"Top result quality: {metrics['top_result_quality']}")
```

**Metrics:**
- `num_results` - Number of results returned
- `avg_similarity` - Average similarity score
- `min_similarity` - Lowest score
- `max_similarity` - Highest score
- `similarity_spread` - Difference between highest and lowest
- `top_result_quality` - HIGH/MEDIUM/LOW based on top score

## Inspecting Retrieved Context

### Complete Output

```python
from retrieval import retrieve, print_retrieval_results

response = retrieve("How do I reset my password?", top_k=3)
print_retrieval_results(response)
```

Output:
```
============================================================================
SIMILARITY SEARCH RESULTS
============================================================================

Query: How do I reset my password?
Top-k: 3
Retrieved: 3 chunks

[1] Similarity: 0.8234 | Distance: 0.1766
    ID: guide.md:0
    Source: guide.md
    Section: Account Access
    Chunk Index: 0
    Text: Password reset instructions for learner accounts...

[2] Similarity: 0.7891 | Distance: 0.2109
    ID: guide.md:5
    Source: guide.md
    Section: Security
    ...
```

### Quick Result Summary

```python
for result in response.results:
    print(f"[{result.rank}] {result.similarity:.4f} | {result.text[:70]}...")
```

## Using Retrieved Context in LLM Prompts

### Retrieve and Format

```python
from retrieval import retrieve_with_context

query = "How can I reset my password?"
context = retrieve_with_context(query, top_k=3)
```

### Build Complete Prompt

```python
prompt = f"""Answer the following question based on the provided context.
If the context does not contain the answer, say so.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

# Send to LLM
# response = llm_client.chat.completions.create(
#     model="gpt-4",
#     messages=[{"role": "user", "content": prompt}]
# )
```

## Tuning Top-K

Start with k=3 and adjust based on:

1. **Chunk size**: Smaller chunks → larger k
2. **Query complexity**: Complex queries → larger k
3. **Context window**: Limited window → smaller k
4. **Quality vs. latency**: Need speed → smaller k

**Testing example:**

```python
from retrieval import compare_k_values, print_k_comparison

query = "your complex query"
comparison = compare_k_values(query, [1, 3, 5, 10])
print_k_comparison(query, comparison)

# Evaluate: Are the extra chunks helpful or just noise?
```

## Common Patterns

### Search with Metadata Filter

```python
# Retrieve only from specific document
response = retrieve(
    query,
    top_k=5,
    where={"source": "guide.md"}
)

# Or specific section
response = retrieve(
    query,
    top_k=5,
    where={"section": "Security"}
)
```

### Multi-Query Retrieval

```python
queries = [
    "password reset",
    "account access",
    "security procedures"
]

all_results = []
for query in queries:
    response = retrieve(query, top_k=3)
    all_results.extend(response.results)
```

### Filtering by Similarity Threshold

```python
response = retrieve(query, top_k=10)

# Only use high-confidence results
high_confidence = [
    r for r in response.results 
    if r.similarity > 0.7
]
```

## Troubleshooting

**"Vector store is empty"**
- Run indexing first: `python index_corpus.py --data-dir data`

**"No results found"**
- Increase top_k: `retrieve(query, top_k=10)`
- Check if corpus has relevant content
- Try different query phrasing

**"Low similarity scores (< 0.5)"**
- Corpus may not contain relevant information
- Query may be too specific or misaligned with corpus
- Try broader query terms

**"Slow retrieval (>1 second)"**
- Normal for first query (initialization)
- Increase batch size for faster embedding
- Check system resources

## Performance Characteristics

Approximate latency for typical queries:
- **Query embedding**: 10-50ms (first query slower)
- **Vector search**: 1-5ms (depends on corpus size)
- **Formatting**: <1ms
- **Total**: 15-60ms per query

Factors affecting performance:
- Query embedding size (fixed at 1536-dim)
- Number of indexed chunks (search grows logarithmically)
- Top-k value (higher k → slightly slower)
- Vector database implementation (Chroma is optimized)

## Next Steps

1. **Index your corpus**: `python index_corpus.py --data-dir data`
2. **Test retrieval**: `python retrieval_demo.py --mode basic`
3. **Compare k values**: `python retrieval_demo.py --mode k-compare`
4. **Build LLM context**: Use retrieved results in prompt generation
5. **Tune parameters**: Experiment with k, metadata filters, similarity thresholds

## File Structure

```
src/
├── retrieval.py              ← Core retrieval engine
├── retrieval_demo.py         ← Interactive demo
├── vector_store.py           ← Vector database
├── embeddings.py             ← Embedding generation
├── indexing.py               ← Corpus indexing
└── ...

outputs/
├── chroma_db/                ← Indexed corpus
└── ...
```

## References

- Chroma Query Collections: https://docs.trychroma.com/docs/querying-collections/query-and-get
- Qdrant Search: https://qdrant.tech/documentation/concepts/search/
- Pinecone Semantic Search: https://docs.pinecone.io/guides/search/semantic-search
