# Batch Embedding Run Summary

The batch pipeline in `src/embeddings.py` embeds uncached token chunks, retries transient API failures with exponential backoff, persists successful vectors, and reuses them on later runs. Cache keys include chunk text, metadata, and model name.

## Configuration

| Setting | Value |
| --- | --- |
| Corpus | `data/` |
| Chunks | 7 |
| Batch size | 3 |
| Maximum retries | 3 |
| Backoff | 1, 2, 4 seconds |
| Cache | `outputs/embedding_cache.json` |
| Approximate rate | `$0.00002` per 1K input tokens |

## First run

| Metric | Result |
| --- | ---: |
| Total chunks | 7 |
| Batches | 3 |
| Embeddings generated | 7 |
| Skipped chunks | 0 |
| Retry attempts | 1 |
| Failed batches | 0 |
| Input tokens | 336 |
| Approximate cost | `$0.00000672` |

One simulated transient rate-limit response was retried successfully. Successful batches are written to the JSON cache immediately, so an interruption does not discard completed work.

## Rerun

| Metric | Result |
| --- | ---: |
| Total chunks | 7 |
| Batches sent | 0 |
| Embeddings generated | 0 |
| Skipped chunks | 7 |
| Retry attempts | 0 |
| Failed batches | 0 |
| Approximate cost | `$0.00000000` |

The rerun detects all seven cached chunk embeddings and makes no duplicate embedding requests.