# Retrieval Evaluation

`src/retrieval_evaluation.py` evaluates retrieval against a small, trusted set
of manually labelled chunk IDs. Each label should use the stable
`source:chunk_index` ID created during indexing.

```python
from src.retrieval_evaluation import evaluate, print_evaluation_report
from src.vector_store import VectorStore

labelled_queries = [
    {
        "query": "What should I check when a motor vibrates?",
        "relevant_chunk_ids": {"vibration_procedure.txt:0"},
    },
]

report = evaluate(labelled_queries, VectorStore(), k=5)
print_evaluation_report(report)
```

Each row contains the retrieved IDs, expected IDs, hits, recall, and
precision. The summary reports macro-average `recall@k` and `precision@k` and
keeps every query whose recall is below 1.0 in `failures`.

To test the evaluator without an API call or a populated Chroma database:

```text
python -c "import src.retrieval_test as t; t.test_evaluation_reports_recall_precision_and_failures(); t.test_evaluation_rejects_missing_labels_and_empty_sets(); print('evaluation tests passed')"
```

When a query fails, inspect its expected and retrieved IDs before changing the
system. Possible next experiments include increasing `k`, improving chunking,
adding metadata filters, or enabling hybrid keyword scoring through
`retrieval_options`. Change one variable at a time so the metric identifies
what helped.