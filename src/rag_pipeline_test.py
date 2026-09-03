"""Tests for the query-time RAG stages."""

import unittest

try:
    from .rag_pipeline import NO_CONTEXT_ANSWER, answer_query, assemble_context
except ImportError:
    from rag_pipeline import NO_CONTEXT_ANSWER, answer_query, assemble_context


class FakeStore:
    def __init__(self, results):
        self.results = results
        self.query_vector = None
        self.top_k = None

    def search(self, query_vector, top_k):
        self.query_vector = query_vector
        self.top_k = top_k
        return self.results


class RagPipelineTests(unittest.TestCase):
    def test_answer_query_connects_all_stages_and_returns_sources(self):
        store = FakeStore([{
            "text": "Isolate electrical power before opening the motor housing.",
            "metadata": {"source": "electrical_safety.txt", "section": "Isolation"},
        }])
        captured = {}

        def embedder(texts):
            captured["query"] = texts[0]
            return [[0.25, 0.75]]

        def generator(query, context):
            captured["context"] = context
            return "Power must be isolated before inspection."

        result = answer_query(
            "What should I do before inspecting the motor?",
            store,
            k=1,
            embedder=embedder,
            generator=generator,
        )

        self.assertEqual(store.query_vector, [0.25, 0.75])
        self.assertEqual(store.top_k, 1)
        self.assertIn("[1] Source: electrical_safety.txt (Isolation)", captured["context"])
        self.assertEqual(result["answer"], "Power must be isolated before inspection.")
        self.assertEqual(result["sources"][0]["source"], "electrical_safety.txt")

    def test_empty_retrieval_returns_fallback_without_generation(self):
        store = FakeStore([])

        def generator(*args):
            raise AssertionError("generation must not run without context")

        result = answer_query(
            "What should I do?",
            store,
            embedder=lambda texts: [[1.0]],
            generator=generator,
        )

        self.assertEqual(result, {"answer": NO_CONTEXT_ANSWER, "sources": []})

    def test_assemble_context_numbers_sources(self):
        context = assemble_context([
            {"text": "First", "metadata": {"source": "one.txt"}},
            {"text": "Second", "metadata": {"source": "two.txt"}},
        ])
        self.assertEqual(context, "[1] Source: one.txt\nFirst\n\n[2] Source: two.txt\nSecond")


if __name__ == "__main__":
    unittest.main()