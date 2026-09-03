"""Tests for the query-time RAG stages."""

import unittest

try:
    from .rag_pipeline import (
        NO_CONTEXT_ANSWER,
        answer_query,
        assemble_context,
        build_augmented_prompt,
    )
except ImportError:
    from rag_pipeline import NO_CONTEXT_ANSWER, answer_query, assemble_context, build_augmented_prompt


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

    def test_augmented_prompt_stays_within_budget_and_reserves_answer_space(self):
        chunks = [
            {"text": "Power must be isolated before inspection.", "metadata": {"source": "safety.txt"}},
            {"text": "A very long chunk " * 100, "metadata": {"source": "manual.txt"}},
        ]
        result = build_augmented_prompt(
            "What should I do first?",
            chunks,
            model_token_budget=80,
            answer_token_reserve=20,
        )

        self.assertIn("Answer only from the provided context", result["prompt"])
        self.assertIn("[1] Source: safety.txt", result["prompt"])
        self.assertNotIn("[2] Source: manual.txt", result["prompt"])
        self.assertLessEqual(result["total_reserved_tokens"], 80)


if __name__ == "__main__":
    unittest.main()