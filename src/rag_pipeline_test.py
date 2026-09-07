"""Tests for the query-time RAG stages."""

import unittest

try:
    from .conversational_rag import ConversationalRAG, default_rewrite_query
    from .rag_pipeline import (
        NO_CONTEXT_ANSWER,
        answer_query,
        assemble_context,
        build_augmented_prompt,
    )
except ImportError:
    from conversational_rag import ConversationalRAG, default_rewrite_query
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

    def test_conversational_rag_rewrites_follow_up_and_tracks_history(self):
        store = FakeStore([{
            "text": "Isolate electrical power before inspecting the motor.",
            "metadata": {"source": "electrical_safety.txt"},
        }])
        embedded_queries = []
        generated_queries = []

        def embedder(texts):
            embedded_queries.append(texts[0])
            return [[0.5, 0.5]]

        def generator(query, context):
            generated_queries.append(query)
            return "Power must be isolated before inspection."

        def rewriter(question, history):
            if not history.turns:
                return question
            return "What safety precautions apply before inspecting the motor?"

        conversation = ConversationalRAG(
            store,
            embedder=embedder,
            generator=generator,
            rewriter=rewriter,
            top_k=1,
        )
        first = conversation.ask("What should I do before inspecting the motor?")
        second = conversation.ask("What about safety?")

        self.assertEqual(first["rewritten_query"], first["question"])
        self.assertEqual(second["rewritten_query"], "What safety precautions apply before inspecting the motor?")
        self.assertEqual(embedded_queries[1], second["rewritten_query"])
        self.assertEqual(generated_queries[1], second["rewritten_query"])
        self.assertEqual(len(conversation.history.turns), 2)
        self.assertEqual(len(second["history"]), 4)
        self.assertTrue(second["retrieved"])

    def test_default_rewriter_uses_previous_turn(self):
        store = FakeStore([])
        conversation = ConversationalRAG(
            store,
            embedder=lambda texts: [[1.0]],
            generator=lambda query, context: "No context.",
        )
        conversation.ask("What is the restart procedure for the conveyor?")

        rewritten = default_rewrite_query(
            "Does it require a safety check?",
            conversation.history,
        )

        self.assertIn("restart procedure for the conveyor", rewritten)
        self.assertIn("Does it require a safety check?", rewritten)


if __name__ == "__main__":
    unittest.main()