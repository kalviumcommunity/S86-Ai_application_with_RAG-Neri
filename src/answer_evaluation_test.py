"""Tests for answer quality scoring."""

import unittest

try:
    from .answer_evaluation import score_answer
except ImportError:
    from answer_evaluation import score_answer


class AnswerEvaluationTests(unittest.TestCase):
    def test_scores_supported_claim_and_citation(self):
        chunks = [{
            "text": "Disconnect the main power supply before inspection.",
            "metadata": {"source": "electrical_safety.txt"},
        }]
        case = {
            "id": "one",
            "question": "What should I do?",
            "claims": [{
                "text": "Disconnect the main power supply before inspection.",
                "required_terms": ["disconnect", "main power", "inspection"],
                "supporting_sources": ["electrical_safety.txt"],
            }],
        }

        result = score_answer(case, "Disconnect the main power supply before inspection. [1]", chunks)

        self.assertEqual(result["correctness"], 1.0)
        self.assertEqual(result["grounding"], 1.0)
        self.assertEqual(result["citation_quality"], 1.0)

    def test_rejects_invalid_citation_marker(self):
        chunks = [{
            "text": "Stop the equipment immediately.",
            "metadata": {"source": "vibration_procedure.txt"},
        }]
        case = {
            "id": "two",
            "question": "What should I do?",
            "claims": [{
                "text": "Stop the equipment immediately.",
                "required_terms": ["stop", "equipment", "immediately"],
                "supporting_sources": ["vibration_procedure.txt"],
            }],
        }

        result = score_answer(case, "Stop the equipment immediately. [2]", chunks)

        self.assertEqual(result["correctness"], 1.0)
        self.assertFalse(result["citation_markers_valid"])
        self.assertEqual(result["citation_quality"], 0.0)


if __name__ == "__main__":
    unittest.main()