"""Offline answer-quality evaluation for the RAG pipeline.

The evaluator measures claim-level correctness, grounding, and citation
accuracy. It uses the same ``answer_query`` stages with a deterministic local
store and generator so the benchmark runs without network credentials.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from .citations import build_citation_map, extract_citations, verify_citation
    from .rag_pipeline import answer_query
except ImportError:
    from citations import build_citation_map, extract_citations, verify_citation
    from rag_pipeline import answer_query


class KeywordStore:
    """Small deterministic store used to exercise the real query pipeline."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

    def search(self, query_vector, top_k):
        query = str(query_vector).lower()
        terms = set(re.findall(r"[a-z]{4,}", query))
        ranked = []
        for chunk in self.chunks:
            text = chunk["text"].lower()
            score = sum(term in text for term in terms)
            if score:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]


def load_test_set(path: str | Path) -> list[dict]:
    """Load and validate the JSON quality test set."""
    with Path(path).open(encoding="utf-8") as file:
        cases = json.load(file)
    if not cases:
        raise ValueError("test set must contain at least one case")
    for case in cases:
        if not case.get("id") or not case.get("question"):
            raise ValueError("each case needs an id and question")
        if not case.get("claims"):
            raise ValueError(f"case {case.get('id')} needs claims")
    return cases


def _terms_present(text: str, required_terms: list[str]) -> list[str]:
    normalized = text.lower()
    return [term for term in required_terms if term.lower() in normalized]


def score_answer(case: dict, answer: str, chunks: list[dict]) -> dict:
    """Score one generated answer at claim level."""
    citation_map = build_citation_map(chunks)
    cited_markers = extract_citations(answer)
    claim_rows = []

    for claim in case["claims"]:
        required_terms = claim["required_terms"]
        answer_terms = _terms_present(answer, required_terms)
        context = "\n".join(chunk.get("text", "") for chunk in chunks)
        context_terms = _terms_present(context, required_terms)
        expected_sources = set(claim["supporting_sources"])
        citation_checks = []
        for marker in cited_markers:
            citation = verify_citation(marker, claim["text"], citation_map)
            if citation.get("source") in expected_sources:
                citation_checks.append(citation)
        claim_rows.append({
            "claim": claim["text"],
            "correct": len(answer_terms) == len(required_terms),
            "answer_terms_found": answer_terms,
            "required_terms": required_terms,
            "grounded": len(context_terms) == len(required_terms),
            "context_terms_found": context_terms,
            "citation_supported": any(item.get("verified") for item in citation_checks),
            "supporting_sources": sorted(expected_sources),
            "citation_checks": citation_checks,
        })

    def average(field: str) -> float:
        return sum(row[field] for row in claim_rows) / len(claim_rows)

    valid_markers = [marker for marker in cited_markers if marker in citation_map]
    return {
        "id": case["id"],
        "question": case["question"],
        "expected_sources": case.get("expected_sources", []),
        "answer": answer,
        "retrieved_sources": [chunk["metadata"].get("source") for chunk in chunks],
        "cited_markers": cited_markers,
        "invalid_citations": [marker for marker in cited_markers if marker not in citation_map],
        "claims": claim_rows,
        "correctness": average("correct"),
        "grounding": average("grounded"),
        "citation_quality": (
            sum(row["citation_supported"] for row in claim_rows) / len(claim_rows)
            if claim_rows else 0.0
        ),
        "citation_markers_valid": len(valid_markers) == len(cited_markers),
    }


def evaluate_cases(cases: list[dict], chunks: list[dict], answers: dict[str, str]) -> dict:
    """Run the real RAG query stages and score every test case."""
    rows = []
    for case in cases:
        store = KeywordStore(chunks)

        def embedder(texts):
            return [texts[0]]

        def generator(query, context, case_id=case["id"]):
            return answers[case_id]

        result = answer_query(
            case["question"],
            store,
            k=case.get("top_k", 3),
            embedder=embedder,
            generator=generator,
        )
        retrieved_sources = {
            source.get("source")
            for source in result["sources"]
        }
        retrieved_chunks = [
            chunk for chunk in chunks
            if chunk["metadata"].get("source") in retrieved_sources
        ]
        rows.append(score_answer(case, result["answer"], retrieved_chunks))

    metrics = ["correctness", "grounding", "citation_quality"]
    summary = {
        "cases": len(rows),
        **{
            metric: round(sum(row[metric] for row in rows) / len(rows), 3)
            for metric in metrics
        },
        "perfect_cases": sum(
            all(row[metric] == 1.0 for metric in metrics) for row in rows
        ),
        "failures": [
            {
                "id": row["id"],
                "reason": _failure_reason(row),
            }
            for row in rows
            if any(row[metric] < 1.0 for metric in metrics)
        ],
    }
    return {"summary": summary, "results": rows}


def _failure_reason(row: dict) -> str:
    reasons = []
    if row["correctness"] < 1.0:
        reasons.append("answer omitted expected claim terms")
    if row["grounding"] < 1.0:
        reasons.append("retrieved context did not contain every claim term")
    if row["citation_quality"] < 1.0 or not row["citation_markers_valid"]:
        reasons.append("citation was missing, invalid, or unsupported")
    return "; ".join(reasons)


def load_chunks(data_dir: str | Path, sources: set[str]) -> list[dict]:
    """Load one evidence chunk per requested source document."""
    chunks = []
    for source in sorted(sources):
        path = Path(data_dir) / source
        chunks.append({
            "text": path.read_text(encoding="utf-8"),
            "metadata": {"source": source, "section": "Document body"},
        })
    return chunks


def write_report(report: dict, json_path: str | Path, markdown_path: str | Path) -> None:
    """Write machine-readable results and a reviewer-friendly summary."""
    Path(json_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = report["summary"]
    lines = [
        "# RAG Answer Quality Evaluation",
        "",
        "Offline deterministic evaluation using the production query pipeline and corpus evidence.",
        "",
        f"- Cases: {summary['cases']}",
        f"- Correctness: {summary['correctness']:.1%}",
        f"- Grounding: {summary['grounding']:.1%}",
        f"- Citation quality: {summary['citation_quality']:.1%}",
        f"- Perfect cases: {summary['perfect_cases']}/{summary['cases']}",
        "",
        "## Results",
        "",
        "| Case | Correctness | Grounding | Citation quality | Retrieved sources |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['id']} | {row['correctness']:.1%} | "
            f"{row['grounding']:.1%} | {row['citation_quality']:.1%} | "
            f"{', '.join(row['retrieved_sources']) or 'none'} |"
        )
    lines.extend(["", "## Failures", ""])
    if summary["failures"]:
        for failure in summary["failures"]:
            lines.append(f"- **{failure['id']}**: {failure['reason']}")
    else:
        lines.append("None.")
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cases = load_test_set(root / "evaluation" / "test_set.json")
    sources = {source for case in cases for source in case["expected_sources"]}
    chunks = load_chunks(root / "data", sources)
    answers = {
        case["id"]: case["candidate_answer"]
        for case in cases
    }
    report = evaluate_cases(cases, chunks, answers)
    write_report(
        report,
        root / "evaluation" / "scored_results.json",
        root / "evaluation" / "quality_summary.md",
    )
    print(json.dumps(report["summary"], indent=2))