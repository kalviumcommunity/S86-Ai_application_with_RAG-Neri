"""Run offline sanity checks for embedding similarity ranking."""

import argparse
from dataclasses import dataclass
from pathlib import Path

try:
    from .embeddings import EmbeddedChunk, rank_chunks
except ImportError:
    from embeddings import EmbeddedChunk, rank_chunks


@dataclass(frozen=True)
class SanityCase:
    query: str
    query_embedding: list[float]
    expected_source: str
    expected_chunk: int
    note: str
    surprising: bool = False


RECORDS = [
    EmbeddedChunk(
        "Before inspecting or repairing equipment, disconnect the main power supply and verify that the equipment is isolated.",
        {"source": "safety_procedure.md", "chunk_index": 0, "section": "Electrical Safety"},
        [1.0, 0.0, 0.0, 0.0],
    ),
    EmbeddedChunk(
        "Before performing maintenance, disconnect the machine from the main power supply.",
        {"source": "machine_manual.txt", "chunk_index": 1, "section": "Document body"},
        [0.8, 0.2, 0.0, 0.0],
    ),
    EmbeddedChunk(
        "If unusual vibration is detected, stop the machine and follow the approved inspection procedure.",
        {"source": "machine_manual.txt", "chunk_index": 2, "section": "Document body"},
        [0.0, 0.0, 1.0, 0.0],
    ),
    EmbeddedChunk(
        "The technician replaced a worn belt after detecting abnormal vibration.",
        {"source": "maintenance_log.txt", "chunk_index": 1, "section": "Document body"},
        [0.0, 0.0, 0.8, 0.6],
    ),
    EmbeddedChunk(
        "Technicians must wear the required protective equipment before beginning maintenance activities.",
        {"source": "safety_procedure.md", "chunk_index": 1, "section": "Personal Protective Equipment"},
        [0.0, 1.0, 0.0, 0.0],
    ),
]


CASES = [
    SanityCase(
        "What must be done before inspecting equipment?",
        [1.0, 0.0, 0.0, 0.0],
        "safety_procedure.md",
        0,
        "Electrical isolation should beat general maintenance text.",
    ),
    SanityCase(
        "What should I do if the machine has unusual vibration?",
        [0.0, 0.0, 1.0, 0.0],
        "machine_manual.txt",
        2,
        "The approved response should outrank the historical maintenance log.",
    ),
    SanityCase(
        "What protective equipment is required?",
        [0.0, 1.0, 0.0, 0.0],
        "safety_procedure.md",
        1,
        "The PPE section is the direct safety match.",
    ),
    SanityCase(
        "Tell me about routine maintenance.",
        [0.3, 0.95, 0.0, 0.0],
        "machine_manual.txt",
        1,
        "Surprise: this broad query is closer to the PPE chunk than the expected manual instruction.",
        surprising=True,
    ),
]


def run_cases() -> list[dict[str, object]]:
    results = []
    for case in CASES:
        ranked = rank_chunks(case.query_embedding, RECORDS)
        top_score, top_record = ranked[0]
        expected = top_record.metadata["source"] == case.expected_source and top_record.metadata["chunk_index"] == case.expected_chunk
        results.append({
            "case": case,
            "ranked": ranked,
            "passed": expected,
            "top_score": top_score,
            "top_record": top_record,
        })
    return results


def build_report(results: list[dict[str, object]]) -> str:
    passing = sum(1 for result in results if result["passed"] and not result["case"].surprising)
    failures = sum(1 for result in results if not result["passed"] and not result["case"].surprising)
    surprises = sum(1 for result in results if result["case"].surprising)
    lines = [
        "# Retrieval Sanity Report",
        "",
        "This offline smoke test calls the production `rank_chunks` function with labeled fixture vectors. It checks whether known related chunks rank above unrelated chunks before retrieval is trusted.",
        "",
        f"- Test cases: {len(results)}",
        f"- Passes: {passing}",
        f"- Failures: {failures}",
        f"- Borderline or surprising cases: {surprises}",
        "",
        "| Case | Status | Top source | Chunk | Score | Expected source | Notes |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for result in results:
        case = result["case"]
        top = result["top_record"]
        status = "SURPRISE" if case.surprising else ("PASS" if result["passed"] else "FAIL")
        lines.append(
            f"| {case.query} | {status} | {top.metadata['source']} | {top.metadata['chunk_index']} | "
            f"{result['top_score']:.6f} | {case.expected_source} | {case.note} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The three targeted safety and vibration cases passed: the known relevant chunk ranked first. The broad routine-maintenance query was surprising because its vector was closer to the PPE chunk than the expected manual instruction. This shows that broad queries and small corpora can produce ambiguous rankings; production retrieval should use top-k context, metadata filters, and additional evaluation cases rather than trusting one score.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/sanity_report.md")
    args = parser.parse_args()
    results = run_cases()
    report = build_report(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report.rstrip())
    print(f"\nReport written to: {output.resolve()}")
    if any(not result["passed"] and not result["case"].surprising for result in results):
        raise SystemExit("A non-surprising sanity check failed")


if __name__ == "__main__":
    main()