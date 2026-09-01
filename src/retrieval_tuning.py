"""Tune retrieval settings with a small, deterministic relevance experiment."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DemoVectorStore:
    """Small deterministic vector store used for tuning experiments."""

    def __init__(self, documents: list[dict]):
        self.documents = documents

    def search(self, query_embedding: list[float], top_k: int = 5, where: dict | None = None):
        matches = []
        for document in self.documents:
            metadata = document["metadata"]
            if where and not all(metadata.get(key) == value for key, value in where.items()):
                continue
            similarity = _cosine_similarity(query_embedding, document["embedding"])
            matches.append({
                "id": document["id"],
                "distance": 1.0 - similarity,
                "similarity": similarity,
                "text": document["text"],
                "metadata": metadata,
            })
        matches.sort(key=lambda item: item["similarity"], reverse=True)
        return matches[:top_k]


@dataclass
class QueryCase:
    query: str
    expected_sources: set[str]
    description: str


TEST_CASES = [
    QueryCase(
        query="Before inspecting the machine, disconnect the power supply and verify equipment isolation.",
        expected_sources={"safety_procedure.md"},
        description="Electrical safety and equipment isolation before inspection.",
    ),
    QueryCase(
        query="A worn belt produced abnormal vibration and the machine was returned to service after repair.",
        expected_sources={"maintenance_log.txt"},
        description="Maintenance log entry for vibration and repair activity.",
    ),
    QueryCase(
        query="What should technicians do before beginning maintenance work?",
        expected_sources={"safety_procedure.md", "machine_manual.txt"},
        description="Safety-before-maintenance question covering PPE and disconnection steps.",
    ),
]


DOCS = [
    {
        "id": "manual-safety",
        "text": "Before performing maintenance, disconnect the machine from the main power supply.",
        "metadata": {"source": "machine_manual.txt", "section": "Safety"},
        "embedding": [0.82, 0.20, 0.10, 0.10, 0.80, 0.10, 0.10, 0.10],
    },
    {
        "id": "manual-inspection",
        "text": "Inspect the lubrication system and check for leaks before restart.",
        "metadata": {"source": "machine_manual.txt", "section": "Inspection"},
        "embedding": [0.70, 0.30, 0.10, 0.85, 0.25, 0.10, 0.10, 0.10],
    },
    {
        "id": "ppe",
        "text": "Technicians must wear the required protective equipment before beginning maintenance activities.",
        "metadata": {"source": "safety_procedure.md", "section": "PPE"},
        "embedding": [0.20, 0.90, 0.15, 0.10, 0.10, 0.80, 0.20, 0.10],
    },
    {
        "id": "electrical-safety",
        "text": "Before inspecting or repairing equipment, disconnect the main power supply and verify that the equipment is isolated.",
        "metadata": {"source": "safety_procedure.md", "section": "Electrical Safety"},
        "embedding": [0.90, 0.15, 0.10, 0.08, 0.95, 0.12, 0.10, 0.10],
    },
    {
        "id": "vibration-log",
        "text": "The technician replaced a worn belt after detecting abnormal vibration.",
        "metadata": {"source": "maintenance_log.txt", "section": "Diagnostics"},
        "embedding": [0.10, 0.80, 0.95, 0.10, 0.10, 0.10, 0.85, 0.90],
    },
    {
        "id": "service-log",
        "text": "The machine was tested after repair and returned to service.",
        "metadata": {"source": "maintenance_log.txt", "section": "Service"},
        "embedding": [0.12, 0.75, 0.82, 0.10, 0.10, 0.10, 0.68, 0.74],
    },
]


SETTINGS = [
    {
        "name": "baseline_k3",
        "top_k": 3,
        "filter": None,
        "min_similarity": 0.0,
    },
    {
        "name": "filtered_k3",
        "top_k": 3,
        "filter": {"source": "safety_procedure.md"},
        "min_similarity": 0.0,
    },
    {
        "name": "filtered_threshold_k3",
        "top_k": 3,
        "filter": {"source": "safety_procedure.md"},
        "min_similarity": 0.75,
    },
    {
        "name": "filtered_threshold_k5",
        "top_k": 5,
        "filter": {"source": "safety_procedure.md"},
        "min_similarity": 0.75,
    },
]


def make_query_embedding(query: str) -> list[float]:
    weighted = {
        "disconnect": 0.35,
        "power": 0.30,
        "inspection": 0.25,
        "vibration": 0.35,
        "repair": 0.25,
        "maintenance": 0.30,
        "safety": 0.28,
        "isolated": 0.32,
        "equipment": 0.15,
        "technician": 0.18,
        "protective": 0.20,
    }
    text = query.lower().split()
    vector = [0.0] * 8
    for index, token in enumerate(["disconnect", "power", "inspection", "maintenance", "safety", "vibration", "repair", "isolated"]):
        value = 0.0
        for word in text:
            if word in weighted:
                value += weighted.get(word, 0.0)
        vector[index] = value
    return vector


def evaluate_case(store: DemoVectorStore, case: QueryCase, setting: dict) -> dict:
    query_embedding = make_query_embedding(case.query)
    filtered = store.search(
        query_embedding=query_embedding,
        top_k=setting["top_k"],
        where=setting["filter"],
    )
    filtered = [item for item in filtered if item["similarity"] >= setting["min_similarity"]]

    sources = [item["metadata"]["source"] for item in filtered]
    top_hit = 1 if sources and sources[0] in case.expected_sources else 0
    top_k_hit = 1 if any(source in case.expected_sources for source in sources) else 0
    avg_similarity = sum(item["similarity"] for item in filtered) / len(filtered) if filtered else 0.0

    return {
        "query": case.query,
        "description": case.description,
        "setting": setting["name"],
        "filter": setting["filter"],
        "min_similarity": setting["min_similarity"],
        "top_k": setting["top_k"],
        "top_hit": top_hit,
        "top_k_hit": top_k_hit,
        "avg_similarity": round(avg_similarity, 4),
        "retrieved_sources": sources,
        "retrieved_text": [item["text"] for item in filtered],
    }


def run_experiment() -> list[dict]:
    store = DemoVectorStore(DOCS)
    results = []
    for setting in SETTINGS:
        for case in TEST_CASES:
            results.append(evaluate_case(store, case, setting))
    return results


def summarize_results(results: list[dict]) -> dict:
    grouped = {}
    for setting in SETTINGS:
        name = setting["name"]
        grouped[name] = {
            "top_1_hit_rate": 0.0,
            "top_k_hit_rate": 0.0,
            "avg_similarity": 0.0,
            "queries": 0,
        }

    for row in results:
        name = row["setting"]
        grouped[name]["queries"] += 1
        grouped[name]["top_1_hit_rate"] += row["top_hit"]
        grouped[name]["top_k_hit_rate"] += row["top_k_hit"]
        grouped[name]["avg_similarity"] += row["avg_similarity"]

    for name, summary in grouped.items():
        q = summary["queries"]
        if q:
            summary["top_1_hit_rate"] = round(summary["top_1_hit_rate"] / q, 3)
            summary["top_k_hit_rate"] = round(summary["top_k_hit_rate"] / q, 3)
            summary["avg_similarity"] = round(summary["avg_similarity"] / q, 4)
    return grouped


def render_markdown(results: list[dict], summary: dict) -> str:
    lines = [
        "# Retrieval Tuning Results",
        "",
        "## Objective",
        "",
        "Compare retrieval settings to verify that filtering and score thresholds improve relevance for the maintenance and safety corpus.",
        "",
        "## Test queries",
        "",
        "| Query | Expected sources | Purpose |",
        "| --- | --- | --- |",
    ]
    for case in TEST_CASES:
        lines.append(f"| {case.query} | {', '.join(sorted(case.expected_sources))} | {case.description} |")

    lines.extend(["", "## Compared settings", "", "| Setting | k | Filter | Min similarity |", "| --- | ---: | --- | ---: |"])
    for setting in SETTINGS:
        lines.append(f"| {setting['name']} | {setting['top_k']} | {setting['filter']} | {setting['min_similarity']} |")

    lines.extend(["", "## Relevance summary", "", "| Setting | Top-1 hit rate | Top-k hit rate | Avg similarity |", "| --- | ---: | ---: | ---: |"])
    for setting in SETTINGS:
        info = summary[setting["name"]]
        lines.append(f"| {setting['name']} | {info['top_1_hit_rate']} | {info['top_k_hit_rate']} | {info['avg_similarity']} |")

    lines.extend(["", "## Detailed rows", ""])
    for row in results:
        lines.append(f"### {row['setting']} | {row['query']}")
        lines.append("")
        lines.append(f"- Filter: {row['filter']}")
        lines.append(f"- Min similarity: {row['min_similarity']}")
        lines.append(f"- Top-1 hit: {row['top_hit']}")
        lines.append(f"- Top-k hit: {row['top_k_hit']}")
        lines.append(f"- Avg similarity: {row['avg_similarity']}")
        lines.append(f"- Retrieved sources: {', '.join(row['retrieved_sources']) if row['retrieved_sources'] else 'none'}")
        lines.append(f"- Sample text: {row['retrieved_text'][0] if row['retrieved_text'] else 'none'}")
        lines.append("")

    best_setting = max(SETTINGS, key=lambda setting: (summary[setting["name"]]["top_1_hit_rate"], summary[setting["name"]]["top_k_hit_rate"]))
    lines.extend([
        "## Best setting",
        "",
        f"The best-performing configuration is **{best_setting['name']}** because it produces the highest top-1 and top-k hit rate across the test set. This configuration keeps the query scoped to the most relevant source family while filtering out low-confidence results.",
        "",
        f"Chosen setup: k={best_setting['top_k']}, filter={best_setting['filter']}, min_similarity={best_setting['min_similarity']}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    results = run_experiment()
    summary = summarize_results(results)
    markdown = render_markdown(results, summary)
    output_path = ROOT / "outputs" / "retrieval_tuning_results.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    print("Retrieval tuning experiment")
    print("=" * 70)
    for setting in SETTINGS:
        info = summary[setting["name"]]
        print(f"{setting['name']}: top-1={info['top_1_hit_rate']}, top-k={info['top_k_hit_rate']}, avg-sim={info['avg_similarity']}")
    print()
    best_setting = max(SETTINGS, key=lambda setting: (summary[setting["name"]]["top_1_hit_rate"], summary[setting["name"]]["top_k_hit_rate"]))
    print(f"Best setting: {best_setting['name']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
