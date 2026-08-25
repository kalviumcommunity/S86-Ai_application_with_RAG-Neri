"""Compare chunking strategies for the cleaned NERI document corpus."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .document_intake import DocumentRecord, ingest_documents
    from .text_cleaning import clean_documents
except ImportError:
    from document_intake import DocumentRecord, ingest_documents
    from text_cleaning import clean_documents


@dataclass(frozen=True)
class Chunk:
    source: str
    source_path: str
    strategy: str
    index: int
    text: str


def fixed_size_chunks(
    document: DocumentRecord,
    size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split a document into fixed character windows with overlap."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")

    step = size - overlap
    chunks = []
    for index, start in enumerate(range(0, len(document.text), step)):
        text = document.text[start:start + size].strip()
        if text:
            chunks.append(Chunk(document.source, document.path, "fixed", index, text))
    return chunks


def paragraph_chunks(document: DocumentRecord) -> list[Chunk]:
    """Keep each non-empty paragraph as one meaning-preserving chunk."""
    paragraphs = [paragraph.strip() for paragraph in document.text.split("\n\n") if paragraph.strip()]
    return [
        Chunk(document.source, document.path, "paragraph", index, text)
        for index, text in enumerate(paragraphs)
    ]


def chunks_for_document(
    document: DocumentRecord,
    size: int = 500,
    overlap: int = 50,
) -> dict[str, list[Chunk]]:
    return {
        "fixed": fixed_size_chunks(document, size, overlap),
        "paragraph": paragraph_chunks(document),
    }


def average_size(chunks: list[Chunk]) -> float:
    return sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0.0


def build_report(
    documents: list[DocumentRecord],
    size: int,
    overlap: int,
    samples_per_strategy: int,
) -> str:
    all_chunks = {strategy: [] for strategy in ("fixed", "paragraph")}
    per_document: dict[str, dict[str, list[Chunk]]] = {}

    for document in documents:
        document_chunks = chunks_for_document(document, size, overlap)
        per_document[document.source] = document_chunks
        for strategy, chunks in document_chunks.items():
            all_chunks[strategy].extend(chunks)

    lines = [
        "# Document Chunking Comparison",
        "",
        "Parameters: fixed-size chunks use "
        f"{size} characters with {overlap} characters of overlap.",
        "",
        "## Corpus Statistics",
        "",
        "| Strategy | Chunk count | Average chunk size (characters) |",
        "| --- | ---: | ---: |",
    ]
    for strategy in ("fixed", "paragraph"):
        chunks = all_chunks[strategy]
        lines.append(f"| {strategy} | {len(chunks)} | {average_size(chunks):.1f} |")

    lines.extend(["", "## Per-Document Statistics", "", "| Source | Strategy | Chunks | Average characters |", "| --- | --- | ---: | ---: |"])
    for source, strategies in per_document.items():
        for strategy in ("fixed", "paragraph"):
            chunks = strategies[strategy]
            lines.append(f"| {source} | {strategy} | {len(chunks)} | {average_size(chunks):.1f} |")

    lines.extend(["", "## Sample Chunks", ""])
    for strategy in ("fixed", "paragraph"):
        lines.extend([f"### {strategy.title()} strategy", ""])
        for chunk in all_chunks[strategy][:samples_per_strategy]:
            lines.extend([
                f"#### {chunk.source} - chunk {chunk.index}",
                f"Source path: `{chunk.source_path}`",
                "",
                "```text",
                chunk.text,
                "```",
                "",
            ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT_DIR / "data"))
    parser.add_argument("--output", default=str(ROOT_DIR / "outputs" / "chunking_comparison.md"))
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--samples", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents, skipped = ingest_documents(Path(args.data_dir).resolve())
    cleaned = clean_documents(documents)
    if not cleaned:
        raise SystemExit("No supported documents were found.")
    if args.samples < 0:
        raise SystemExit("--samples must be non-negative")

    report = build_report(cleaned, args.size, args.overlap, args.samples)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    print(report.split("## Sample Chunks", maxsplit=1)[0].rstrip())
    print(f"\nReport written to: {output}")
    if skipped:
        print(f"Skipped documents: {len(skipped)}")


if __name__ == "__main__":
    main()