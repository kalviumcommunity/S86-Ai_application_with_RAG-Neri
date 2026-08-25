"""Compare chunking strategies for the cleaned NERI document corpus."""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import tiktoken

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .document_intake import DocumentRecord, ingest_documents
    from .text_cleaning import clean_documents
except ImportError:
    from document_intake import DocumentRecord, ingest_documents
    from text_cleaning import clean_documents


TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class Chunk:
    source: str
    source_path: str
    strategy: str
    index: int
    text: str
    char_start: int
    char_end: int
    section: str

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "source_path": self.source_path,
            "strategy": self.strategy,
            "chunk_index": self.index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "section": self.section,
        }


def section_at(text: str, position: int) -> str:
    """Return the nearest Markdown heading, or a stable fallback section."""
    section = "Document body"
    offset = 0
    for line in text.splitlines(keepends=True):
        if offset > position:
            break
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading:
            section = heading.group(1)
        offset += len(line)
    return section


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
        raw_text = document.text[start:start + size]
        text = raw_text.strip()
        if text:
            leading_whitespace = len(raw_text) - len(raw_text.lstrip())
            char_start = start + leading_whitespace
            chunks.append(Chunk(
                document.source,
                document.path,
                "fixed",
                index,
                text,
                char_start,
                char_start + len(text),
                section_at(document.text, char_start),
            ))
    return chunks


def paragraph_chunks(document: DocumentRecord) -> list[Chunk]:
    """Keep each non-empty paragraph as one meaning-preserving chunk."""
    chunks = []
    search_from = 0
    for index, paragraph in enumerate(document.text.split("\n\n")):
        text = paragraph.strip()
        if not text:
            search_from += len(paragraph) + 2
            continue
        char_start = document.text.index(text, search_from)
        chunks.append(Chunk(
            document.source,
            document.path,
            "paragraph",
            len(chunks),
            text,
            char_start,
            char_start + len(text),
            section_at(document.text, char_start),
        ))
        search_from = char_start + len(text)
    return chunks


def token_chunks(
    document: DocumentRecord,
    size: int = 64,
    overlap: int = 16,
) -> list[Chunk]:
    """Split a document by tokenizer tokens, repeating overlap tokens."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")

    tokens = TOKENIZER.encode(document.text)
    step = size - overlap
    chunks = []
    for index, start in enumerate(range(0, len(tokens), step)):
        end = min(start + size, len(tokens))
        text = TOKENIZER.decode(tokens[start:end]).strip()
        if not text:
            continue
        approximate_start = document.text.find(text, 0 if not chunks else chunks[-1].char_start)
        char_start = approximate_start if approximate_start >= 0 else 0
        chunks.append(Chunk(
            document.source,
            document.path,
            "token",
            index,
            text,
            char_start,
            char_start + len(text),
            section_at(document.text, char_start),
        ))
        if end == len(tokens):
            break
    return chunks


def token_count(text: str) -> int:
    return len(TOKENIZER.encode(text))


def boundary_overlap_demo() -> dict[str, str | int]:
    """Create a deterministic example where overlap preserves one complete idea."""
    prefix = "Maintenance note: "
    idea = "the motor must be isolated before inspection begins"
    suffix = ". Record the result in the maintenance log."
    text = prefix + idea + suffix
    tokens = TOKENIZER.encode(text)
    idea_start = len(TOKENIZER.encode(prefix))
    idea_tokens = len(TOKENIZER.encode(idea))
    boundary = idea_start + 3
    overlap = idea_tokens - 3
    without_overlap = TOKENIZER.decode(tokens[boundary:boundary + idea_tokens])
    with_overlap = TOKENIZER.decode(tokens[boundary - overlap:boundary + idea_tokens])
    return {
        "idea": idea,
        "without_overlap": without_overlap,
        "with_overlap": with_overlap,
        "overlap_tokens": overlap,
    }


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


def trace_chunk(
    chunks: list[Chunk],
    source: str,
    strategy: str,
    index: int,
) -> Chunk:
    """Find a retrieved chunk using its source metadata."""
    for chunk in chunks:
        if chunk.source == source and chunk.strategy == strategy and chunk.index == index:
            return chunk
    raise LookupError(f"Chunk not found: {source}, {strategy}, {index}")


def build_report(
    documents: list[DocumentRecord],
    size: int,
    overlap: int,
    samples_per_strategy: int,
    token_size: int = 64,
    token_overlap: int = 16,
) -> str:
    all_chunks = {strategy: [] for strategy in ("fixed", "paragraph", "token")}
    per_document: dict[str, dict[str, list[Chunk]]] = {}

    for document in documents:
        document_chunks = chunks_for_document(document, size, overlap)
        document_chunks["token"] = token_chunks(document, token_size, token_overlap)
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
    token_results = all_chunks["token"]
    lines.append(f"| token ({token_size}/{token_overlap}) | {len(token_results)} | {average_size(token_results):.1f} |")

    lines.extend(["", "## Token Budget", "", f"Token strategy: `{token_size}` tokens per chunk with `{token_overlap}` repeated tokens of overlap, using `cl100k_base`.", "", "| Source | Chunks | Average tokens |", "| --- | ---: | ---: |"])
    for source, strategies in per_document.items():
        chunks = strategies["token"]
        lines.append(f"| {source} | {len(chunks)} | {sum(token_count(chunk.text) for chunk in chunks) / len(chunks):.1f} |")

    lines.extend(["", "## Per-Document Statistics", "", "| Source | Strategy | Chunks | Average characters |", "| --- | --- | ---: | ---: |"])
    for source, strategies in per_document.items():
        for strategy in ("fixed", "paragraph"):
            chunks = strategies[strategy]
            lines.append(f"| {source} | {strategy} | {len(chunks)} | {average_size(chunks):.1f} |")

    lines.extend(["", "## Sample Chunks", ""])
    for strategy in ("fixed", "paragraph", "token"):
        lines.extend([f"### {strategy.title()} strategy", ""])
        for chunk in all_chunks[strategy][:samples_per_strategy]:
            lines.extend([
                f"#### {chunk.source} - chunk {chunk.index}",
                f"Metadata: `{json.dumps(chunk.metadata, sort_keys=True)}`",
                "",
                "```text",
                chunk.text,
                "```",
                "",
            ])

    example = trace_chunk(all_chunks["paragraph"], documents[0].source, "paragraph", 0)
    lines.extend([
        "## Traceability Example",
        "",
        "A retrieved result can use its metadata to identify the exact source and character range:",
        "",
        f"- Metadata: `{json.dumps(example.metadata, sort_keys=True)}`",
        f"- Trace result: `{example.source_path}` -> characters "
        f"`{example.char_start}:{example.char_end}` in section **{example.section}**",
        "",
    ])

    demo = boundary_overlap_demo()
    lines.extend([
        "## Boundary Overlap Demonstration",
        "",
        f"Target idea: **{demo['idea']}**",
        "",
        "Without overlap, the boundary splits the idea:",
        f"```text\n{demo['without_overlap']}\n```",
        f"With `{demo['overlap_tokens']}` overlap tokens, the next chunk contains the complete idea:",
        f"```text\n{demo['with_overlap']}\n```",
        "",
        "The overlap preserves retrieval context when an important instruction falls across two token windows.",
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
    parser.add_argument("--token-size", type=int, default=64)
    parser.add_argument("--token-overlap", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents, skipped = ingest_documents(Path(args.data_dir).resolve())
    cleaned = clean_documents(documents)
    if not cleaned:
        raise SystemExit("No supported documents were found.")
    if args.samples < 0:
        raise SystemExit("--samples must be non-negative")

    report = build_report(
        cleaned,
        args.size,
        args.overlap,
        args.samples,
        args.token_size,
        args.token_overlap,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    print(report.split("## Sample Chunks", maxsplit=1)[0].rstrip())
    print(f"\nReport written to: {output}")
    if skipped:
        print(f"Skipped documents: {len(skipped)}")


if __name__ == "__main__":
    main()