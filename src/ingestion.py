"""Run and validate the complete document ingestion pipeline."""

import argparse
from pathlib import Path

try:
    from .chunking import Chunk, token_chunks
    from .document_intake import DocumentRecord, load_text
    from .text_cleaning import clean_text
except ImportError:
    from chunking import Chunk, token_chunks
    from document_intake import DocumentRecord, load_text
    from text_cleaning import clean_text


Failure = tuple[str, str]


def ingest(
    folder: str | Path,
    token_size: int = 64,
    token_overlap: int = 16,
) -> tuple[list[Path], int, list[Chunk], list[Failure]]:
    """Load, clean, chunk, and tag every file under ``folder``.

    Each file is handled independently so one bad document cannot hide the
    status of the rest of the corpus.
    """
    data_dir = Path(folder)
    files = sorted(path for path in data_dir.rglob("*") if path.is_file())
    chunks: list[Chunk] = []
    failures: list[Failure] = []
    documents = 0

    for path in files:
        try:
            document = DocumentRecord(
                source=path.name,
                path=str(path.relative_to(data_dir)),
                text=clean_text(load_text(path)),
            )
            chunks.extend(token_chunks(document, token_size, token_overlap))
            documents += 1
        except Exception as error:
            failures.append((str(path.relative_to(data_dir)), str(error)))

    return files, documents, chunks, failures


def validate_ingestion(
    files: list[Path],
    documents: int,
    failures: list[Failure],
) -> None:
    """Raise when the run does not account for every discovered file."""
    if documents + len(failures) != len(files):
        raise AssertionError("a document was silently dropped!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--token-size", type=int, default=64)
    parser.add_argument("--token-overlap", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files, documents, chunks, failures = ingest(
        data_dir, args.token_size, args.token_overlap
    )
    validate_ingestion(files, documents, failures)
    print(
        f"files={len(files)} docs={documents} "
        f"chunks={len(chunks)} failures={len(failures)}"
    )
    for name, error in failures:
        print(f"FAILED: {name}: {error}")
    if chunks:
        print(f"sample: {chunks[0].text[:80]} | {chunks[0].metadata}")


if __name__ == "__main__":
    main()