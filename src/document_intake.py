from dataclasses import dataclass
from pathlib import Path
import argparse

from bs4 import BeautifulSoup
from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".html", ".htm"}


@dataclass(frozen=True)
class DocumentRecord:
    source: str
    path: str
    text: str


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore").strip()

    if suffix in {".html", ".htm"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)

    raise ValueError(f"unsupported file type: {suffix}")


def ingest_documents(data_dir: Path) -> tuple[list[DocumentRecord], list[tuple[str, str]]]:
    docs: list[DocumentRecord] = []
    skipped: list[tuple[str, str]] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            skipped.append((str(path), f"unsupported file type: {suffix or '<none>'}"))
            print(f"SKIP {path.name}: unsupported file type {suffix or '<none>'}")
            continue

        try:
            text = load_text(path)
            record = DocumentRecord(
                source=path.name,
                path=str(path.relative_to(data_dir)),
                text=text,
            )
            docs.append(record)
            preview = text.replace("\n", " ")[:80]
            print(f"OK {record.source}: {len(record.text)} chars | {preview!r}")
        except Exception as error:
            skipped.append((str(path), str(error)))
            print(f"SKIP {path.name}: {error}")

    return docs, skipped


def summarize(docs: list[DocumentRecord], skipped: list[tuple[str, str]]) -> None:
    print("\n" + "=" * 64)
    print("DOCUMENT INTAKE SUMMARY")
    print("=" * 64)
    print(f"Loaded documents: {len(docs)}")
    print(f"Skipped documents: {len(skipped)}")

    total_chars = sum(len(doc.text) for doc in docs)
    print(f"Total extracted characters: {total_chars}")

    if docs:
        longest = max(docs, key=lambda doc: len(doc.text))
        print(f"Longest document: {longest.source} ({len(longest.text)} chars)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load PDF, TXT, MD, and HTML files into normalized plain text "
            "for downstream RAG processing."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing source documents (default: data)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    docs, skipped = ingest_documents(data_dir)
    summarize(docs, skipped)


if __name__ == "__main__":
    main()
