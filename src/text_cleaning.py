import argparse
import re
import unicodedata
from dataclasses import replace
from pathlib import Path

try:
    from .document_intake import DocumentRecord, ingest_documents
except ImportError:
    from document_intake import DocumentRecord, ingest_documents


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving
    meaningful content and document structure.
    """

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove page-number boilerplate
    text = re.sub(
        r"Page\s+\d+\s+of\s+\d+",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove common standalone page-number patterns
    text = re.sub(
        r"^\s*Page\s+\d+\s*$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    # Collapse spaces and tabs
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove spaces at the beginning/end of lines
    text = re.sub(
        r" *\n *",
        "\n",
        text
    )

    # Collapse excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# --------------------------------------------------
# Clean one document
# --------------------------------------------------

def clean_document(document: DocumentRecord) -> DocumentRecord:
    """
    Apply the same cleaning pipeline to one document.
    """

    cleaned_text = clean_text(document.text)

    return replace(
        document,
        text=cleaned_text
    )


# --------------------------------------------------
# Clean the entire corpus
# --------------------------------------------------

def clean_documents(
    documents: list[DocumentRecord]
) -> list[DocumentRecord]:
    """
    Apply the same cleaning function to every document.
    """

    return [
        clean_document(document)
        for document in documents
    ]


# --------------------------------------------------
# Before / after display
# --------------------------------------------------

def show_before_after(
    original: DocumentRecord,
    cleaned: DocumentRecord
) -> None:

    print("\n" + "=" * 70)
    print(f"SOURCE: {original.source}")
    print("=" * 70)

    print("\nBEFORE")
    print("-" * 70)
    print(original.text[:500])

    print("\nAFTER")
    print("-" * 70)
    print(cleaned.text[:500])

    print("\nCHARACTER COUNT")
    print("-" * 70)
    print(f"Before: {len(original.text)}")
    print(f"After : {len(cleaned.text)}")
    print(f"Removed: {len(original.text) - len(cleaned.text)}")


# --------------------------------------------------
# Corpus summary
# --------------------------------------------------

def show_summary(
    original_documents: list[DocumentRecord],
    cleaned_documents: list[DocumentRecord]
) -> None:

    print("\n" + "=" * 70)
    print("TEXT CLEANING SUMMARY")
    print("=" * 70)

    print(
        f"Documents processed: "
        f"{len(original_documents)}"
    )

    print(
        f"Documents cleaned: "
        f"{len(cleaned_documents)}"
    )

    total_before = sum(
        len(document.text)
        for document in original_documents
    )

    total_after = sum(
        len(document.text)
        for document in cleaned_documents
    )

    print(
        f"Total characters before: "
        f"{total_before}"
    )

    print(
        f"Total characters after: "
        f"{total_after}"
    )

    print(
        f"Characters removed: "
        f"{total_before - total_after}"
    )

    if len(original_documents) == len(cleaned_documents):
        print(
            "Cleaning applied consistently: YES"
        )
    else:
        print(
            "Cleaning applied consistently: NO"
        )


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Clean extracted document text for "
            "downstream RAG processing."
        )
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing source documents"
    )

    return parser.parse_args()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    args = parse_args()

    data_dir = Path(args.data_dir).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}"
        )

    # ----------------------------------------------
    # Load documents using Hasini's existing loader
    # ----------------------------------------------

    documents, skipped = ingest_documents(
        data_dir
    )

    if not documents:
        print(
            "No supported documents were found "
            "in the data directory."
        )
        return

    # ----------------------------------------------
    # Clean every document
    # ----------------------------------------------

    cleaned_documents = clean_documents(
        documents
    )

    # ----------------------------------------------
    # Show before/after evidence
    # ----------------------------------------------

    for original, cleaned in zip(
        documents,
        cleaned_documents
    ):
        show_before_after(
            original,
            cleaned
        )

    # ----------------------------------------------
    # Show corpus-level results
    # ----------------------------------------------

    show_summary(
        documents,
        cleaned_documents
    )


if __name__ == "__main__":
    main()