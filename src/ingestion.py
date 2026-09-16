from pathlib import Path
import re

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
}


def clean_text(text: str) -> str:
    """Clean extracted document text."""

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def extract_pdf(
    file_path: Path,
) -> list[dict]:
    """Extract text from each PDF page."""

    reader = PdfReader(
        str(file_path)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = clean_text(
            page.extract_text() or ""
        )

        if text:
            pages.append(
                {
                    "text": text,
                    "page": page_number,
                }
            )

    return pages


def extract_text_file(
    file_path: Path,
) -> list[dict]:
    """Extract text from a TXT file."""

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    text = clean_text(text)

    if not text:
        return []

    return [
        {
            "text": text,
            "page": None,
        }
    ]


def extract_document(
    file_path: Path,
) -> list[dict]:
    """Extract text from a supported document."""

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path)

    if extension == ".txt":
        return extract_text_file(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split text into overlapping chunks."""

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks


def get_document_type(
    file_path: Path,
    document_type: str | None = None,
) -> str:
    """
    Determine the Neri document type.

    If document_type is provided, use it.
    Otherwise, determine it from the parent folder.
    """

    if document_type:
        allowed_types = {
            "manual",
            "maintenance_log",
            "safety",
        }

        if document_type not in allowed_types:
            raise ValueError(
                "Invalid document type. "
                "Use manual, maintenance_log, or safety."
            )

        return document_type

    parent_folder = (
        file_path.parent.name.lower()
    )

    if parent_folder == "manuals":
        return "manual"

    if parent_folder == "maintenance_logs":
        return "maintenance_log"

    if parent_folder == "safety":
        return "safety"

    return "unknown"


def create_chunks(
    file_path: Path,
    document_type: str | None = None,
    machine: str | None = None,
    version: str | None = None,
    owner: str | None = None,
) -> list[dict]:
    """
    Extract and chunk a document while preserving metadata.
    """

    pages = extract_document(
        file_path
    )

    document_type = get_document_type(
        file_path,
        document_type,
    )

    chunks = []

    chunk_counter = 1

    for page_data in pages:

        page_chunks = chunk_text(
            page_data["text"]
        )

        for chunk in page_chunks:

            chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "document": file_path.name,
                        "document_type": document_type,
                        "machine": (
                            machine
                            if machine
                            else "unknown"
                        ),
                        "version": (
                            version
                            if version
                            else "unknown"
                        ),
                        "owner": (
                            owner
                            if owner
                            else "unknown"
                        ),
                        "section": "unknown",
                        "page": page_data["page"],
                        "chunk_id": (
                            f"{file_path.stem}_"
                            f"{chunk_counter}"
                        ),
                    },
                }
            )

            chunk_counter += 1

    return chunks