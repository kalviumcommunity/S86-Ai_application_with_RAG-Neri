def build_sources(
    retrieved_chunks: list[dict],
) -> list[dict]:
    """Build source references for the Neri response."""

    sources = []

    for source_id, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        metadata = chunk["metadata"]

        sources.append(
            {
                "id": source_id,
                "document": metadata.get(
                    "document",
                    "Unknown document",
                ),
                "document_type": metadata.get(
                    "document_type",
                    "unknown",
                ),
                "section": metadata.get(
                    "section",
                    "unknown",
                ),
                "page": metadata.get(
                    "page"
                ),
            }
        )

    return sources