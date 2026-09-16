from src.embeddings import create_embedding
from src.vector_store import search_documents


# ============================================================
# Helper
# ============================================================

def _convert_results(
    results: dict,
) -> list[dict]:
    """
    Convert ChromaDB results into Neri's
    internal retrieval format.
    """

    documents = results.get(
        "documents",
        [[]],
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]],
    )[0]

    distances = results.get(
        "distances",
        [[]],
    )[0]

    retrieved_chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        retrieved_chunks.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return retrieved_chunks


# ============================================================
# Main Retrieval
# ============================================================

def retrieve_documents(
    query: str,
    machine: str | None = None,
    n_results: int = 5,
) -> list[dict]:
    """
    Retrieve approved documentation for a troubleshooting query.

    Normal retrieval:
        Searches documentation assigned to the selected machine.

    Safety-critical retrieval:
        In addition to normal retrieval, explicitly searches
        the selected machine's approved safety documentation.

    This ensures that safety evidence is not missed simply
    because a safety document did not appear in the normal
    semantic top-k results.
    """

    if not query.strip():
        return []

    query_embedding = create_embedding(
        query
    )

    # ========================================================
    # Normal Machine Retrieval
    # ========================================================

    results = search_documents(
        embedding=query_embedding,
        n_results=n_results,
    )

    normal_chunks = _convert_results(
        results
    )

    # Strict machine filtering.
    if machine:

        filtered_chunks = []

        for chunk in normal_chunks:

            metadata = chunk.get(
                "metadata",
                {},
            )

            chunk_machine = metadata.get(
                "machine",
                "unknown",
            )

            if (
                chunk_machine.lower()
                == machine.lower()
            ):
                filtered_chunks.append(
                    chunk
                )

        normal_chunks = filtered_chunks

    # ========================================================
    # Explicit Safety Retrieval
    # ========================================================

    safety_chunks = []

    if machine:

        safety_where = {
            "$and": [
                {
                    "machine": {
                        "$eq": machine
                    }
                },
                {
                    "document_type": {
                        "$eq": "safety"
                    }
                },
            ]
        }

        safety_results = search_documents(
            embedding=query_embedding,
            n_results=3,
            where=safety_where,
        )

        safety_chunks = _convert_results(
            safety_results
        )

    # ========================================================
    # Combine Results
    # ========================================================

    combined_chunks = []

    seen_chunk_ids = set()

    # Add normal results first.
    for chunk in normal_chunks:

        chunk_id = chunk.get(
            "metadata",
            {},
        ).get(
            "chunk_id"
        )

        if chunk_id not in seen_chunk_ids:

            combined_chunks.append(
                chunk
            )

            seen_chunk_ids.add(
                chunk_id
            )

    # Add explicit safety results.
    for chunk in safety_chunks:

        chunk_id = chunk.get(
            "metadata",
            {},
        ).get(
            "chunk_id"
        )

        if chunk_id not in seen_chunk_ids:

            combined_chunks.append(
                chunk
            )

            seen_chunk_ids.add(
                chunk_id
            )

    return combined_chunks