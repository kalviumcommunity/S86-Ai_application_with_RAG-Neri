"""Vector database setup for RAG semantic retrieval."""

import os
from pathlib import Path
from typing import Optional

try:
    import chromadb
except ImportError:
    raise ImportError("chromadb is required. Install with: pip install chromadb")

from dotenv import load_dotenv

try:
    from .embeddings import EmbeddedChunk
except ImportError:
    from embeddings import EmbeddedChunk


# Configuration
VECTOR_DIMENSION = 1536  # OpenAI embedding dimension
COLLECTION_NAME = "rag_chunks"
SIMILARITY_METRIC = "cosine"


class VectorStore:
    """Manages vector storage and semantic retrieval."""

    def __init__(
        self,
        persist_dir: str | Path = "outputs/chroma_db",
        collection_name: str = COLLECTION_NAME,
        vector_dimension: int = VECTOR_DIMENSION,
    ):
        """Initialize vector store with persistent storage.

        Args:
            persist_dir: Directory for persistent Chroma database
            collection_name: Name of the collection
            vector_dimension: Embedding dimension (must match embedding model)
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension

        # Create persistent storage directory
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client with persistent storage (new API)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": SIMILARITY_METRIC},
        )

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> dict:
        """Insert or update multiple chunks in the vector store.

        Args:
            embedded_chunks: List of EmbeddedChunk objects with text, metadata, and embeddings

        Returns:
            Summary dict with counts and status
        """
        if not embedded_chunks:
            return {"inserted": 0, "updated": 0, "failed": 0}

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(embedded_chunks):
            # Create stable ID from source + chunk position
            source = chunk.metadata.get("source", "unknown")
            chunk_idx = chunk.metadata.get("chunk_index", i)
            chunk_id = f"{source}:{chunk_idx}"

            ids.append(chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append(chunk.metadata)

        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            return {"inserted": len(embedded_chunks), "updated": 0, "failed": 0}
        except Exception as e:
            return {"inserted": 0, "updated": 0, "failed": len(embedded_chunks), "error": str(e)}

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Search for semantically similar chunks.

        Args:
            query_embedding: Query vector (must be VECTOR_DIMENSION size)
            top_k: Number of top results to return
            where: Optional Chroma metadata filter

        Returns:
            List of result dicts with id, distance, text, metadata, embedding
        """
        if len(query_embedding) != self.vector_dimension:
            raise ValueError(
                f"Query embedding dimension {len(query_embedding)} "
                f"does not match collection dimension {self.vector_dimension}"
            )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["embeddings", "documents", "metadatas", "distances"],
        )

        # Format results for easier consumption
        formatted = []
        if results.get("ids") and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                embedding = results["embeddings"][0][i] if results.get("embeddings") is not None and len(results.get("embeddings", [[]])[0]) > i else []
                formatted.append({
                    "id": chunk_id,
                    "distance": results["distances"][0][i],
                    "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "text": results["documents"][0][i] if results.get("documents") is not None and len(results.get("documents", [[]])[0]) > i else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") is not None and len(results.get("metadatas", [[]])[0]) > i else {},
                    "embedding": embedding,
                })
        return formatted

    def get(self, chunk_id: str) -> Optional[dict]:
        """Retrieve a specific chunk by ID.

        Args:
            chunk_id: ID of the chunk to retrieve

        Returns:
            Dict with id, text, metadata, embedding or None if not found
        """
        results = self.collection.get(
            ids=[chunk_id],
            include=["embeddings", "documents", "metadatas"],
        )

        if not results["ids"] or len(results["ids"]) == 0:
            return None

        embedding = results["embeddings"][0] if results.get("embeddings") is not None and len(results.get("embeddings", [])) > 0 else []
        return {
            "id": results["ids"][0],
            "text": results["documents"][0] if results.get("documents") is not None and len(results.get("documents", [])) > 0 else "",
            "metadata": results["metadatas"][0] if results.get("metadatas") is not None and len(results.get("metadatas", [])) > 0 else {},
            "embedding": embedding,
        }

    def delete(self, chunk_ids: list[str]) -> dict:
        """Delete chunks by ID.

        Args:
            chunk_ids: List of IDs to delete

        Returns:
            Summary dict
        """
        try:
            self.collection.delete(ids=chunk_ids)
            return {"deleted": len(chunk_ids), "failed": 0}
        except Exception as e:
            return {"deleted": 0, "failed": len(chunk_ids), "error": str(e)}

    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all chunks from the collection."""
        # Get all IDs and delete them
        results = self.collection.get(include=[])
        if results["ids"]:
            self.collection.delete(ids=results["ids"])


def test_vector_store() -> None:
    """Verify vector store setup with insert and read-back test."""
    print("Testing Vector Store Setup...")
    print("-" * 50)

    # Initialize store
    store = VectorStore()
    print(f"✓ Connected to vector database")
    print(f"  Collection: {store.collection_name}")
    print(f"  Dimension: {store.vector_dimension}")
    print(f"  Metric: {SIMILARITY_METRIC}")

    # Create test record
    test_embedding = [0.1] * VECTOR_DIMENSION  # Dummy embedding
    test_chunk = EmbeddedChunk(
        text="Password reset instructions for learner accounts.",
        metadata={
            "source": "account-guide.md",
            "chunk_index": 0,
            "section": "Account access",
        },
        embedding=test_embedding,
    )

    # Insert test record
    store.upsert([test_chunk])
    print(f"✓ Inserted test record")

    # Read back
    chunk_id = f"account-guide.md:0"
    stored = store.get(chunk_id)

    if stored is None:
        print(f"✗ Failed to read back record")
        return

    print(f"✓ Read back verification:")
    print(f"  ID: {stored['id']}")
    print(f"  Vector length: {len(stored['embedding'])}")
    print(f"  Text: {stored['text'][:50]}...")
    print(f"  Metadata: {stored['metadata']}")

    # Verify collection count
    count = store.count()
    print(f"✓ Collection now contains: {count} chunk(s)")

    print("-" * 50)
    print("Vector store setup complete and verified!")


if __name__ == "__main__":
    test_vector_store()
