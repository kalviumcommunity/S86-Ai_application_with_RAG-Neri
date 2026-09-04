"""Vector database setup for RAG semantic retrieval."""

from pathlib import Path
from typing import Optional

try:
    import chromadb
except ImportError:
    raise ImportError("chromadb is required. Install with: pip install chromadb")

try:
    from .embeddings import EmbeddedChunk
except ImportError:
    from embeddings import EmbeddedChunk


# ============================================================
# CONFIGURATION
# ============================================================

# Current embedding model returns 3072-dimensional vectors.
VECTOR_DIMENSION = 3072

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
        """
        Initialize persistent Chroma vector store.

        Args:
            persist_dir: Directory containing Chroma database.
            collection_name: Chroma collection name.
            vector_dimension: Expected embedding dimension.
        """

        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": SIMILARITY_METRIC,
            },
        )

    # ========================================================
    # UPSERT
    # ========================================================

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> dict:
        """Insert or update chunks in Chroma."""

        if not embedded_chunks:
            return {
                "inserted": 0,
                "updated": 0,
                "failed": 0,
            }

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for index, chunk in enumerate(embedded_chunks):

            source = chunk.metadata.get(
                "source",
                "unknown"
            )

            chunk_index = chunk.metadata.get(
                "chunk_index",
                index
            )

            chunk_id = f"{source}:{chunk_index}"

            # Validate embedding dimension BEFORE sending to Chroma
            if len(chunk.embedding) != self.vector_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch for {chunk_id}: "
                    f"expected {self.vector_dimension}, "
                    f"got {len(chunk.embedding)}"
                )

            ids.append(chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append(chunk.metadata)

        # Let errors propagate.
        # This makes debugging much easier than silently returning failed=19.
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return {
            "inserted": len(embedded_chunks),
            "updated": 0,
            "failed": 0,
        }

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Search for semantically similar chunks."""

        if len(query_embedding) != self.vector_dimension:
            raise ValueError(
                f"Query embedding dimension "
                f"{len(query_embedding)} does not match "
                f"expected dimension {self.vector_dimension}"
            )

        if self.collection.count() == 0:
            return []

        # Never request more results than exist.
        top_k = min(top_k, self.collection.count())

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=[
                "embeddings",
                "documents",
                "metadatas",
                "distances",
            ],
        )

        formatted = []

        if results.get("ids") and results["ids"][0]:

            for i, chunk_id in enumerate(results["ids"][0]):

                embedding = []

                if (
                    results.get("embeddings") is not None
                    and len(results["embeddings"][0]) > i
                ):
                    embedding = results["embeddings"][0][i]

                text = ""

                if (
                    results.get("documents") is not None
                    and len(results["documents"][0]) > i
                ):
                    text = results["documents"][0][i]

                metadata = {}

                if (
                    results.get("metadatas") is not None
                    and len(results["metadatas"][0]) > i
                ):
                    metadata = results["metadatas"][0][i]

                distance = results["distances"][0][i]

                formatted.append(
                    {
                        "id": chunk_id,
                        "distance": distance,
                        "similarity": 1 - distance,
                        "text": text,
                        "metadata": metadata,
                        "embedding": embedding,
                    }
                )

        return formatted

    # ========================================================
    # GET
    # ========================================================

    def get(self, chunk_id: str) -> Optional[dict]:
        """Retrieve a specific chunk."""

        results = self.collection.get(
            ids=[chunk_id],
            include=[
                "embeddings",
                "documents",
                "metadatas",
            ],
        )

        if not results.get("ids"):
            return None

        embedding = []

        if (
            results.get("embeddings") is not None
            and results["embeddings"]
        ):
            embedding = results["embeddings"][0]

        text = ""

        if (
            results.get("documents") is not None
            and results["documents"]
        ):
            text = results["documents"][0]

        metadata = {}

        if (
            results.get("metadatas") is not None
            and results["metadatas"]
        ):
            metadata = results["metadatas"][0]

        return {
            "id": results["ids"][0],
            "text": text,
            "metadata": metadata,
            "embedding": embedding,
        }

    # ========================================================
    # DELETE
    # ========================================================

    def delete(self, chunk_ids: list[str]) -> dict:
        """Delete chunks by ID."""

        try:
            self.collection.delete(ids=chunk_ids)

            return {
                "deleted": len(chunk_ids),
                "failed": 0,
            }

        except Exception as error:

            return {
                "deleted": 0,
                "failed": len(chunk_ids),
                "error": str(error),
            }

    # ========================================================
    # COUNT
    # ========================================================

    def count(self) -> int:
        """Return number of indexed chunks."""

        return self.collection.count()

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self) -> None:
        """Delete all chunks from the collection."""

        results = self.collection.get(
            include=[]
        )

        ids = results.get("ids", [])

        if ids:
            self.collection.delete(ids=ids)

    # ========================================================
    # RECREATE COLLECTION
    # ========================================================

    def recreate_collection(self) -> None:
        """
        Completely delete and recreate the Chroma collection.

        Useful when the embedding dimension has changed.
        """

        try:
            self.client.delete_collection(
                name=self.collection_name
            )
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": SIMILARITY_METRIC,
            },
        )


# ============================================================
# TEST
# ============================================================

def test_vector_store() -> None:
    """Basic vector store test."""

    print("=" * 60)
    print("VECTOR STORE TEST")
    print("=" * 60)

    store = VectorStore()

    print(f"Collection : {store.collection_name}")
    print(f"Dimension  : {store.vector_dimension}")
    print(f"Count      : {store.count()}")

    print("=" * 60)


if __name__ == "__main__":
    test_vector_store()