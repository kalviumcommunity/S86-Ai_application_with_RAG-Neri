"""Example: Use RAG vector store for retrieval-augmented generation."""

from vector_store import VectorStore
from embeddings import embed


def retrieve_context(query: str, store: VectorStore, top_k: int = 3) -> str:
    """Retrieve relevant context for a query.

    Args:
        query: User question or query
        store: Initialized VectorStore
        top_k: Number of chunks to retrieve

    Returns:
        Formatted context string for LLM prompt
    """
    # Generate embedding for query
    query_embedding = embed([query])[0]

    # Search vector store
    results = store.search(query_embedding, top_k=top_k)

    if not results:
        return "(No relevant context found)"

    # Format results as context
    context_parts = []
    for i, result in enumerate(results, 1):
        source = result["metadata"].get("source", "Unknown")
        section = result["metadata"].get("section", "")
        section_str = f" ({section})" if section else ""
        context_parts.append(
            f"[{i}] Source: {source}{section_str}\n{result['text']}\n"
        )

    return "\n".join(context_parts)


def build_rag_prompt(query: str, context: str) -> str:
    """Build a prompt that combines user query with retrieved context.

    Args:
        query: User question
        context: Retrieved context from vector store

    Returns:
        Complete prompt for LLM
    """
    prompt = f"""Answer the following question based on the provided context.
If the context does not contain the answer, say so.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
    return prompt


# Example usage
if __name__ == "__main__":
    # Initialize vector store
    store = VectorStore(persist_dir="outputs/chroma_db")

    # Example query
    query = "How do I reset my password?"

    # Retrieve context
    print("Retrieving context...\n")
    context = retrieve_context(query, store, top_k=3)
    print("RETRIEVED CONTEXT:")
    print("-" * 50)
    print(context)
    print("-" * 50)

    # Build complete prompt for LLM
    prompt = build_rag_prompt(query, context)
    print("\nCOMPLETE PROMPT FOR LLM:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)

    # You would then send this prompt to your LLM:
    # response = llm_client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # print("\nLLM RESPONSE:")
    # print(response.choices[0].message.content)
