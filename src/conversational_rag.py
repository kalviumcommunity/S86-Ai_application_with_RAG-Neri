"""Conversation-aware retrieval around the query-time RAG pipeline."""

from dataclasses import dataclass, field
from typing import Callable

try:
    from .rag_pipeline import NO_CONTEXT_ANSWER, answer_query
except ImportError:
    from rag_pipeline import NO_CONTEXT_ANSWER, answer_query


@dataclass
class ConversationTurn:
    """One user question and the grounded answer returned for it."""

    user: str
    assistant: str
    rewritten_query: str
    sources: list[dict] = field(default_factory=list)


class ConversationHistory:
    """Store the turns needed to resolve references in later questions."""

    def __init__(self):
        self.turns: list[ConversationTurn] = []

    def add(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)

    def messages(self) -> list[dict[str, str]]:
        """Return history in the role/content shape used by chat APIs."""
        messages = []
        for turn in self.turns:
            messages.extend([
                {"role": "user", "content": turn.user},
                {"role": "assistant", "content": turn.assistant},
            ])
        return messages

    @property
    def last_turn(self) -> ConversationTurn | None:
        return self.turns[-1] if self.turns else None


def default_rewrite_query(follow_up: str, history: ConversationHistory) -> str:
    """Make a simple follow-up explicit without requiring another model call.

    Applications can inject an LLM rewriter for more complex references. The
    prior user question and answer are included so the default remains useful
    for short follow-ups such as "What about the motor?".
    """
    if not follow_up.strip():
        raise ValueError("follow_up must not be empty")
    previous = history.last_turn
    if previous is None:
        return follow_up.strip()
    return (
        f"Follow-up to the question '{previous.user}': {follow_up.strip()} "
        f"Use the subject and constraints from that question."
    )


class ConversationalRAG:
    """Run grounded RAG turns while preserving rewrite and retrieval details."""

    def __init__(
        self,
        vector_store,
        embedder,
        generator,
        rewriter: Callable[[str, ConversationHistory], str] = default_rewrite_query,
        top_k: int = 4,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self.vector_store = vector_store
        self.embedder = embedder
        self.generator = generator
        self.rewriter = rewriter
        self.top_k = top_k
        self.history = ConversationHistory()

    def ask(self, question: str) -> dict:
        """Rewrite, retrieve, answer, and record one conversational turn."""
        if not question.strip():
            raise ValueError("question must not be empty")

        rewritten_query = self.rewriter(question, self.history)
        if not rewritten_query.strip():
            raise ValueError("rewriter returned an empty query")

        result = answer_query(
            rewritten_query,
            self.vector_store,
            k=self.top_k,
            embedder=self.embedder,
            generator=self.generator,
        )
        turn = ConversationTurn(
            user=question,
            assistant=result["answer"],
            rewritten_query=rewritten_query,
            sources=result["sources"],
        )
        self.history.add(turn)
        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved": bool(result["sources"]),
            "history": self.history.messages(),
        }


__all__ = [
    "ConversationHistory",
    "ConversationTurn",
    "ConversationalRAG",
    "NO_CONTEXT_ANSWER",
    "default_rewrite_query",
]