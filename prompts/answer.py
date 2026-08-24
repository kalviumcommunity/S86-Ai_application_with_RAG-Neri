"""Reusable prompt templates for grounded question answering."""

SYSTEM_TEMPLATE = (
    "You are a support assistant. "
    "Answer only from the provided context. "
    "If the answer is not in the context, say you do not know. "
    "Always cite a source."
)

ANSWER_TEMPLATE_V1 = (
    "Context:\n{context}\n\n"
    "Question: {question}"
)

ANSWER_TEMPLATE_V2 = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Return JSON only with this shape: "
    "{\"answer\": string, \"source\": string}."
)

# Backward-compatible alias used by existing scripts.
ANSWER_TEMPLATE = ANSWER_TEMPLATE_V2


def render(template, **values):
    return template.format(**values)
