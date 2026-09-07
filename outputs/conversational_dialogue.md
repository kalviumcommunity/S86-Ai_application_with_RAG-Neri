# Conversational RAG Dialogue

This example uses the indexed maintenance corpus and keeps both user and assistant turns in `ConversationHistory`.

## Turn 1

**User:** What should I do before inspecting the motor?

**Rewritten retrieval query:** What should I do before inspecting the motor?

**Retrieved context:**

```text
[1] Source: electrical_safety.txt (Isolation)
Isolate electrical power before opening or inspecting the motor housing.
```

**Assistant:** Isolate electrical power before inspecting the motor. [1]

## Turn 2

**User:** What about safety?

**Rewritten retrieval query:** What safety precautions apply before inspecting the motor?

The rewrite resolves the follow-up's missing subject, "the motor", from the first turn before embedding and retrieval.

**Retrieved context:**

```text
[1] Source: electrical_safety.txt (Isolation)
Isolate electrical power before opening or inspecting the motor housing.
```

**Assistant:** Isolate electrical power before inspecting the motor. [1]

The second answer is grounded in the context retrieved with the rewritten query, not with the raw question "What about safety?".