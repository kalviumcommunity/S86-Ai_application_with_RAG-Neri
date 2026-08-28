# Retrieval Sanity Report

This offline smoke test calls the production `rank_chunks` function with labeled fixture vectors. It checks whether known related chunks rank above unrelated chunks before retrieval is trusted.

- Test cases: 4
- Passes: 3
- Failures: 0
- Borderline or surprising cases: 1

| Case | Status | Top source | Chunk | Score | Expected source | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| What must be done before inspecting equipment? | PASS | safety_procedure.md | 0 | 1.000000 | safety_procedure.md | Electrical isolation should beat general maintenance text. |
| What should I do if the machine has unusual vibration? | PASS | machine_manual.txt | 2 | 1.000000 | machine_manual.txt | The approved response should outrank the historical maintenance log. |
| What protective equipment is required? | PASS | safety_procedure.md | 1 | 1.000000 | safety_procedure.md | The PPE section is the direct safety match. |
| Tell me about routine maintenance. | SURPRISE | safety_procedure.md | 1 | 0.953583 | machine_manual.txt | Surprise: this broad query is closer to the PPE chunk than the expected manual instruction. |

## Interpretation

The three targeted safety and vibration cases passed: the known relevant chunk ranked first. The broad routine-maintenance query was surprising because its vector was closer to the PPE chunk than the expected manual instruction. This shows that broad queries and small corpora can produce ambiguous rankings; production retrieval should use top-k context, metadata filters, and additional evaluation cases rather than trusting one score.
