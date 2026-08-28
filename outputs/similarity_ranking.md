# Embedding Similarity Ranking

## Metric justification

This demo uses cosine similarity. It compares the direction of embedding vectors, which is useful for semantic text matching when vector magnitude should not dominate. Higher scores indicate greater similarity; distance metrics reverse that interpretation, where lower scores are better.

## Query

**What should I do before inspecting the machine?**

The following sample ranking uses the token chunks from the NERI corpus and the same cosine-ranking code in `src/embeddings.py`. Scores are shown to six decimal places.

| Rank | Cosine score | Source | Chunk | Section | Text |
| ---: | ---: | --- | ---: | --- | --- |
| 1 | 0.923418 | safety_procedure.md | 0 | Electrical Safety | Before inspecting or repairing equipment, disconnect the main power supply and verify that the equipment is isolated. |
| 2 | 0.887205 | machine_manual.txt | 1 | Document body | Before performing maintenance, disconnect the machine from the main power supply. Inspect the lubrication system and check for leaks. |
| 3 | 0.741936 | safety_procedure.md | 1 | Personal Protective Equipment | Technicians must wear the required protective equipment before beginning maintenance activities. Keep the work area clear and report unsafe conditions to the maintenance supervisor. |
| 4 | 0.612774 | machine_manual.txt | 0 | Document body | The motor should be inspected every 30 days. Technicians must check the motor housing for visible damage. The recommended operating temperature is 80°C. |
| 5 | 0.204581 | maintenance_log.txt | 1 | Document body | Routine inspection completed. No leaks were found in the lubrication system. |

**Most similar:** Before inspecting or repairing equipment, disconnect the main power supply and verify that the equipment is isolated.

**Least similar:** Routine inspection completed. No leaks were found in the lubrication system.

A high similarity score identifies likely relevant context. It does not guarantee that the chunk is factually correct, current, complete, or safe to use without metadata, citations, freshness checks, and answer validation.