# NERI

## AI-Powered Industrial Maintenance Troubleshooting Assistant

NERI is an AI-powered maintenance troubleshooting assistant designed to help manufacturing technicians quickly find reliable information when machines fail.

Manufacturing companies maintain large collections of equipment manuals, maintenance logs, and safety procedures. However, this information is often scattered across documents, making it difficult for technicians to quickly find the correct troubleshooting procedure during a machine failure.

NERI uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from approved company documents and provide technicians with source-grounded troubleshooting guidance.

---

## Problem

When a machine fails, technicians often need to search through multiple documents to find information about:

- Possible causes of the failure
- Troubleshooting procedures
- Safety precautions
- Previous maintenance incidents
- Equipment-specific instructions

Manually searching through large document collections can increase machine downtime and may lead to technicians relying on incomplete or incorrect information.

---

## Solution

NERI provides a single interface where technicians can describe a machine problem using natural language.

The system retrieves relevant information from company-approved documentation and uses it to generate a clear troubleshooting response.

The response can include:

- Possible causes
- Troubleshooting steps
- Safety warnings
- Supporting document references
- Relevant sections or pages
- Previous troubleshooting information

This allows technicians to spend less time searching through documents and more time resolving machine problems safely.

---

## How NERI Works

```text
Technician
    │
    ▼
Select Machine
    │
    ▼
Describe Machine Problem
    │
    ▼
NERI
    │
    ▼
Retrieve Relevant Documents
    │
    ▼
Generate Grounded Response
    │
    ├── Possible Causes
    │
    ├── Safety Warnings
    │
    ├── Troubleshooting Steps
    │
    └── Source References

```

## Key Features

### Natural-Language Troubleshooting

Technicians can describe machine problems in their own words instead of searching manually through technical documentation.

### Equipment-Specific Guidance

NERI can use the selected machine and relevant documentation to provide troubleshooting information specific to the equipment.

### Possible Cause Identification

The system presents potential causes of a reported machine problem based on the available documentation.

### Step-by-Step Troubleshooting

Troubleshooting guidance is presented as clear, numbered steps to make procedures easier to follow.

### Safety Warnings

Relevant safety procedures and precautions are presented prominently so technicians can consider safety requirements before performing troubleshooting actions.

### Source-Referenced Answers

NERI provides supporting document references so technicians can verify where the information came from.

### Troubleshooting History

Previous troubleshooting incidents can be reviewed to help technicians and maintenance teams understand recurring machine problems.

### Document Management

Authorized users can manage company knowledge sources such as:

- Equipment manuals
- Maintenance logs
- Safety procedures

---

## Source-Grounded AI

NERI is designed to provide answers based on company-approved documentation rather than relying only on general model knowledge.

The system follows a **Retrieval-Augmented Generation (RAG)** approach:

```text
Company Documents
        │
        ▼
Document Processing
        │
        ▼
Document Chunks
        │
        ▼
Embeddings
        │
        ▼
Vector Database
        │
        ▼
Relevant Information
        │
        ▼
Language Model
        │
        ▼
Grounded Response
        │
        ▼
Source References
```

## Document Chunking

The chunking comparison and metadata tagging are implemented in [src/chunking.py](src/chunking.py). It cleans the corpus first, then compares:

- **Fixed-size with overlap:** 500-character windows with 50 characters of overlap. This gives predictable embedding sizes and preserves text near window boundaries.
- **Paragraph:** one non-empty paragraph per chunk. This respects the meaning and structure of the manuals, logs, and safety procedure, although chunk sizes vary.

Run the comparison with:

```text
python src/chunking.py
```

## Retrieval Evaluation

## Query-Time RAG Pipeline

The query-time flow is implemented as separate, testable stages in
[src/rag_pipeline.py](src/rag_pipeline.py):

```text
User query
        -> embed_query(query)
        -> retrieve_context(query_vector, vector_store, k)
        -> assemble_context(chunks)
        -> generate_answer(query, context)
        -> answer + sources
```

`answer_query` orchestrates these stages and returns a dictionary containing
the grounded answer and the metadata for every retrieved source. The embedder
and generator are injectable, so retrieval and prompt assembly can be tested
without network access. If retrieval returns no chunks, generation is skipped
and the pipeline returns `I could not find relevant context for that question.`

`build_augmented_prompt` uses the `cl100k_base` tokenizer to keep the complete
prompt within `model_token_budget`. It reserves `answer_token_reserve` tokens
for generation, then adds retrieved chunks in rank order until the remaining
context budget is full. Every chunk is marked as `[n] Source: filename` and the
prompt instructs the model to answer only from that evidence.

Sample augmented prompt and budget report:

```text
Answer only from the provided context. Cite evidence with its [number]. If the context is insufficient, say what information is missing.

Context:
[1] Source: electrical_safety.txt (Isolation)
Isolate electrical power before opening the motor housing.

[2] Source: motor_inspection.txt (Inspection)
Record the inspection result in the maintenance log.

Question: What should I do before inspecting the motor?

Budget: model=120, prompt=86, answer reserve=20, total reserved=106, chunks=2
```

Run the focused offline tests with:

```text
python -m unittest src.rag_pipeline_test
```

An end-to-end run using the configured embedding and chat APIs can be driven
after indexing with a small script:

```python
from src.rag_pipeline import answer_query
from src.vector_store import VectorStore

result = answer_query(
        "What should I do before inspecting the motor?",
        VectorStore(persist_dir="outputs/chroma_db"),
)
print(result["answer"])
print(result["sources"])
```

Example output from the offline stage test:

```text
Power must be isolated before inspection.
[{"source": "electrical_safety.txt", "section": "Isolation"}]
```

Retrieval quality can be measured with labelled chunk IDs using the evaluator
documented in [RETRIEVAL_EVALUATION.md](RETRIEVAL_EVALUATION.md). It reports
recall and precision at top-k and retains failed queries for inspection.

Every chunk keeps its text beside the same metadata fields: `source`, `source_path`, `strategy`, `chunk_index`, `char_start`, `char_end`, and `section`. The report includes sample text plus metadata and demonstrates tracing a retrieved chunk back to its source file and exact character range. Paragraph chunking is the chosen strategy for this corpus because the source documents use short, structured sections where keeping a complete procedure or safety instruction together is more valuable than uniform chunk sizes. Fixed-size chunks remain a useful baseline for dense text and can be tuned later with retrieval tests.

Chunk sizes must also fit the model's context budget: retrieved chunks, the prompt, and the expected answer all share the context window. Increasing chunk size or top-k can improve context but can also exceed that budget and increase embedding and generation cost.

## Token-Aware Chunk Sizing

The same module also provides `token_chunks`, which uses the `cl100k_base` tokenizer rather than character length. The default is 64 tokens per chunk with 16 repeated tokens of overlap. This conservative size leaves room for the system prompt, the user question, and multiple retrieved chunks in a typical 4K+ token context window, while the 25% overlap preserves instructions that cross a boundary. The generated report includes a controlled boundary example comparing the result with and without overlap.

Run `python src/chunking.py` to regenerate the token counts and boundary demonstration. The size and overlap can be tested with `--token-size` and `--token-overlap`; smaller chunks improve precision and cost, while larger chunks provide more context but consume more of the model budget.

## Full Ingestion Validation

Run the complete load, clean, token-chunk, and metadata-tagging pipeline with:

```text
python src/ingestion.py
```

The command reports discovered files, successfully ingested documents, chunks, and per-file failures. It also asserts that every file is accounted for by either a successful document or a recorded failure, then prints one chunk with its metadata. Adjust token sizing with `--token-size` and `--token-overlap`.

## Embeddings

Generate vectors for the prepared ingestion chunks with:

```text
python src/embeddings.py
```

Set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `EMBEDDING_MODEL` in `.env` first. `EMBED_MODEL` is also supported for compatibility with the existing project configuration. The script batches requests, preserves each chunk's text and metadata beside its vector, retries transient failures with exponential backoff, and caches successful embeddings in `outputs/embedding_cache.json`. It reports total chunks, generated embeddings, skipped chunks, retries, failures, input tokens, and an approximate cost. Use `--batch-size`, `--max-retries`, `--backoff-seconds`, and `--cache` to configure the run. Set `EMBEDDING_COST_PER_1K` to override the default approximate rate of `$0.00002` per 1K input tokens.

### Similarity Ranking

Compare a question with every embedded chunk and write a ranked sample report:

```text
python src/embeddings.py --query "What should I do before inspecting the machine?" --top-k 5
```

The command uses cosine similarity, prints ranked source text and metadata, and writes the results to `outputs/similarity_ranking.md`. A high score means that a chunk is likely relevant in embedding space; it does not prove that the information is correct, current, complete, or safe to use without validation. The reusable `rank_chunks` function sorts from highest similarity to lowest similarity.

### Retrieval Sanity Checks

Run known relevance checks before trusting retrieval results:

```text
python src/sanity_test.py
```

The offline checker uses the production `rank_chunks` function with labeled fixture vectors. It verifies that electrical-safety, vibration, and PPE queries rank their known relevant chunks first, then records a deliberately broad routine-maintenance query as a surprise when it favors the PPE section. The report is written to `outputs/sanity_report.md`; the surprise demonstrates why broad queries need top-k review, metadata filters, and additional evaluation cases.

---

## Safety First

NERI is designed with safety as an important part of the troubleshooting experience.

Safety-related information should be clearly separated from general troubleshooting instructions and presented prominently when relevant.

NERI should not invent maintenance procedures or provide unsupported instructions when sufficient information is not available in the approved documentation.

When the available information is insufficient, the system should clearly communicate that it cannot provide a reliable troubleshooting procedure.

---

## Target Users

### Manufacturing Floor Technicians

The primary users of NERI. They use the system during machine failures to quickly find troubleshooting information.

### Maintenance Supervisors

Can review troubleshooting incidents and identify recurring machine problems.

### Plant Engineers

Can use the system to access equipment-specific technical information.

### Safety Officers

Can ensure that relevant safety procedures and documentation are available to technicians.

### Authorized Document Managers

Can manage the documents that form the knowledge base used by NERI.

---

## Technology

NERI is being developed as an AI-powered RAG application using technologies such as:

- Python
- Large Language Models (LLMs)
- OpenAI-compatible APIs
- ChromaDB
- Embeddings
- Retrieval-Augmented Generation (RAG)

---

## Project Architecture

```text
                    ┌─────────────────────┐
                    │      Technician     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    NERI Interface   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Retrieval      │
                    └──────────┬──────────┘
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
             Equipment     Maintenance    Safety
              Manuals         Logs       Procedures
                  │            │            │
                  └────────────┼────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Relevant Context  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Language Model   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    NERI Response    │
                    │                     │
                    │   Possible Causes   │
                    │   Safety Warnings   │
                    │   Troubleshooting   │
                    │   Sources           │
                    └─────────────────────┘
```

---

## Core Principles

NERI is built around four core principles:

### Reliable Information

Provide information based on approved company documentation.

### Safe Troubleshooting

Make relevant safety information visible before troubleshooting actions.

### Source Transparency

Allow technicians to verify the information behind the AI-generated response.

### Reduced Downtime

Help technicians find useful troubleshooting information faster during machine failures.

---

## Project Vision

NERI aims to transform how manufacturing teams access maintenance knowledge.

Instead of spending valuable time searching through thousands of documents, technicians can describe a machine problem and receive a clear, safety-conscious, source-referenced troubleshooting response.

> **NERI — Find the information. Fix the problem. Work safely.**
......

---

