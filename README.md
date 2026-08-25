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

The chunking comparison is implemented in [src/chunking.py](src/chunking.py). It cleans the corpus first, then compares:

- **Fixed-size with overlap:** 500-character windows with 50 characters of overlap. This gives predictable embedding sizes and preserves text near window boundaries.
- **Paragraph:** one non-empty paragraph per chunk. This respects the meaning and structure of the manuals, logs, and safety procedure, although chunk sizes vary.

Run the comparison with:

```text
python src/chunking.py
```

The command reports the chunk count and average character size for each strategy and writes sample chunks with source paths to [outputs/chunking_comparison.md](outputs/chunking_comparison.md). Paragraph chunking is the chosen strategy for this corpus because the source documents use short, structured sections where keeping a complete procedure or safety instruction together is more valuable than uniform chunk sizes. Fixed-size chunks remain a useful baseline for dense text and can be tuned later with retrieval tests.

Chunk sizes must also fit the model's context budget: retrieved chunks, the prompt, and the expected answer all share the context window. Increasing chunk size or top-k can improve context but can also exceed that budget and increase embedding and generation cost.

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

---

