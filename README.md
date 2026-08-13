# BRD Chatbot — Reference Ingestion & Grounded AI Retrieval System

## Overview

This repository contains the **reference document ingestion pipeline**, **pgvector semantic retrieval engine**, and **grounded LLM chat service** for the BRD Agent project.

It processes approved Business Requirement Documents (BRDs), stores field-aligned content chunks with 384-dimensional vector embeddings in PostgreSQL using `pgvector`, and provides grounded prompt context construction for LLM response generation.

---

## Project Status

| Component | Status | Description |
|---|---|---|
| **Task 1** — Canonical Field Contract & Corpus Scope | ✅ Complete | 26 answerable leaf fields, 2 structural sections, dependency matrix, 15 seed BRDs. |
| **Task 2** — DOCX Ingestion & PostgreSQL Persistence | ✅ Complete | Element-order loader, 4-path heading parser, field chunker, idempotent repository. |
| **Task 4** — Semantic Retrieval & Grounded LLM Chat | ✅ Complete | `pgvector` HNSW cosine similarity search, `fastembed` 384-dim vectors, `GroundedLLMService`. |

---

## Repository Structure

```text
brd_chatbot/
├── config/
│   ├── brd_fields.json          # Single source of truth for 26 canonical leaf fields
│   └── reference_corpus.json    # Seed BRD registrations
├── data/
│   └── reference_brds/          # 15 approved BRD DOCX files (placed here)
├── docs/
│   ├── ingest_design.md         # Ingestion architecture documentation
│   └── semantic_retrieval.md    # Vector retrieval & LLM chat specification
├── ingest/                      # TASK 1 & 2 INGESTION PIPELINE
│   ├── loader.py       # DOCX loading (paragraphs + tables in document order)
│   ├── parser.py       # Canonical field resolution (26-field contract)
│   ├── chunker.py      # Field content -> ReferenceChunk objects (<=3500 chars)
│   ├── validator.py    # Chunk validation rules
│   ├── repository.py   # PostgreSQL persistence (psycopg v3)
│   └── cli.py          # Ingestion CLI entry point
├── retrieval/                   # TASK 4 VECTOR RETRIEVAL LAYER
│   ├── embeddings.py   # FastEmbed generator (384-dim BAAI/bge-small-en-v1.5)
│   ├── store.py        # PostgresSemanticStore (pgvector cosine similarity search)
│   ├── models.py       # SearchResult model
│   └── cli.py          # Vector retrieval CLI tool
├── ai/                          # GROUNDED LLM CHAT SERVICE
│   └── client.py       # GroundedLLMService & prompt context builder
├── migrations/
│   ├── 001_create_reference_corpus.sql    # Core document and chunk tables
│   └── 002_add_pgvector.sql               # pgvector extension, vector column, HNSW index
├── scripts/
│   ├── validate_task1_config.py           # Task 1 structural validator
│   ├── embed_reference_chunks.py        # Offline vector embedding generator script
│   └── create_synthetic_fixture.py        # Test fixture generator
├── tests/                       # AUTOMATED TEST SUITE (54 PASSING TESTS)
│   ├── test_parser.py                     # Parser unit tests
│   ├── test_chunker.py                    # Chunker unit tests
│   ├── test_synthetic_future_brd.py       # Future BRD integration test
│   └── test_semantic_retrieval.py         # Semantic retrieval & LLM integration tests
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure PostgreSQL & pgvector

Copy `.env.example` to `.env` and fill in your database credentials:

```bash
cp .env.example .env
```

Apply database migrations:

```bash
psql -d brd_chatbot -f migrations/001_create_reference_corpus.sql
psql -d brd_chatbot -f migrations/002_add_pgvector.sql
```

### 3. Ingest BRDs and Generate Vector Embeddings

```bash
# 1. Ingest approved BRD documents
python3 -m ingest.cli --all

# 2. Generate and store 384-dim vector embeddings in PostgreSQL
python3 scripts/embed_reference_chunks.py
```

### 4. Query Semantic Retrieval & Grounded LLM Chat

```bash
# Semantic vector search
python3 -m retrieval.cli --query "leave management business objective" --field-id 1.2

# Grounded LLM Chat Response
python3 -m retrieval.cli --query "leave management business objective" --field-id 1.2 --llm
```

### 5. Run Automated Tests

```bash
python3 -m pytest tests/ -v
```

---

## Architecture Flow

```
DOCX File ➔ [loader.py] ➔ [parser.py] ➔ [chunker.py] ➔ [repository.py] ➔ PostgreSQL
                                                                               │
                                                                               ▼
User Query ➔ [EmbeddingGenerator] ➔ [PostgresSemanticStore] ➔ Top-K Chunks ➔ [GroundedLLMService] ➔ Grounded Answer
```
