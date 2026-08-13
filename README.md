# BRD Chatbot — Reference Ingestion & Grounded AI Retrieval System

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15%2B-blue)
![pgvector](https://img.shields.io/badge/pgvector-0.5.0%2B-green)
![Tests](https://img.shields.io/badge/tests-15%20passed-brightgreen)

## Overview

This repository contains the **reference document ingestion pipeline**, **pgvector semantic retrieval engine**, and **grounded AI retrieval service** for the BRD Agent project.

It processes approved Business Requirement Documents (BRDs), maps content to **26 canonical leaf fields**, stores field-aligned chunks with **384-dimensional vector embeddings** (FastEmbed `BAAI/bge-small-en-v1.5`) in PostgreSQL using `pgvector`, and provides grounded retrieval context.

---

## Repository Structure

```text
brd_chatbot/
├── config/
│   ├── brd_fields.json          # Single source of truth for 26 canonical leaf fields
│   └── reference_corpus.json    # Seed BRD registrations (15 reference documents)
├── data/
│   └── reference_brds/          # 15 approved BRD DOCX files
├── ingest/                      # INGESTION & PARSING PIPELINE
│   ├── loader.py       # Membaca paragraf + tabel .docx sesuai urutan dokumen
│   ├── parser.py       # Pencocokan judul ke 26 canonical fields (Regex + Fuzzy)
│   ├── chunker.py      # Field content -> ReferenceChunk (target 1200-2000 chars)
│   ├── validator.py    # Chunk & field quality validation rules
│   └── repository.py   # PostgreSQL persistence (psycopg v3)
├── retrieval/                   # VECTOR RETRIEVAL LAYER
│   ├── embeddings.py   # FastEmbed generator (384-dim BAAI/bge-small-en-v1.5)
│   ├── semantic.py     # PostgresSemanticStore (pgvector cosine similarity search)
│   ├── lexical_baseline.py # Lexical/BM25 baseline for comparison
│   └── models.py       # Unified Dataclass Models (LoadedDocument, ReferenceChunk, SearchResult)
├── generation/                  # LLM GENERATION & ORCHESTRATION LAYER
│   ├── generator.py    # Core generation engine & field-scoped context injection
│   ├── llm_client.py   # LLM interface wrappers (FakeLLMClient, etc.)
│   ├── models.py       # Dataclass models (GeneratedSection, GeneratedDocument)
│   ├── prompts.py      # System instructions & dynamic prompt builders
│   └── renderer.py     # Markdown rendering for generated sections
├── migrations/
│   ├── 001_create_reference_corpus.sql    # Core document and chunk tables
│   └── 002_add_pgvector.sql               # pgvector extension, vector column, HNSW index
├── scripts/
│   ├── ingest_references.py               # Single-command operator ingestion CLI
│   ├── evaluate_retrieval.py              # Retrieval accuracy evaluation benchmark
│   └── demo_generation.py                 # Standalone generation subsystem demo
├── tests/                       # AUTOMATED TEST SUITE (15 PASSING TESTS)
│   ├── test_ingest.py                     # Parser, chunker, & repo unit tests
│   ├── test_lexical_baseline.py           # Lexical search tests
│   ├── test_relevance.py                  # Retrieval relevance tests
│   ├── test_retrieval.py                  # Retrieval function contract tests
│   └── test_synthetic_future_brd.py       # Integration test for future BRD (Seq 16)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

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
psql -h localhost -U postgres -d brd_chatbot -f migrations/001_create_reference_corpus.sql
psql -h localhost -U postgres -d brd_chatbot -f migrations/002_add_pgvector.sql
```

### 3. Ingest BRDs and Generate Vector Embeddings

```bash
# Dry-run validation (without DB writes):
python3 scripts/ingest_references.py --all --dry-run

# Production ingestion (loads, chunks, embeds 384-dim vectors, & saves to PostgreSQL):
python3 scripts/ingest_references.py --all
```

### 4. Run Automated Tests

```bash
pytest tests/ -v
```

### 5. Run Generation Demo

To see the AI generation subsystem in action with mock LLM responses:

```bash
python3 scripts/demo_generation.py
```

---

## Architecture Flow

```text
DOCX File ➔ [loader.py] ➔ [parser.py] ➔ [chunker.py] ➔ [repository.py] ➔ PostgreSQL
                                                                                │
                                                                                ▼
User Query ➔ [EmbeddingGenerator] ➔ [PostgresSemanticStore] ➔ Top-K Chunks ➔ [generator.py] ➔ Grounded Section
```
