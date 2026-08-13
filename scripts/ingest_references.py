#!/usr/bin/env python3
"""
scripts/ingest_references.py
=============================
Single-command operator ingestion script for approved reference BRD DOCX files.
Performs loading, canonical field parsing, field-aligned chunking, quality validation,
PostgreSQL database persistence, and FastEmbed vector embedding generation in ONE step.

Usage:
    python scripts/ingest_references.py --all
    python scripts/ingest_references.py --document 11
    python scripts/ingest_references.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# Add workspace root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from ingest.chunker import create_chunks
from ingest.loader import load_document
from ingest.parser import parse_document
from ingest.repository import ReferenceRepository
from ingest.validator import validate_ingest
from retrieval.embeddings import EmbeddingGenerator

CORPUS_PATH = ROOT / "config" / "reference_corpus.json"
SOURCE_DIR = ROOT / "data" / "reference_brds"


def load_corpus() -> list[dict]:
    config = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return config["documents"]


def find_source_file(sequence: int) -> Path:
    pattern = re.compile(rf"^\s*{sequence}\s*-\s*")
    matches = [
        p for p in SOURCE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".docx" and pattern.match(p.name)
    ]
    if not matches:
        raise FileNotFoundError(f"No DOCX found for sequence {sequence} in {SOURCE_DIR}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple DOCX files found for sequence {sequence}: {[p.name for p in matches]}")
    return matches[0]


def get_document_key(document_meta: dict) -> str:
    return document_meta.get("document_key") or document_meta["document_id"]


def process_document(
    document_meta: dict,
    dry_run: bool,
    repository: ReferenceRepository | None,
    embedder: EmbeddingGenerator | None,
) -> bool:
    sequence = document_meta["sequence"]
    source_path = find_source_file(sequence)

    print("\n" + "=" * 60)
    print(f"Document: {document_meta['title']}")
    print(f"Source: {source_path.name}")

    loaded = load_document(source_path)
    parsed = parse_document(loaded)
    document_key = get_document_key(document_meta)
    chunks = create_chunks(document_key, parsed)
    validation = validate_ingest(document_meta, parsed, chunks)

    print(f"Canonical fields: 26")
    print(f"Detected fields: {len(parsed.fields)}")
    print(f"Empty fields: {len(parsed.empty_fields)}")

    if parsed.empty_fields:
        print("  " + ", ".join(parsed.empty_fields))

    print(f"Missing fields: {len(parsed.missing_fields)}")
    if parsed.missing_fields:
        print("  " + ", ".join(parsed.missing_fields))

    print(f"Generated chunks: {len(chunks)}")

    counts = Counter(chunk.field_id for chunk in chunks)
    for field_id in sorted(
        counts,
        key=lambda v: [int(part) for part in v.split(".")],
    ):
        print(f"  {field_id} -> {counts[field_id]} chunk(s)")

    for warning in validation.warnings:
        print(f"[WARN] {warning}")

    for error in validation.errors:
        print(f"[ERROR] {error}")

    if not validation.is_valid:
        print("Result: FAILED")
        return False

    if dry_run:
        print("Database writes & Embeddings: NONE (dry-run)")
        print("Result: PASS")
        return True

    if repository is None:
        raise RuntimeError("Repository is required for non-dry-run ingestion.")

    embeddings = None
    if embedder is not None and chunks:
        print(f"Generating vector embeddings for {len(chunks)} chunk(s)...")
        texts = [f"{c.field_title}\n{c.content}" for c in chunks]
        embeddings = embedder.embed_batch(texts)

    result_meta = repository.save_document_with_chunks(
        document_meta,
        loaded,
        chunks,
        embeddings=embeddings,
    )

    print(f"Database writes: DONE (inserted {result_meta['inserted_chunks']} chunks, {result_meta['embedded_chunks']} embedded)")
    print("Result: PASS")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-command offline approved BRD reference ingestion tool."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--document", type=int, help="BRD sequence number, e.g. 11")
    target.add_argument("--all", action="store_true", help="Process all registered BRDs")
    parser.add_argument("--dry-run", action="store_true", help="Parse, chunk, validate without DB writes or embedding generation")

    args = parser.parse_args()
    corpus = load_corpus()

    if args.document is not None:
        documents = [doc for doc in corpus if doc["sequence"] == args.document]
        if not documents:
            parser.error(f"Unknown document sequence: {args.document}")
    else:
        documents = corpus

    repository = None if args.dry_run else ReferenceRepository.from_env()
    embedder = None if args.dry_run else EmbeddingGenerator()

    success = 0
    failed = 0

    for document_meta in documents:
        try:
            res = process_document(
                document_meta=document_meta,
                dry_run=args.dry_run,
                repository=repository,
                embedder=embedder,
            )
            if res:
                success += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            print(f"\n[FAILED] {document_meta['title']}: {exc}")

    print("\n" + "=" * 60)
    print(f"Summary: {success} succeeded, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
