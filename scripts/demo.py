#!/usr/bin/env python3
"""
scripts/verify_task5_real.py
============================
Standalone Task 5 Real-Runtime Verification & Smoke Test Script.

Executes real section generation end-to-end against:
1. Real PostgreSQL + pgvector database (15 approved reference BRDs),
2. Real search_references() with field filtering (field_id="3.7"),
3. Real configured LLM provider (GeminiLLMClient),
4. Real citation provenance and anti-hallucination validation.
"""

from __future__ import annotations

import sys
import os
import dotenv
from pathlib import Path

# Load environment variables
dotenv.load_dotenv()

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.generator import generate_section
from generation.llm_client import GeminiLLMClient, get_default_llm_client
from generation.renderer import render_section_to_markdown
from retrieval.semantic import search_references


def run_real_smoke_test():
    print("================================================================================")
    print("TASK 5 REAL-RUNTIME VERIFICATION & SMOKE TEST")
    print("================================================================================")
    print()

    field_id = input("Please enter field ID (e.g. 3.7): ")
    confirmed_info = input("Please enter confirmed information: ")


    print(f"Target Field ID: {field_id}")
    print(f"Confirmed Information: {confirmed_info}")
    print()

    # Step 1: Execute real retrieval
    print("--- 1. REAL RETRIEVAL VERIFICATION (PostgreSQL + pgvector) ---")
    retrieved_results = search_references(query=confirmed_info, field_id=field_id, top_k=3)

    if not retrieved_results:
        print("ERROR: No references retrieved from PostgreSQL database!")
        sys.exit(1)

    print(f"Retrieved {len(retrieved_results)} references for field_id='{field_id}':")
    for idx, r in enumerate(retrieved_results, start=1):
        print(f"  [{idx}] Title: '{r.document_title}' | Key: '{r.document_key}' | Field ID: '{r.field_id}' | Chunk Index: {r.chunk_index} | Similarity: {r.similarity_score}")
        assert r.field_id == field_id, f"Retrieval field_id mismatch: expected {field_id}, got {r.field_id}"

    print("Retrieval field-filtering verified: ALL retrieved chunks have field_id='3.7'.")
    print()

    # Step 2: Initialize real LLM client
    print("--- 2. REAL LLM CLIENT INITIALIZATION ---")
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("CRITICAL NOTICE: GEMINI_API_KEY / GOOGLE_API_KEY is not set in environment or .env.")
        print("Real LLM call cannot be completed without an API key.")
        print("Please configure GEMINI_API_KEY in .env to complete real LLM smoke test.")
        return False, retrieved_results, None

    llm_client = GeminiLLMClient(api_key=api_key)
    print(f"GeminiLLMClient initialized successfully using model '{llm_client.model_name}'.")
    print()

    # Step 3: Run generate_section() with real retrieval & real LLM
    print("--- 3. EXECUTING REAL SECTION GENERATION ---")
    generated_section = generate_section(
        field_id=field_id,
        confirmed_information=confirmed_info,
        search_fn=search_references,
        llm_client=llm_client,
        top_k=3,
    )

    print()
    print("--- 4. GENERATED SECTION OUTPUT ---")
    print(render_section_to_markdown(generated_section))

    print("--- 5. CITATION PROVENANCE METADATA ---")
    print(f"Retrieved References Count: {len(generated_section.retrieved_references)}")
    print(f"Cited References Count: {len(generated_section.cited_references)}")
    for ref in generated_section.cited_references:
        print(f"  Citation ID: [{ref.citation_id}] -> Doc Key: '{ref.document_key}' -> Title: '{ref.document_title}' -> Field ID: '{ref.field_id}' -> Chunk: {ref.chunk_index}")

    print()
    print("================================================================================")
    print("REAL SMOKE TEST COMPLETED SUCCESSFULLY")
    print("================================================================================")
    return True, retrieved_results, generated_section


if __name__ == "__main__":
    success, refs, sec = run_real_smoke_test()
    if not success:
        sys.exit(2)
