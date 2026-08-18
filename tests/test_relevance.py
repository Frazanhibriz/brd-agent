"""
tests/test_relevance.py
========================
Automated test suite for AI2-4B semantic retrieval relevance evaluation.
Validates loading of simulated Chat-style evaluation dataset, Hit@1/Hit@3 calculation,
and pgvector retrieval accuracy.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from dotenv import load_dotenv

load_dotenv()

from retrieval.models import SearchResult
from retrieval.semantic import PostgresSemanticStore, search_references

ROOT = Path(__file__).resolve().parents[1]
EVAL_DATASET_PATH = ROOT / "tests" / "fixtures" / "simulated_chat_eval_dataset.json"


def test_search_references_function_contract():
    mock_store = MagicMock(spec=PostgresSemanticStore)
    mock_store.search_references.return_value = [
        SearchResult(
            document_key="brd_guest_checkout",
            document_title="Guest Checkout",
            field_id="3.2",
            field_title="Product Specification",
            chunk_index=0,
            content="Payment gateway integration details.",
            similarity_score=0.88,
        )
    ]

    results = search_references("payment gateway", field_id="3.2", top_k=3, store=mock_store)
    assert len(results) == 1
    assert results[0].field_id == "3.2"
    mock_store.search_references.assert_called_with(query="payment gateway", field_id="3.2", top_k=3)



def is_db_available() -> bool:
    required = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER")
    if not all(os.environ.get(k) for k in required):
        return False
    try:
        store = PostgresSemanticStore.from_env()
        results = store.search_semantic(query="test", top_k=1)
        return len(results) > 0
    except Exception:
        return False


def test_eval_dataset_fixture_valid():
    assert EVAL_DATASET_PATH.exists()
    dataset = json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))
    assert isinstance(dataset, list)
    assert len(dataset) >= 15

    for item in dataset:
        assert "id" in item
        assert "query" in item
        assert "expected_document_keys" in item


@pytest.mark.skipif(not is_db_available(), reason="PostgreSQL semantic store with embeddings unavailable.")
def test_ai2_4b_relevance_evaluation_metrics():
    dataset = json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))
    store = PostgresSemanticStore.from_env()

    hit_1 = 0
    hit_3 = 0

    for item in dataset:
        query = item["query"]
        field_id = item.get("field_id")
        expected_keys = set(item["expected_document_keys"])

        results = store.search_references(
            query=query,
            field_id=field_id,
            top_k=3,
        )

        retrieved_keys = [r.document_key for r in results]

        for idx, key in enumerate(retrieved_keys, start=1):
            if key in expected_keys:
                if idx == 1:
                    hit_1 += 1
                if idx <= 3:
                    hit_3 += 1
                break

    hit_3_rate = hit_3 / len(dataset)
    hit_1_rate = hit_1 / len(dataset)

    assert hit_3_rate >= 0.90, f"Hit@3 rate {hit_3_rate:.2f} fell below 90% threshold"
    assert hit_1_rate >= 0.80, f"Hit@1 rate {hit_1_rate:.2f} fell below 80% threshold"
