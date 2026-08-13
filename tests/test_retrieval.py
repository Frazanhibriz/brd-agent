"""
tests/test_retrieval.py
========================
Unit tests for AI2-4 search_references contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from retrieval.models import SearchResult
from retrieval.semantic import PostgresSemanticStore, search_references


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
