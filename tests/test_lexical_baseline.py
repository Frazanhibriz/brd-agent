"""
tests/test_lexical_baseline.py
==============================
Unit tests for AI2-3 Lexical Baseline (BM25 keyword matching).
"""

from __future__ import annotations

import pytest
from retrieval.models import LexicalDocumentChunk
from retrieval.lexical_baseline import LexicalBaseline


@pytest.fixture
def sample_corpus() -> list[LexicalDocumentChunk]:
    return [
        LexicalDocumentChunk(
            document_key="brd_ats",
            document_title="Recruitment ATS",
            field_id="1.1.1",
            field_title="Background",
            chunk_index=0,
            content="Applications come from LinkedIn, email, and employee referrals. Resumes get lost in scattered inboxes.",
        ),
        LexicalDocumentChunk(
            document_key="brd_payroll",
            document_title="Payroll Processing",
            field_id="1.1.1",
            field_title="Background",
            chunk_index=0,
            content="Manual payroll calculations cause delays and monthly overtime processing errors.",
        ),
        LexicalDocumentChunk(
            document_key="brd_leave",
            document_title="Leave Management System",
            field_id="1.2",
            field_title="Business Objective",
            chunk_index=0,
            content="Automate employee leave requests, PTO balances, and manager approvals.",
        ),
        LexicalDocumentChunk(
            document_key="brd_ats",
            document_title="Recruitment ATS",
            field_id="3.2",
            field_title="Product Specification",
            chunk_index=0,
            content="Applicant tracking system integration with external job portals and automatic resume parsing.",
        ),
    ]


def test_lexical_baseline_keyword_matching(sample_corpus):
    baseline = LexicalBaseline(sample_corpus)
    results = baseline.search_lexical(query="resumes email linkedin", top_k=2)

    assert len(results) >= 1
    assert results[0].document_key == "brd_ats"
    assert results[0].field_id == "1.1.1"
    assert results[0].similarity_score > 0.0


def test_lexical_baseline_field_filter(sample_corpus):
    baseline = LexicalBaseline(sample_corpus)
    results = baseline.search_lexical(query="tracking parsing integration", field_id="3.2", top_k=2)

    assert len(results) == 1
    assert results[0].field_id == "3.2"
    assert results[0].document_key == "brd_ats"


def test_lexical_baseline_empty_query(sample_corpus):
    baseline = LexicalBaseline(sample_corpus)
    results = baseline.search_lexical(query="", top_k=3)
    assert len(results) == 0


def test_lexical_baseline_no_matching_terms(sample_corpus):
    baseline = LexicalBaseline(sample_corpus)
    results = baseline.search_lexical(query="quantum astrophysics spaceship", top_k=3)
    assert len(results) == 0
