"""
tests/test_generation_models.py
===============================
Focused tests for AI2-5 generation data contracts.
"""

from __future__ import annotations

from generation.models import GeneratedDocument, GeneratedSection, ReferenceCitation
from retrieval.models import SearchResult


def test_reference_citation_from_search_result_preserves_provenance():
    result = SearchResult(
        document_key="brd_leave_management_system",
        document_title="Leave Management System",
        field_id="3.7",
        field_title="Feature and Functionality",
        chunk_index=2,
        content="Managers approve or reject leave requests based on policy.",
        similarity_score=0.9123,
    )

    citation = ReferenceCitation.from_search_result("R1", result)

    assert citation.citation_id == "R1"
    assert citation.document_key == result.document_key
    assert citation.document_title == result.document_title
    assert citation.field_id == result.field_id
    assert citation.chunk_index == result.chunk_index
    assert citation.similarity_score == result.similarity_score
    assert citation.content == result.content


def test_generated_document_groups_generated_sections():
    citation = ReferenceCitation(
        citation_id="R1",
        document_key="brd_leave_management_system",
        document_title="Leave Management System",
        field_id="3.7",
        chunk_index=0,
        similarity_score=0.88,
        content="Reference content.",
    )
    section = GeneratedSection(
        field_id="3.7",
        field_title="Feature and Functionality",
        content="The system shall support leave request approval. [R1]",
        retrieved_references=(citation,),
        cited_references=(citation,),
    )

    document = GeneratedDocument(sections=(section,))

    assert document.sections == (section,)
