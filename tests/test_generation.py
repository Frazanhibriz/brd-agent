"""
tests/test_generation.py
========================
Focused test suite for Task 5: Citable Reference-Document Grounding and Final Document Generation.
"""

from __future__ import annotations

import pytest
from generation.generator import (
    CANONICAL_FIELD_ORDER,
    generate_final_document,
    generate_section,
)
from generation.llm_client import FakeLLMClient
from generation.models import GeneratedDocument, GeneratedSection, ReferenceCitation
from generation.prompts import build_section_generation_prompt
from generation.renderer import render_document_to_markdown, render_section_to_markdown
from retrieval.models import SearchResult


def mock_search_references(query: str, field_id: str | None = None, top_k: int = 3) -> list[SearchResult]:
    """Mock search_references function returning deterministic SearchResult items for tests."""
    return [
        SearchResult(
            document_key="brd_leave_management_system",
            document_title="Leave Management System",
            field_id=field_id or "3.7",
            field_title="Feature and Functionality",
            chunk_index=0,
            content="Leave request workflow description chunk.",
            similarity_score=0.91,
        ),
        SearchResult(
            document_key="brd_payroll_automation",
            document_title="Payroll Automation System",
            field_id=field_id or "3.7",
            field_title="Feature and Functionality",
            chunk_index=1,
            content="Payroll processing workflow chunk.",
            similarity_score=0.85,
        ),
    ][:top_k]


# ---------------------------------------------------------------------------
# Test 1 & 4: Valid canonical section generation & field targeting
# ---------------------------------------------------------------------------
def test_generate_section_valid_canonical_field():
    fake_client = FakeLLMClient(
        canned_response="The system shall provide invoice generation within 1 hour. [R1]"
    )
    sec = generate_section(
        field_id="1.1.1",
        confirmed_information="The system shall support automated background invoice processing.",
        search_fn=mock_search_references,
        llm_client=fake_client,
    )

    assert isinstance(sec, GeneratedSection)
    assert sec.field_id == "1.1.1"
    assert sec.field_title == "Background"
    assert "1 hour" in sec.content
    assert sec.is_unresolved is False
    assert len(sec.retrieved_references) == 2
    assert len(sec.cited_references) == 1
    assert sec.cited_references[0].citation_id == "R1"
    assert sec.cited_references[0].document_title == "Leave Management System"


# ---------------------------------------------------------------------------
# Test 2, 3 & 14: Rejection of field_id=None, structural, and invalid field IDs
# ---------------------------------------------------------------------------
def test_generate_section_rejects_none_field_id():
    fake_client = FakeLLMClient()
    with pytest.raises(ValueError, match="field_id is required"):
        generate_section(
            field_id=None,  # type: ignore[arg-type]
            confirmed_information="Some confirmed info.",
            search_fn=mock_search_references,
            llm_client=fake_client,
        )


def test_generate_section_rejects_structural_section():
    fake_client = FakeLLMClient()
    with pytest.raises(ValueError, match="structural header section"):
        generate_section(
            field_id="1.1",  # Structural section ID
            confirmed_information="Some overview info.",
            search_fn=mock_search_references,
            llm_client=fake_client,
        )


def test_generate_section_rejects_invalid_field_id():
    fake_client = FakeLLMClient()
    with pytest.raises(ValueError, match="Invalid or non-answerable field_id"):
        generate_section(
            field_id="9.9.9",
            confirmed_information="Some invalid field info.",
            search_fn=mock_search_references,
            llm_client=fake_client,
        )


# ---------------------------------------------------------------------------
# Test 5 & 6: Search references query, field_id passing, and top_k parameter
# ---------------------------------------------------------------------------
def test_search_references_called_with_exact_field_id_and_top_k():
    captured_calls = []

    def spy_search(query: str, field_id: str | None = None, top_k: int = 3) -> list[SearchResult]:
        captured_calls.append({"query": query, "field_id": field_id, "top_k": top_k})
        return mock_search_references(query, field_id, top_k)

    fake_client = FakeLLMClient(canned_response="Generated content [R1].")
    generate_section(
        field_id="3.3.2",
        confirmed_information="System must process orders.",
        conversation_context="Additional order context.",
        search_fn=spy_search,
        llm_client=fake_client,
        top_k=3,
    )

    assert len(captured_calls) == 1
    assert captured_calls[0]["field_id"] == "3.3.2"
    assert captured_calls[0]["top_k"] == 3
    assert "System must process orders." in captured_calls[0]["query"]
    assert "Additional order context." in captured_calls[0]["query"]


# ---------------------------------------------------------------------------
# Test 7, 8 & 9: Citation mapping, provenance, and rejection of invented citation
# ---------------------------------------------------------------------------
def test_provenance_preserved_and_invented_citation_rejected():
    # Valid output citing R1
    fake_client_valid = FakeLLMClient(canned_response="Requirements overview [R1].")
    sec_valid = generate_section(
        field_id="1.2",
        confirmed_information="Business objective description.",
        search_fn=mock_search_references,
        llm_client=fake_client_valid,
    )
    assert len(sec_valid.retrieved_references) == 2
    assert len(sec_valid.cited_references) == 1
    assert sec_valid.cited_references[0].citation_id == "R1"

    # Invalid output citing invented citation R7
    fake_client_invalid = FakeLLMClient(canned_response="Requirements overview [R7].")
    with pytest.raises(ValueError, match="Model generated invalid citation: \\[R7\\]"):
        generate_section(
            field_id="1.2",
            confirmed_information="Business objective description.",
            search_fn=mock_search_references,
            llm_client=fake_client_invalid,
        )


# ---------------------------------------------------------------------------
# Test 10: Empty/insufficient confirmed information handling
# ---------------------------------------------------------------------------
def test_empty_confirmed_information_returns_unresolved_section():
    fake_client = FakeLLMClient()
    sec = generate_section(
        field_id="1.1.1",
        confirmed_information="",  # Empty confirmed info
        search_fn=mock_search_references,
        llm_client=fake_client,
    )

    assert sec.is_unresolved is True
    assert sec.content == "[Content unresolved - No confirmed information provided for this section]"
    assert sec.retrieved_references == ()
    assert sec.cited_references == ()
    assert len(fake_client.calls) == 0  # No LLM call made!


# ---------------------------------------------------------------------------
# Test 11 & 17: Prompt separation & Anti-hallucination authority hierarchy
# ---------------------------------------------------------------------------
def test_anti_hallucination_authority_hierarchy_in_prompt():
    ref = ReferenceCitation(
        citation_id="R1",
        document_key="brd_invoice_v1",
        document_title="Invoice System BRD",
        field_id="3.3.2",
        chunk_index=0,
        similarity_score=0.9,
        content="Invoice SLA must be 2 hours.",
    )

    sys_inst, user_prompt = build_section_generation_prompt(
        field_id="3.3.2",
        field_title="Invoice Processing",
        big_question="What is the invoice SLA?",
        information_needed="Detailed SLA parameters.",
        confirmed_information="Invoice SLA is 1 hour.",
        conversation_context="Discussion about potential SLA options.",
        references=[ref],
    )

    # Verify authority order and explicitly separated sections
    assert "=== 1. CONFIRMED PROJECT INFORMATION (AUTHORITATIVE TRUTH) ===" in user_prompt
    assert "Invoice SLA is 1 hour." in user_prompt
    assert "=== 2. CONVERSATION CONTEXT (NON-AUTHORITATIVE DISCUSSION) ===" in user_prompt
    assert "Discussion about potential SLA options." in user_prompt
    assert "=== 3. RETRIEVED REFERENCE BRDs (KNOWLEDGE & GROUNDING ONLY) ===" in user_prompt
    assert "[R1] Document: 'Invoice System BRD'" in user_prompt
    assert "Invoice SLA must be 2 hours." in user_prompt

    # Verify anti-hallucination rules present in system instruction
    assert "STRICT SOURCE-AUTHORITY HIERARCHY" in sys_inst
    assert "CONFIRMED PROJECT INFORMATION: Absolute, authoritative truth" in sys_inst
    assert "Do NOT replace it with values found in Reference BRDs" in sys_inst


# ---------------------------------------------------------------------------
# Test 12 & 13: Final document generation and canonical field ordering
# ---------------------------------------------------------------------------
def test_generate_final_document_canonical_ordering_and_assembly():
    confirmed_sections = {
        "1.1.1": "The company requires a new ERP integration.",
        "1.2": "Objective is to reduce invoice processing time by 50%.",
    }

    fake_client = FakeLLMClient(
        canned_response="Section requirement implementation [R1]."
    )

    doc = generate_final_document(
        confirmed_sections=confirmed_sections,
        search_fn=mock_search_references,
        llm_client=fake_client,
    )

    assert isinstance(doc, GeneratedDocument)
    assert len(doc.sections) == 26  # All 26 canonical answerable fields present

    # Verify exact canonical order matches CANONICAL_FIELD_ORDER
    doc_field_ids = [sec.field_id for sec in doc.sections]
    assert doc_field_ids == CANONICAL_FIELD_ORDER

    # Verify sections with confirmed info are resolved, others are unresolved
    sec_1_1_1 = next(s for s in doc.sections if s.field_id == "1.1.1")
    assert sec_1_1_1.is_unresolved is False

    sec_3_3_2 = next(s for s in doc.sections if s.field_id == "3.3.2")
    assert sec_3_3_2.is_unresolved is True


# ---------------------------------------------------------------------------
# Markdown Renderer Tests
# ---------------------------------------------------------------------------
def test_markdown_renderer():
    sec = GeneratedSection(
        field_id="1.1.1",
        field_title="Background",
        content="The system shall automate workflows.",
        cited_references=(
            ReferenceCitation(
                citation_id="R1",
                document_key="brd_lms",
                document_title="LMS BRD",
                field_id="1.1.1",
                chunk_index=0,
                similarity_score=0.9,
                content="LMS content.",
            ),
        ),
    )

    md_sec = render_section_to_markdown(sec)
    assert "### Section 1.1.1: Background" in md_sec
    assert "The system shall automate workflows." in md_sec
    assert "**Grounding References:**" in md_sec
    assert "- [R1] LMS BRD (`brd_lms`, Chunk 0) - Section 1.1.1" in md_sec

    doc = GeneratedDocument(sections=(sec,))
    md_doc = render_document_to_markdown(doc)
    assert "# Business Requirements Document (BRD)" in md_doc
    assert "### Section 1.1.1: Background" in md_doc
