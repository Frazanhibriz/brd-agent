"""
tests/test_generation.py
========================
Focused test suite for Task 5: Citable Reference-Document Grounding and Final Document Generation.
"""

from __future__ import annotations

import json
import pytest
from generation.generator import (
    CANONICAL_FIELD_ORDER,
    UnsafeGenerationError,
    generate_final_document,
    generate_section,
)
from generation.llm_client import FakeLLMClient
from generation.models import GeneratedDocument, GeneratedSection, ReferenceCitation
from generation.prompts import build_section_generation_prompt, extract_canonical_gaps
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
        canned_response=json.dumps({
            "requirements": [
                {
                    "text": "The system shall provide invoice generation within 1 hour.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R1"]
                }
            ],
            "unresolved_gap_ids": []
        })
    )
    sec = generate_section(
        field_id="1.1.1",
        confirmed_information="The system shall support automated background invoice processing within 1 hour.",
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

    fake_client = FakeLLMClient(
        canned_response=json.dumps({
            "requirements": [
                {
                    "text": "System must process orders.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R1"]
                }
            ],
            "unresolved_gap_ids": []
        })
    )
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
    fake_client_valid = FakeLLMClient(
        canned_response=json.dumps({
            "requirements": [
                {
                    "text": "Requirements overview.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R1"]
                }
            ],
            "unresolved_gap_ids": []
        })
    )
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
    fake_client_invalid = FakeLLMClient(
        canned_response=json.dumps({
            "requirements": [
                {
                    "text": "Requirements overview.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R7"]
                }
            ],
            "unresolved_gap_ids": []
        })
    )
    with pytest.raises(ValueError, match="R7"):
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
    assert "=== 1. CONFIRMED PROJECT EVIDENCE (AUTHORITATIVE TRUTH - C* IDENTIFIERS) ===" in user_prompt
    assert "Invoice SLA is 1 hour." in user_prompt
    assert "=== 2. CONVERSATION CONTEXT (NON-AUTHORITATIVE DISCUSSION) ===" in user_prompt
    assert "Discussion about potential SLA options." in user_prompt
    assert "=== 3. RETRIEVED REFERENCE BRDs (NON-AUTHORITATIVE GROUNDING ONLY - R* IDENTIFIERS) ===" in user_prompt
    assert "[R1] Document: 'Invoice System BRD'" in user_prompt
    assert "Invoice SLA must be 2 hours." in user_prompt
    assert "=== 4. CANONICAL FIELD GAPS (SELECTABLE G* IDENTIFIERS FOR UNRESOLVED ITEMS) ===" in user_prompt

    # Verify anti-hallucination rules present in system instruction
    assert "STRICT EVIDENCE, GROUNDING & GAP RULES" in sys_inst
    assert "C* identifiers" in sys_inst
    assert "R* identifiers" in sys_inst
    assert "G* identifiers" in sys_inst


# ---------------------------------------------------------------------------
# Test 12 & 13: Final document generation and canonical field ordering
# ---------------------------------------------------------------------------
def test_generate_final_document_canonical_ordering_and_assembly():
    confirmed_sections = {
        "1.1.1": "The company requires a new ERP integration.",
        "1.2": "Objective is to reduce invoice processing time by 50%.",
    }

    fake_client = FakeLLMClient(
        canned_response=json.dumps({
            "requirements": [
                {
                    "text": "Section requirement implementation.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R1"]
                }
            ],
            "unresolved_gap_ids": []
        })
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


# ---------------------------------------------------------------------------
# Task 5 Hardening & anti-hallucination focused tests
# ---------------------------------------------------------------------------
def test_confirmed_information_conflict_beats_reference_value():
    """Verify that confirmed project information beats conflicting reference values."""
    ref = ReferenceCitation(
        citation_id="R1",
        document_key="brd_invoice_v1",
        document_title="Invoice System BRD",
        field_id="3.3.2",
        chunk_index=0,
        similarity_score=0.9,
        content="Invoice generation must be completed within 2 hours.",
    )

    sys_inst, user_prompt = build_section_generation_prompt(
        field_id="3.3.2",
        field_title="Transaction Processing",
        big_question="What is the invoice turnaround SLA?",
        information_needed="Invoice turnaround SLA requirements.",
        confirmed_information="Invoice generation must be completed within 1 hour.",
        conversation_context=None,
        references=[ref],
    )

    assert "Invoice generation must be completed within 1 hour." in user_prompt
    assert "2 hours" in user_prompt
    assert "C* identifiers" in sys_inst
    assert "AUTHORITATIVE PROJECT EVIDENCE" in sys_inst

    def enforce_conflict_rule(prompt: str, sys_instruction: str | None) -> str:
        assert "Invoice generation must be completed within 1 hour." in prompt
        return json.dumps({
            "requirements": [
                {
                    "text": "Invoice generation shall be completed within 1 hour.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R1"]
                }
            ],
            "unresolved_gap_ids": []
        })

    fake_client = FakeLLMClient(generator_fn=enforce_conflict_rule)
    sec = generate_section(
        field_id="3.3.2",
        confirmed_information="Invoice generation must be completed within 1 hour.",
        search_fn=lambda q, f, k: [
            SearchResult(
                document_key="brd_invoice_v1",
                document_title="Invoice System BRD",
                field_id="3.3.2",
                field_title="Transaction Processing",
                chunk_index=0,
                content="Invoice generation must be completed within 2 hours.",
                similarity_score=0.9,
            )
        ],
        llm_client=fake_client,
    )
    assert "1 hour" in sec.content
    assert "2 hours" not in sec.content


def test_missing_details_not_invented_from_references():
    """Verify that unconfirmed missing details from references are not promoted to requirements."""
    ref = ReferenceCitation(
        citation_id="R1",
        document_key="brd_saas_billing",
        document_title="Subscription Billing BRD",
        field_id="3.7",
        chunk_index=1,
        similarity_score=0.88,
        content="Maximum pause duration is 3 months. Billing is paused automatically. Users may pause at most twice per year.",
    )

    sys_inst, user_prompt = build_section_generation_prompt(
        field_id="3.7",
        field_title="Feature and Functionality",
        big_question="What are the subscription pause rules?",
        information_needed="What are the settlement terms, parties involved, and timing? How are discrepancies resolved?",
        confirmed_information="Customers should be able to pause their subscription.",
        conversation_context=None,
        references=[ref],
    )

    assert "C* identifiers" in sys_inst
    assert "R* identifiers" in sys_inst

    fake_client = FakeLLMClient(
        canned_response=json.dumps({
            "requirements": [
                {
                    "text": "Customers shall be able to pause their subscription.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R1"]
                }
            ],
            "unresolved_gap_ids": [
                "G1"
            ]
        })
    )
    sec = generate_section(
        field_id="3.7",
        confirmed_information="Customers should be able to pause their subscription.",
        search_fn=lambda q, f, k: [
            SearchResult(
                document_key="brd_saas_billing",
                document_title="Subscription Billing BRD",
                field_id="3.7",
                field_title="Feature and Functionality",
                chunk_index=1,
                content="Maximum pause duration is 3 months.",
                similarity_score=0.88,
            )
        ],
        llm_client=fake_client,
    )
    assert "pause" in sec.content.lower()
    assert "Pending Confirmation / Unresolved" in sec.content
    assert "What are the settlement terms" in sec.content
    assert "3 months" not in sec.content


def test_citation_provenance_metadata_mapping_valid():
    """Verify exact mapping from citation_id -> document_key -> document_title -> field_id -> chunk_index."""
    fake_client = FakeLLMClient(
        canned_response=json.dumps({
            "requirements": [
                {
                    "text": "Generated content grounded in reference standards.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R1"]
                }
            ],
            "unresolved_gap_ids": []
        })
    )
    sec = generate_section(
        field_id="3.7",
        confirmed_information="Failed recurring subscription payments should be retried automatically.",
        search_fn=mock_search_references,
        llm_client=fake_client,
    )

    assert len(sec.cited_references) == 1
    ref = sec.cited_references[0]
    assert ref.citation_id == "R1"
    assert ref.document_key == "brd_leave_management_system"
    assert ref.document_title == "Leave Management System"
    assert ref.field_id == "3.7"
    assert ref.chunk_index == 0
    assert isinstance(ref.similarity_score, float)


# ---------------------------------------------------------------------------
# Data contract & model tests (merged from test_generation_models.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Strict Reference Isolation Regression Test Suite
# ---------------------------------------------------------------------------
def test_regression_reference_leakage_not_promoted():
    """
    Scenario 1: REFERENCE LEAKAGE REGRESSION
    Confirmed information: "Failed recurring subscription payments should be retried automatically."
    Mock retrieved reference contains:
    - upgrade settlement
    - downgrade credits
    - Finance Administrator
    - monthly reconciliation

    Unsafe generated claims (using R1 as evidence or claiming unconfirmed reference facts) must be rejected.
    """
    field_id = "3.7"
    confirmed_info = "Failed recurring subscription payments should be retried automatically."

    reference_chunks = [
        SearchResult(
            document_key="brd_subscription_billing",
            document_title="Subscription Billing (SaaS)",
            field_id="3.7",
            field_title="Settlement Plan",
            chunk_index=0,
            content=(
                "For immediate upgrades, the system calculates an upgrade settlement. "
                "Downgrade credits are applied to the next monthly invoice. "
                "The Finance Administrator performs monthly reconciliation."
            ),
            similarity_score=0.88,
        )
    ]

    # LLM attempts to promote reference facts into requirements by using R1 as factual evidence
    unsafe_llm_response = json.dumps({
        "requirements": [
            {
                "text": "For an immediate upgrade, calculate prorated adjustment.",
                "evidence_ids": ["R1"],  # ILLEGAL: R1 used as factual evidence!
                "grounding_reference_ids": ["R1"]
            },
            {
                "text": "Failed recurring subscription payments shall be retried automatically.",
                "evidence_ids": ["C1"],
                "grounding_reference_ids": ["R1"]
            }
        ],
        "unresolved_gap_ids": []
    })

    fake_client = FakeLLMClient(canned_response=unsafe_llm_response)

    with pytest.raises(UnsafeGenerationError, match="R\\* identifier"):
        generate_section(
            field_id=field_id,
            confirmed_information=confirmed_info,
            search_fn=lambda q, f, k: reference_chunks,
            llm_client=fake_client,
        )


def test_regression_conflict_resolution_confirmed_beats_reference():
    """
    Scenario 2: CONFLICT REGRESSION
    Confirmed information: "Invoice generation must complete within 1 hour."
    Reference contains: "Invoice generation must complete within 2 hours."

    Expected:
    - Introducing unsupported numeric/time value (2 hours) must be rejected with UnsafeGenerationError.
    """
    field_id = "3.3.2"
    confirmed_info = "Invoice generation must complete within 1 hour."

    reference_chunks = [
        SearchResult(
            document_key="brd_invoice_v1",
            document_title="Invoice System BRD",
            field_id="3.3.2",
            field_title="Description",
            chunk_index=0,
            content="Invoice generation must complete within 2 hours of batch submission.",
            similarity_score=0.92,
        )
    ]

    # LLM attempts to claim 2 hours under C1
    conflicted_llm_response = json.dumps({
        "requirements": [
            {
                "text": "Invoice generation shall complete within 2 hours of batch submission.",
                "evidence_ids": ["C1"],
                "grounding_reference_ids": ["R1"]
            }
        ],
        "unresolved_gap_ids": []
    })

    fake_client = FakeLLMClient(canned_response=conflicted_llm_response)

    with pytest.raises(UnsafeGenerationError, match="numeric/metric fact"):
        generate_section(
            field_id=field_id,
            confirmed_information=confirmed_info,
            search_fn=lambda q, f, k: reference_chunks,
            llm_client=fake_client,
        )


def test_regression_missing_information_unconfirmed_rule_not_promoted():
    """
    Scenario 3: MISSING INFORMATION REGRESSION
    Confirmed information: "Customers should be able to pause their subscription."
    Reference contains: "Maximum pause duration is 90 days."

    Expected:
    - Introducing unconfirmed 90 days rule must be rejected with UnsafeGenerationError.
    """
    field_id = "3.7"
    confirmed_info = "Customers should be able to pause their subscription."

    reference_chunks = [
        SearchResult(
            document_key="brd_subscription_billing",
            document_title="Subscription Billing (SaaS)",
            field_id="3.7",
            field_title="Settlement Plan",
            chunk_index=1,
            content="Maximum pause duration is 90 days. Users may pause subscription at most twice per year.",
            similarity_score=0.89,
        )
    ]

    # LLM attempts to assert 90 days rule under C1
    leaked_rule_llm_response = json.dumps({
        "requirements": [
            {
                "text": "Customers shall be able to pause their subscription. Maximum pause duration shall be 90 days.",
                "evidence_ids": ["C1"],
                "grounding_reference_ids": ["R1"]
            }
        ],
        "unresolved_gap_ids": []
    })

    fake_client = FakeLLMClient(canned_response=leaked_rule_llm_response)

    with pytest.raises(UnsafeGenerationError, match="numeric/metric fact"):
        generate_section(
            field_id=field_id,
            confirmed_information=confirmed_info,
            search_fn=lambda q, f, k: reference_chunks,
            llm_client=fake_client,
        )


def test_positive_structured_section_generation():
    """
    Positive Test for Structured Section Generation.
    C1: "Failed recurring subscription payments should be retried automatically."
    LLM output properly cites C1 as evidence and R1 as grounding.
    """
    field_id = "3.7"
    confirmed_info = "Failed recurring subscription payments should be retried automatically."

    reference_chunks = [
        SearchResult(
            document_key="brd_subscription_billing",
            document_title="Subscription Billing (SaaS)",
            field_id="3.7",
            field_title="Settlement Plan",
            chunk_index=0,
            content="Settlement and reconciliation workflow.",
            similarity_score=0.88,
        )
    ]

    structured_response = json.dumps({
        "requirements": [
            {
                "text": "Failed recurring subscription payments shall be retried automatically.",
                "evidence_ids": ["C1"],
                "grounding_reference_ids": ["R1"]
            }
        ],
        "unresolved_gap_ids": []
    })

    fake_client = FakeLLMClient(canned_response=structured_response)

    sec = generate_section(
        field_id=field_id,
        confirmed_information=confirmed_info,
        search_fn=lambda q, f, k: reference_chunks,
        llm_client=fake_client,
    )

    assert isinstance(sec, GeneratedSection)
    assert sec.is_unresolved is False
    assert "retried automatically" in sec.content
    assert len(sec.cited_references) == 1
    assert sec.cited_references[0].citation_id == "R1"
    assert sec.cited_references[0].document_key == "brd_subscription_billing"


def test_regression_unresolved_reference_leakage_rejected():
    """
    Scenario: UNRESOLVED ITEMS REFERENCE LEAKAGE REGRESSION
    Confirmed info: "Failed recurring subscription payments should be retried automatically."
    Retrieved reference contains: Finance Administrator, tax treatment, accounting entries, bank transfer,
    credit note, monthly reconciliation, payment vendor.

    1. Arbitrary free-form unresolved items are rejected at schema validation.
    2. Valid canonical gap selection (G1, G2) renders only canonical field questions,
       preventing ANY reference-only concepts from appearing in the output.
    """
    field_id = "3.7"
    confirmed_info = "Failed recurring subscription payments should be retried automatically."

    reference_chunks = [
        SearchResult(
            document_key="brd_subscription_billing",
            document_title="Subscription Billing (SaaS)",
            field_id="3.7",
            field_title="Settlement Plan",
            chunk_index=0,
            content=(
                "Finance Administrator handles monthly reconciliation. "
                "Tax treatment and accounting entries are defined for credit notes. "
                "Bank transfer mechanism and payment vendor are configured."
            ),
            similarity_score=0.88,
        )
    ]

    # 1. Attempting to provide free-form unresolved items MUST be rejected
    free_form_llm_response = json.dumps({
        "requirements": [
            {
                "text": "Failed recurring subscription payments shall be retried automatically.",
                "evidence_ids": ["C1"],
                "grounding_reference_ids": ["R1"]
            }
        ],
        "unresolved_items": [
            "Role and responsibilities for the Finance Administrator remain pending.",
            "Specific tax treatment and accounting entries are to be determined."
        ]
    })

    fake_client_free_form = FakeLLMClient(canned_response=free_form_llm_response)

    with pytest.raises(UnsafeGenerationError, match="unresolved_items"):
        generate_section(
            field_id=field_id,
            confirmed_information=confirmed_info,
            search_fn=lambda q, f, k: reference_chunks,
            llm_client=fake_client_free_form,
        )

    # 2. Using canonical G* gaps renders ONLY canonical descriptions from brd_fields.json
    valid_gaps_llm_response = json.dumps({
        "requirements": [
            {
                "text": "Failed recurring subscription payments shall be retried automatically.",
                "evidence_ids": ["C1"],
                "grounding_reference_ids": ["R1"]
            }
        ],
        "unresolved_gap_ids": ["G1", "G2"]
    })

    fake_client_valid_gaps = FakeLLMClient(canned_response=valid_gaps_llm_response)

    sec = generate_section(
        field_id=field_id,
        confirmed_information=confirmed_info,
        search_fn=lambda q, f, k: reference_chunks,
        llm_client=fake_client_valid_gaps,
    )

    content_lower = sec.content.lower()
    # Verify canonical gap descriptions are present
    assert "what are the settlement terms" in content_lower
    assert "how are discrepancies resolved" in content_lower

    # Verify that reference-only concepts NEVER appear anywhere in final content
    assert "finance administrator" not in content_lower
    assert "tax treatment" not in content_lower
    assert "accounting entries" not in content_lower
    assert "bank transfer" not in content_lower
    assert "credit note" not in content_lower
    assert "monthly reconciliation" not in content_lower
    assert "payment vendor" not in content_lower


# ---------------------------------------------------------------------------
# Canonical G* Gap Validation Tests
# ---------------------------------------------------------------------------
def test_valid_canonical_gaps_pass_and_render_canonical_descriptions():
    """Test A: Valid canonical G* gap passes and correctly renders canonical gap description."""
    fake_client = FakeLLMClient(
        canned_response=json.dumps({
            "requirements": [
                {
                    "text": "Failed recurring payments shall be retried.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R1"]
                }
            ],
            "unresolved_gap_ids": ["G1"]
        })
    )

    sec = generate_section(
        field_id="3.7",
        confirmed_information="Failed recurring payments shall be retried.",
        search_fn=mock_search_references,
        llm_client=fake_client,
    )

    assert sec.is_unresolved is False
    assert "**Pending Confirmation / Unresolved:**" in sec.content
    assert "What are the settlement terms, parties involved, and timing?" in sec.content


def test_gap_validation_rejects_invented_gap_id():
    """Test B: Invented gap ID G99 raises UnsafeGenerationError."""
    fake_client = FakeLLMClient(
        canned_response=json.dumps({
            "requirements": [
                {
                    "text": "Failed recurring payments shall be retried.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R1"]
                }
            ],
            "unresolved_gap_ids": ["G99"]
        })
    )

    with pytest.raises(UnsafeGenerationError, match="Unresolved gap ID 'G99' is invalid"):
        generate_section(
            field_id="3.7",
            confirmed_information="Failed recurring payments shall be retried.",
            search_fn=mock_search_references,
            llm_client=fake_client,
        )


def test_gap_validation_rejects_r_identifier_as_gap():
    """Test C: R1 used as unresolved gap ID raises UnsafeGenerationError."""
    fake_client = FakeLLMClient(
        canned_response=json.dumps({
            "requirements": [
                {
                    "text": "Failed recurring payments shall be retried.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R1"]
                }
            ],
            "unresolved_gap_ids": ["R1"]
        })
    )

    with pytest.raises(UnsafeGenerationError, match="R\\* identifiers"):
        generate_section(
            field_id="3.7",
            confirmed_information="Failed recurring payments shall be retried.",
            search_fn=mock_search_references,
            llm_client=fake_client,
        )


def test_gap_validation_rejects_c_identifier_as_gap():
    """Test D: C1 used as unresolved gap ID raises UnsafeGenerationError."""
    fake_client = FakeLLMClient(
        canned_response=json.dumps({
            "requirements": [
                {
                    "text": "Failed recurring payments shall be retried.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R1"]
                }
            ],
            "unresolved_gap_ids": ["C1"]
        })
    )

    with pytest.raises(UnsafeGenerationError, match="C\\* identifiers"):
        generate_section(
            field_id="3.7",
            confirmed_information="Failed recurring payments shall be retried.",
            search_fn=mock_search_references,
            llm_client=fake_client,
        )


def test_gap_validation_rejects_free_form_unresolved_items():
    """Test E: Free-form unresolved items string list is rejected."""
    fake_client = FakeLLMClient(
        canned_response=json.dumps({
            "requirements": [
                {
                    "text": "Failed recurring payments shall be retried.",
                    "evidence_ids": ["C1"],
                    "grounding_reference_ids": ["R1"]
                }
            ],
            "unresolved_items": ["Some custom unresolved note."]
        })
    )

    with pytest.raises(UnsafeGenerationError, match="unresolved_items"):
        generate_section(
            field_id="3.7",
            confirmed_information="Failed recurring payments shall be retried.",
            search_fn=mock_search_references,
            llm_client=fake_client,
        )





