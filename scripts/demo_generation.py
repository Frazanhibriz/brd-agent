"""
scripts/demo_generation.py
===========================
Standalone demonstration script for AI2-5 generation capabilities.

Demonstrates:
A. Section Generation with valid canonical field_id, confirmed information, and reference grounding.
B. General Generation Guard (field_id=None safe rejection).
C. Final Document Assembly in canonical BRD field order.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.generator import generate_final_document, generate_section
from generation.llm_client import FakeLLMClient
from generation.renderer import render_document_to_markdown, render_section_to_markdown
from retrieval.models import SearchResult


def mock_demo_retrieval(query: str, field_id: str | None = None, top_k: int = 3) -> list[SearchResult]:
    """Mock search_references returning realistic reference BRD grounding chunks."""
    if field_id == "3.3.2":
        return [
            SearchResult(
                document_key="brd_invoice_management_system",
                document_title="Invoice Management System BRD",
                field_id="3.3.2",
                field_title="Transaction Processing",
                chunk_index=1,
                content="Automatic monthly billing cycle triggers batch invoice generation at 00:00 UTC.",
                similarity_score=0.92,
            ),
            SearchResult(
                document_key="brd_subscription_billing",
                document_title="Subscription Billing BRD",
                field_id="3.3.2",
                field_title="Transaction Processing",
                chunk_index=3,
                content="Invoices are generated electronically in PDF and XML formats upon plan renewal.",
                similarity_score=0.88,
            ),
        ][:top_k]
    
    return [
        SearchResult(
            document_key="brd_general_reference",
            document_title="Enterprise Software Reference BRD",
            field_id=field_id or "1.1.1",
            field_title="Overview",
            chunk_index=0,
            content="Standard enterprise system operational requirements.",
            similarity_score=0.85,
        )
    ][:top_k]


def main():
    print("================================================================================")
    print("AI2-5 GENERATION SUBSYSTEM STANDALONE DEMONSTRATION")
    print("================================================================================")
    print()

    # --------------------------------------------------------------------------
    # DEMO A: SECTION GENERATION WITH CANONICAL FIELD & REFERENCE GROUNDING
    # --------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("DEMO A: Section Generation (field_id='3.3.2')")
    print("--------------------------------------------------------------------------------")

    confirmed_info = "The system shall generate monthly automatic invoices for subscription renewals within 1 hour of payment processing."
    conv_context = "User mentioned potential weekly invoicing, but confirmed monthly invoicing as mandatory."

    fake_client_section = FakeLLMClient(
        canned_response=(
            "The billing engine shall automatically trigger monthly invoice generation within 1 hour of subscription renewal payment. "
            "Batch processing shall run at midnight UTC, producing electronic invoices in PDF and XML formats [R1] [R2]."
        )
    )

    section = generate_section(
        field_id="3.3.2",
        confirmed_information=confirmed_info,
        conversation_context=conv_context,
        search_fn=mock_demo_retrieval,
        llm_client=fake_client_section,
    )

    print(render_section_to_markdown(section))
    print(f"Is Unresolved: {section.is_unresolved}")
    print(f"Retrieved References Count: {len(section.retrieved_references)}")
    print(f"Cited References Count: {len(section.cited_references)}")
    print()

    # --------------------------------------------------------------------------
    # DEMO B: GENERAL CHATROOM GUARD (field_id=None SAFE REJECTION)
    # --------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("DEMO B: General Chatroom Guard (field_id=None)")
    print("--------------------------------------------------------------------------------")

    try:
        generate_section(
            field_id=None,  # type: ignore[arg-type]
            confirmed_information="Brainstorming about invoice concepts.",
            search_fn=mock_demo_retrieval,
            llm_client=fake_client_section,
        )
        print("ERROR: General chat unexpectedly allowed section generation!")
    except ValueError as err:
        print(f"SUCCESSFULLY GUARDED: General Chat (field_id=None) rejected section generation.")
        print(f"Caught expected Exception: {err}")
    print()

    # --------------------------------------------------------------------------
    # DEMO C: FINAL DOCUMENT ASSEMBLY IN CANONICAL FIELD ORDER
    # --------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("DEMO C: Final Document Assembly (Canonical Field Order)")
    print("--------------------------------------------------------------------------------")

    confirmed_sections = {
        "1.1.1": "Legacy invoice processing requires 48 hours of manual audit.",
        "1.2": "Automate invoice processing to reduce operational turnaround to under 1 hour.",
        "3.3.2": "Generate monthly automatic invoices upon payment receipt.",
    }

    fake_client_doc = FakeLLMClient(
        canned_response="Requirement specification for target section based on confirmed facts [R1]."
    )

    doc = generate_final_document(
        confirmed_sections=confirmed_sections,
        search_fn=mock_demo_retrieval,
        llm_client=fake_client_doc,
    )

    print(f"Total Canonical Document Sections: {len(doc.sections)}")
    resolved_count = sum(1 for s in doc.sections if not s.is_unresolved)
    unresolved_count = sum(1 for s in doc.sections if s.is_unresolved)
    print(f"Resolved Sections: {resolved_count}")
    print(f"Unresolved Sections (missing input): {unresolved_count}")
    print()

    # Render first 2 sections of the document to Markdown
    rendered_doc = render_document_to_markdown(doc)
    preview_lines = rendered_doc.split("\n")[:35]
    print("--- DOCUMENT MARKDOWN PREVIEW (First 35 lines) ---")
    print("\n".join(preview_lines))
    print("...\n")

    print("================================================================================")
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("================================================================================")


if __name__ == "__main__":
    main()
