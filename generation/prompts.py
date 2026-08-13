# generation/prompts.py
from __future__ import annotations
from typing import Sequence
from .models import ReferenceCitation

SYSTEM_PROMPT = """You are an expert Business Analyst generating a specific section of a Business Requirement Document (BRD).

STRICT RULES:
1. USER-CONFIRMED INFORMATION IS THE PRIMARY SOURCE OF TRUTH.
2. Reference documents are only for style, structure, and optional examples. NEVER let reference facts override confirmed user information.
3. NEVER invent numbers, dates, SLAs, policies, or technical rules not confirmed by the user.
4. If information is missing or unresolved, explicitly state it as unresolved/open.
5. Generate ONLY the section requested. Do NOT generate other sections.
6. When using wording or ideas from a reference, cite it using [R1], [R2], etc.
"""

def build_section_generation_prompt(
    field_id: str,
    field_title: str,
    big_question: str,
    information_needed: str,
    confirmed_information: str,
    conversation_context: str | None,
    references: Sequence[ReferenceCitation],
) -> tuple[str, str]:
    """Builds the complete prompt for section generation.
    
    Returns:
        tuple[str, str]: A tuple containing (system_instruction, user_prompt).
    """
    system_instruction = SYSTEM_PROMPT

    # Format references
    ref_text = ""
    if references:
        ref_text = "\n\nREFERENCES:\n" + "\n\n".join(
            f"[{r.citation_id}] {r.title}\n{r.snippet}" for r in references
        )

    # Format context
    context_text = f"\n\nCONVERSATION CONTEXT:\n{conversation_context}" if conversation_context else ""

    # Build the user prompt
    user_prompt = (
        f"SECTION ID: {field_id}\n"
        f"SECTION TITLE: {field_title}\n"
        f"OBJECTIVE: {big_question}\n"
        f"REQUIREMENTS: {information_needed}\n\n"
        f"USER CONFIRMED INFORMATION:\n{confirmed_information}"
        f"{context_text}"
        f"{ref_text}"
    )

    return system_instruction, user_prompt

