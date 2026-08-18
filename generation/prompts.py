# generation/prompts.py
from __future__ import annotations
import json
import re
from typing import Sequence
from .models import ReferenceCitation

SYSTEM_PROMPT = """You are an expert Business Analyst generating a structured section of a Business Requirement Document (BRD).

STRICT OUTPUT FORMAT:
You MUST respond with a valid, parseable JSON object matching this exact JSON schema:
{
  "requirements": [
    {
      "text": "The system shall support...",
      "evidence_ids": ["C1"],
      "grounding_reference_ids": ["R1"]
    }
  ],
  "unresolved_gap_ids": [
    "G1",
    "G2"
  ]
}

STRICT EVIDENCE, GROUNDING & GAP RULES:
1. C* identifiers (e.g. C1, C2) represent AUTHORITATIVE PROJECT EVIDENCE.
   - Every requirement in 'requirements' MUST cite at least one valid C* evidence ID in 'evidence_ids'.
   - A requirement MUST ONLY express facts, capabilities, and rules that are directly supported by its cited C* evidence.
   - Never invent an evidence relationship.
2. R* identifiers (e.g. R1, R2) represent NON-AUTHORITATIVE REFERENCE GROUNDING.
   - R* identifiers can NEVER be used in 'evidence_ids' and can NEVER justify a project requirement.
   - R* identifiers may ONLY be placed in 'grounding_reference_ids' to record terminology, professional phrasing, or structural style provenance.
   - If a detail, workflow, actor, vendor, policy, SLA, or number appears only in R* and not in C*, you MUST OMIT IT entirely.
   - You MUST NOT introduce numbers, timeframes, percentages, currencies, or metrics that are not in the cited C* evidence.
3. G* identifiers (e.g. G1, G2) represent CANONICAL FIELD INFORMATION GAPS.
   - The 'unresolved_gap_ids' array may contain ONLY valid G* identifiers from the provided list.
   - Select a G* identifier ONLY if the corresponding canonical question is NOT fully resolved by confirmed project evidence (C*).
   - Do NOT output arbitrary free-form unresolved text. You must ONLY output G* identifiers.
   - Reference BRD content (R*) must NEVER be used to invent or justify new unresolved topics.

Do not include any conversational preamble or markdown code fences other than the raw JSON output.
"""


def extract_confirmed_evidence(confirmed_info: str | dict[str, str]) -> dict[str, str]:
    """
    Converts confirmed project information into discrete C1, C2, ... evidence records.
    """
    if isinstance(confirmed_info, dict):
        return confirmed_info
    if not confirmed_info or not confirmed_info.strip():
        return {}

    raw_lines = [l.strip() for l in confirmed_info.strip().splitlines() if l.strip()]
    if len(raw_lines) > 1:
        evidence_map: dict[str, str] = {}
        for idx, line in enumerate(raw_lines, start=1):
            cleaned = re.sub(r"^[-*•\d+.)]+\s*", "", line).strip()
            evidence_map[f"C{idx}"] = cleaned if cleaned else line
        return evidence_map
    else:
        return {"C1": confirmed_info.strip()}


def extract_canonical_gaps(information_needed: str) -> dict[str, str]:
    """
    Extracts canonical information gap definitions (G1, G2, ...) from the field's information_needed metadata.
    Splits on sentence and question boundaries.
    """
    if not information_needed or not information_needed.strip():
        return {}

    items = re.split(r"(?<=[?.!])\s+", information_needed.strip())
    gap_map: dict[str, str] = {}
    gap_idx = 1
    for item in items:
        cleaned = item.strip()
        if cleaned:
            gap_map[f"G{gap_idx}"] = cleaned
            gap_idx += 1
    return gap_map


def build_section_generation_prompt(
    field_id: str,
    field_title: str,
    big_question: str,
    information_needed: str,
    confirmed_information: str | dict[str, str],
    conversation_context: str | None,
    references: Sequence[ReferenceCitation],
    canonical_gaps: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Builds the complete prompt for section generation.
    
    Returns:
        tuple[str, str]: A tuple containing (system_instruction, user_prompt).
    """
    system_instruction = SYSTEM_PROMPT

    evidence_map = extract_confirmed_evidence(confirmed_information)
    gaps_map = canonical_gaps if canonical_gaps is not None else extract_canonical_gaps(information_needed)

    confirmed_lines: list[str] = []
    if evidence_map:
        for cid, ctext in evidence_map.items():
            confirmed_lines.append(f"[{cid}] {ctext}")
    else:
        confirmed_lines.append("[None provided]")

    sections_text = [
        f"SECTION ID: {field_id}",
        f"SECTION TITLE: {field_title}",
        f"OBJECTIVE: {big_question}",
        f"REQUIREMENTS: {information_needed}",
        "",
        "=== 1. CONFIRMED PROJECT EVIDENCE (AUTHORITATIVE TRUTH - C* IDENTIFIERS) ===",
        "\n".join(confirmed_lines),
    ]

    if conversation_context and conversation_context.strip():
        sections_text.extend([
            "",
            "=== 2. CONVERSATION CONTEXT (NON-AUTHORITATIVE DISCUSSION) ===",
            conversation_context.strip(),
        ])

    if references:
        ref_blocks = []
        for r in references:
            ref_blocks.append(f"[{r.citation_id}] Document: '{r.document_title}' (Chunk {r.chunk_index})\n{r.content}")
        sections_text.extend([
            "",
            "=== 3. RETRIEVED REFERENCE BRDs (NON-AUTHORITATIVE GROUNDING ONLY - R* IDENTIFIERS) ===",
            "\n\n".join(ref_blocks),
        ])

    if gaps_map:
        gap_lines = [f"[{gid}] {gtext}" for gid, gtext in gaps_map.items()]
        sections_text.extend([
            "",
            "=== 4. CANONICAL FIELD GAPS (SELECTABLE G* IDENTIFIERS FOR UNRESOLVED ITEMS) ===",
            "\n".join(gap_lines),
        ])

    user_prompt = "\n".join(sections_text)

    return system_instruction, user_prompt




