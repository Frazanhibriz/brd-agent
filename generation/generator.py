"""
generation/generator.py
========================
Core Orchestration and Generation Engine.
Validates canonical field identity, executes field-scoped reference retrieval,
invokes LLM generation, validates citations against retrieved provenance,
and assembles section & final document models.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Sequence
from retrieval.models import SearchResult
from retrieval.semantic import search_references
from .llm_client import LLMClient, get_default_llm_client
from .models import GeneratedDocument, GeneratedSection, ReferenceCitation
from .prompts import build_section_generation_prompt, extract_confirmed_evidence, extract_canonical_gaps

ROOT = Path(__file__).resolve().parents[1]
BRD_FIELDS_PATH = ROOT / "config" / "brd_fields.json"

class UnsafeGenerationError(ValueError):
    """Raised when generated LLM output violates strict reference-isolation or evidence contracts."""
    pass


def _load_canonical_schema() -> tuple[dict[str, dict[str, str]], list[str], set[str]]:
    """
    Loads canonical BRD fields schema from config/brd_fields.json.
    Returns:
    - fields_meta: mapping of field_id -> {title, big_question, information_needed}
    - canonical_order: list of field_ids in exact canonical document sequence
    - structural_ids: set of non-answerable structural section IDs
    """
    if not BRD_FIELDS_PATH.exists():
        raise FileNotFoundError(f"BRD fields configuration not found at {BRD_FIELDS_PATH}")

    data = json.loads(BRD_FIELDS_PATH.read_text(encoding="utf-8"))

    structural_ids = {s["section_id"] for s in data.get("structural_sections", [])}
    fields_meta = {}
    canonical_order = []

    for f in data.get("fields", []):
        fid = f["field_id"]
        fields_meta[fid] = {
            "title": f.get("title", ""),
            "big_question": f.get("big_question", ""),
            "information_needed": f.get("information_needed", ""),
        }
        canonical_order.append(fid)

    return fields_meta, canonical_order, structural_ids


CANONICAL_FIELDS_META, CANONICAL_FIELD_ORDER, STRUCTURAL_SECTION_IDS = _load_canonical_schema()
CANONICAL_ANSWERABLE_FIELDS = set(CANONICAL_FIELDS_META.keys())


WORD_NUMBERS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million"
}


def _extract_numeric_tokens(text: str) -> list[str]:
    clean = re.sub(r"\[[RC]\d+\]", "", text)
    clean = re.sub(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)+", "", clean)
    tokens = re.findall(r"(?:[\$€£¥Rp]\s*)?\b\d+(?:[.,]\d+)?\b%?", clean)
    return [t.strip() for t in tokens if t.strip()]


def _validate_requirement_facts(req_text: str, cited_evidence_text: str) -> None:
    req_numeric = _extract_numeric_tokens(req_text)
    if req_numeric:
        evidence_numeric = set(_extract_numeric_tokens(cited_evidence_text))
        evidence_digits = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", cited_evidence_text))

        for token in req_numeric:
            clean_digit = re.sub(r"[^\d.]", "", token)
            if token in evidence_numeric or clean_digit in evidence_digits:
                continue
            raise UnsafeGenerationError(
                f"Generated requirement introduces unconfirmed numeric/metric fact '{token}' "
                f"not present in cited confirmed evidence: '{cited_evidence_text}'."
            )

    clean_req = re.sub(r"\[[RC]\d+\]", "", req_text)
    clean_req = re.sub(r"^\s*(?:\d+(?:\.\d+)*[.)]?\s*)+", "", clean_req)
    for word in WORD_NUMBERS:
        if re.search(rf"\b{word}\b", clean_req, re.IGNORECASE):
            if not re.search(rf"\b{word}\b", cited_evidence_text, re.IGNORECASE):
                digit_map = {
                    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                    "ten": "10", "twenty": "20", "thirty": "30", "ninety": "90", "hundred": "100"
                }
                digit_eq = digit_map.get(word.lower())
                if digit_eq and re.search(rf"\b{digit_eq}\b", cited_evidence_text):
                    continue
                raise UnsafeGenerationError(
                    f"Generated requirement introduces unconfirmed numeric/metric fact '{word}' "
                    f"not present in cited confirmed evidence: '{cited_evidence_text}'."
                )


def _parse_structured_llm_json(raw_output: str) -> dict:
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise UnsafeGenerationError(f"LLM output could not be parsed as valid JSON: {exc}\nRaw output: {raw_output[:200]}") from exc

    if not isinstance(data, dict):
        raise UnsafeGenerationError("LLM JSON output must be a JSON object (dict).")
    return data


def generate_section(
    field_id: str,
    confirmed_information: str,
    conversation_context: str | None = None,
    search_fn: Callable[[str, str | None, int], list[SearchResult]] | None = None,
    llm_client: LLMClient | None = None,
    top_k: int = 3,
) -> GeneratedSection:
    """
    Generates content for a single canonical answerable BRD section.

    Rules & Boundaries:
    - field_id is REQUIRED and must be one of the 26 canonical answerable fields.
    - field_id=None or structural/invalid field_id raises ValueError.
    - empty/whitespace confirmed_information returns an unresolved section without LLM fabrication.
    - search_references is invoked with field_id=field_id for field-scoped grounding.
    - LLM output must be structured JSON citing valid C* evidence IDs and optional R* grounding IDs.
    - Code-level validation rejects any unconfirmed numeric/time/percentage/currency values.
    """
    if field_id is None:
        raise ValueError("field_id is required for section generation. field_id=None is not permitted.")

    if field_id not in CANONICAL_ANSWERABLE_FIELDS:
        if field_id in STRUCTURAL_SECTION_IDS:
            raise ValueError(
                f"Section generation rejected: field_id '{field_id}' is a structural header section, not an answerable leaf section."
            )
        raise ValueError(
            f"Invalid or non-answerable field_id: '{field_id}'. Section generation requires one of the 26 canonical answerable fields."
        )

    meta = CANONICAL_FIELDS_META[field_id]
    field_title = meta["title"]

    # Check for empty/missing confirmed information
    if not confirmed_information or not confirmed_information.strip():
        return GeneratedSection(
            field_id=field_id,
            field_title=field_title,
            content="[Content unresolved - No confirmed information provided for this section]",
            retrieved_references=(),
            cited_references=(),
            is_unresolved=True,
        )

    confirmed_evidence_map = extract_confirmed_evidence(confirmed_information)
    canonical_gaps = extract_canonical_gaps(meta["information_needed"])

    # Retrieval formulation
    query_text = confirmed_information.strip()
    if conversation_context and conversation_context.strip():
        query_text += "\n" + conversation_context.strip()

    search_executor = search_fn or search_references
    raw_results = search_executor(query_text, field_id, top_k)

    # Assign deterministic citation IDs: R1, R2, R3...
    retrieved_refs: list[ReferenceCitation] = []
    valid_citation_map: dict[str, ReferenceCitation] = {}
    for idx, res in enumerate(raw_results, start=1):
        cid = f"R{idx}"
        citation = ReferenceCitation.from_search_result(citation_id=cid, result=res)
        retrieved_refs.append(citation)
        valid_citation_map[cid] = citation

    retrieved_tuple = tuple(retrieved_refs)

    # Build system instruction & prompt
    system_instruction, user_prompt = build_section_generation_prompt(
        field_id=field_id,
        field_title=field_title,
        big_question=meta["big_question"],
        information_needed=meta["information_needed"],
        confirmed_information=confirmed_evidence_map,
        conversation_context=conversation_context,
        references=retrieved_tuple,
        canonical_gaps=canonical_gaps,
    )

    # Invoke LLM
    client = llm_client or get_default_llm_client()
    raw_output = client.generate(prompt=user_prompt, system_instruction=system_instruction)

    # Parse and validate structured JSON
    data = _parse_structured_llm_json(raw_output)

    if "requirements" not in data or not isinstance(data["requirements"], list):
        raise UnsafeGenerationError("Structured JSON output must contain a 'requirements' list.")

    # Disallow legacy free-form unresolved_items
    if "unresolved_items" in data and data["unresolved_items"]:
        raise UnsafeGenerationError(
            "Free-form 'unresolved_items' is not permitted. You must select canonical gap IDs using 'unresolved_gap_ids'."
        )

    unresolved_gap_ids = data.get("unresolved_gap_ids", [])
    if not isinstance(unresolved_gap_ids, list):
        raise UnsafeGenerationError("'unresolved_gap_ids' must be a list of canonical G* gap IDs.")

    # Validate gap IDs against canonical_gaps
    validated_gap_ids: list[str] = []
    for gid in unresolved_gap_ids:
        if not isinstance(gid, str):
            raise UnsafeGenerationError(f"Gap ID '{gid}' must be a string.")
        if gid.startswith("R"):
            raise UnsafeGenerationError(
                f"Invalid gap ID '{gid}'. R* identifiers are non-authoritative reference citations and cannot be used as gap IDs."
            )
        if gid.startswith("C"):
            raise UnsafeGenerationError(
                f"Invalid gap ID '{gid}'. C* identifiers are confirmed evidence and cannot be used as gap IDs."
            )
        if not gid.startswith("G") or gid not in canonical_gaps:
            raise UnsafeGenerationError(
                f"Unresolved gap ID '{gid}' is invalid for field {field_id}. Valid canonical G* gap IDs: {sorted(list(canonical_gaps.keys()))}"
            )
        if gid not in validated_gap_ids:
            validated_gap_ids.append(gid)

    req_list = data["requirements"]
    validated_reqs: list[dict] = []
    cited_r_ids: set[str] = set()

    for idx, req in enumerate(req_list):
        if not isinstance(req, dict):
            raise UnsafeGenerationError(f"Requirement at index {idx} must be a JSON object.")

        req_text = req.get("text", "").strip()
        if not req_text:
            continue

        evidence_ids = req.get("evidence_ids")
        if not evidence_ids or not isinstance(evidence_ids, list) or len(evidence_ids) == 0:
            raise UnsafeGenerationError(
                f"Requirement '{req_text[:40]}' is missing required 'evidence_ids' list."
            )

        cited_evidence_parts: list[str] = []
        for eid in evidence_ids:
            if not isinstance(eid, str):
                raise UnsafeGenerationError(f"Evidence ID '{eid}' must be a string.")
            if eid.startswith("R"):
                raise UnsafeGenerationError(
                    f"Requirement '{req_text[:40]}' uses R* identifier '{eid}' as factual evidence. "
                    "R* identifiers can only be used as grounding_reference_ids, never as evidence_ids."
                )
            if eid not in confirmed_evidence_map:
                raise UnsafeGenerationError(
                    f"Requirement '{req_text[:40]}' cites non-existent confirmed evidence ID '{eid}'. "
                    f"Valid C* evidence IDs: {sorted(list(confirmed_evidence_map.keys()))}"
                )
            cited_evidence_parts.append(confirmed_evidence_map[eid])

        combined_evidence_text = "\n".join(cited_evidence_parts)

        # Validate grounding reference IDs
        grounding_refs = req.get("grounding_reference_ids", [])
        if not isinstance(grounding_refs, list):
            raise UnsafeGenerationError(f"grounding_reference_ids must be a list for requirement '{req_text[:40]}'.")

        for rid in grounding_refs:
            if not isinstance(rid, str) or not rid.startswith("R") or rid not in valid_citation_map:
                raise UnsafeGenerationError(
                    f"Requirement cites invalid or non-existent grounding reference ID '{rid}'. "
                    f"Valid R* IDs: {sorted(list(valid_citation_map.keys()))}"
                )
            cited_r_ids.add(rid)

        # Check inline citation markers in text
        for inline_rid in re.findall(r"\[(R\d+)\]", req_text):
            if inline_rid not in valid_citation_map:
                raise UnsafeGenerationError(
                    f"Requirement text contains invalid reference citation '[{inline_rid}]'. "
                    f"Valid R* IDs: {sorted(list(valid_citation_map.keys()))}"
                )
            cited_r_ids.add(inline_rid)

        # Validate numeric/temporal/currency facts
        _validate_requirement_facts(req_text, combined_evidence_text)

        validated_reqs.append({
            "text": req_text,
            "evidence_ids": evidence_ids,
            "grounding_reference_ids": grounding_refs,
        })

    # Construct final content
    content_lines: list[str] = []
    if validated_reqs:
        for r in validated_reqs:
            t = r["text"]
            g_refs = r["grounding_reference_ids"]
            if g_refs:
                tags_to_add = [f"[{gid}]" for gid in g_refs if f"[{gid}]" not in t]
                if tags_to_add:
                    t = f"{t} {' '.join(tags_to_add)}"
            content_lines.append(f"- {t}" if len(validated_reqs) > 1 else t)
    else:
        content_lines.append("[No confirmed requirements generated for this section]")

    if validated_gap_ids:
        content_lines.append("")
        content_lines.append("**Pending Confirmation / Unresolved:**")
        for gid in validated_gap_ids:
            content_lines.append(f"- {canonical_gaps[gid]}")

    final_content = "\n".join(content_lines)
    cited_refs_tuple = tuple(valid_citation_map[rid] for rid in sorted(cited_r_ids))

    return GeneratedSection(
        field_id=field_id,
        field_title=field_title,
        content=final_content.strip(),
        retrieved_references=retrieved_tuple,
        cited_references=cited_refs_tuple,
        is_unresolved=False,
    )




def generate_final_document(
    confirmed_sections: dict[str, str] | None = None,
    conversation_contexts: dict[str, str] | None = None,
    search_fn: Callable[[str, str | None, int], list[SearchResult]] | None = None,
    llm_client: LLMClient | None = None,
    top_k: int = 3,
) -> GeneratedDocument:
    """
    Assembles a complete GeneratedDocument by generating sections in canonical BRD field order.
    """
    confirmed_map = confirmed_sections or {}
    contexts_map = conversation_contexts or {}

    sections: list[GeneratedSection] = []
    for fid in CANONICAL_FIELD_ORDER:
        confirmed_info = confirmed_map.get(fid, "")
        conv_context = contexts_map.get(fid, None)

        sec = generate_section(
            field_id=fid,
            confirmed_information=confirmed_info,
            conversation_context=conv_context,
            search_fn=search_fn,
            llm_client=llm_client,
            top_k=top_k,
        )
        sections.append(sec)

    return GeneratedDocument(sections=tuple(sections))
