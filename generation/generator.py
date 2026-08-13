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
from .prompts import build_section_generation_prompt

ROOT = Path(__file__).resolve().parents[1]
BRD_FIELDS_PATH = ROOT / "config" / "brd_fields.json"


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
    - citations in LLM output are validated against retrieved citation IDs.
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
        confirmed_information=confirmed_information,
        conversation_context=conversation_context,
        references=retrieved_tuple,
    )

    # Invoke LLM
    client = llm_client or get_default_llm_client()
    raw_output = client.generate(prompt=user_prompt, system_instruction=system_instruction)

    # Extract and validate citation markers (e.g. [R1], [R2])
    found_citations = re.findall(r"\[(R\d+)\]", raw_output)

    cited_refs: list[ReferenceCitation] = []
    seen_cids = set()
    for cid in found_citations:
        if cid not in valid_citation_map:
            raise ValueError(
                f"Model generated invalid citation: [{cid}]. Valid citations are: {sorted(list(valid_citation_map.keys()))}"
            )
        if cid not in seen_cids:
            seen_cids.add(cid)
            cited_refs.append(valid_citation_map[cid])

    return GeneratedSection(
        field_id=field_id,
        field_title=field_title,
        content=raw_output.strip(),
        retrieved_references=retrieved_tuple,
        cited_references=tuple(cited_refs),
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
