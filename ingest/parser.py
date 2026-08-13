"""
ingest/parser.py
================
Maps LoadedDocument content blocks to canonical BRD fields.
Loads the canonical field contract directly from `config/brd_fields.json`.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

from retrieval.models import LoadedBlock, LoadedDocument, ParsedDocument, ParsedField

ROOT = Path(__file__).resolve().parents[1]
BRD_FIELDS_PATH = ROOT / "config" / "brd_fields.json"


def _load_canonical_contract() -> tuple[dict[str, str], set[str]]:
    if not BRD_FIELDS_PATH.exists():
        raise FileNotFoundError(f"BRD fields configuration not found at {BRD_FIELDS_PATH}")

    data = json.loads(BRD_FIELDS_PATH.read_text(encoding="utf-8"))

    title_map: dict[str, str] = {}
    structural_ids: set[str] = {"1", "2", "3", "4", "5"}

    for s in data.get("structural_sections", []):
        structural_ids.add(s["section_id"])

    for f in data.get("fields", []):
        title_map[f["field_id"]] = f["title"]

    return title_map, structural_ids


FIELD_TITLE_MAP, STRUCTURAL_IDS = _load_canonical_contract()

SECTION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*(?:[-–—:.)]\s*)?(.*)$")
FUZZY_THRESHOLD = 0.72


def _normalize_title(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _fuzzy_match_title(title_text: str) -> str | None:
    norm = _normalize_title(title_text)
    if not norm:
        return None

    best_match: str | None = None
    best_score = 0.0

    for fid, canon_title in FIELD_TITLE_MAP.items():
        score = SequenceMatcher(None, norm, _normalize_title(canon_title)).ratio()
        if score > best_score:
            best_score = score
            best_match = fid

    if best_score >= FUZZY_THRESHOLD and best_match is not None:
        return best_match
    return None


def match_heading(text: str) -> tuple[str | None, str | None]:
    match = SECTION_PATTERN.match(text)
    if match:
        raw_id, rest = match.groups()
        if raw_id in FIELD_TITLE_MAP:
            return raw_id, FIELD_TITLE_MAP[raw_id]
        if raw_id in STRUCTURAL_IDS:
            return raw_id, None

    matched_id = _fuzzy_match_title(text)
    if matched_id:
        return matched_id, FIELD_TITLE_MAP[matched_id]

    return None, None


_TOP_LEVEL_HEADING_RE = re.compile(
    r"^(CHAPTER\s+[I|V|X\d]+|APPROVAL\s+SHEET|FOREWORD|PREFACE|ABSTRACT|TABLE\s+OF\s+CONTENTS|LIST\s+OF\s+TABLES|RELEASE\s+PLAN|RETIREMENT\s+PLAN|DOCUMENT\s+SIGNOFF|5\.2\s+DOCUMENT\s+SIGNOFF)",
    re.IGNORECASE,
)


def parse_document(loaded: LoadedDocument) -> ParsedDocument:
    """
    Parse a LoadedDocument into canonical fields.
    Collects content blocks under each 26-field heading.
    Preserves nested subheadings (e.g. Heading 4 / Heading 5) within parent canonical fields.
    """
    field_blocks: dict[str, list[LoadedBlock]] = {fid: [] for fid in FIELD_TITLE_MAP}
    detected_headings: set[str] = set()
    unknown_headings: list[str] = []
    current_field_id: str | None = None

    for block in loaded.blocks:
        if block.kind == "paragraph":
            matched_id, _ = match_heading(block.text)
            if matched_id:
                if matched_id in FIELD_TITLE_MAP:
                    detected_headings.add(matched_id)
                    current_field_id = matched_id
                elif matched_id in STRUCTURAL_IDS:
                    current_field_id = None
                continue

            if (block.style and block.style == "Heading 1") or _TOP_LEVEL_HEADING_RE.search(block.text.strip()):
                unknown_headings.append(block.text)
                current_field_id = None
                continue

            if block.style and "Heading" in block.style:
                unknown_headings.append(block.text)
                if current_field_id is not None:
                    field_blocks[current_field_id].append(block)
                continue

        if current_field_id is not None:
            field_blocks[current_field_id].append(block)

    parsed_fields: dict[str, ParsedField] = {}
    empty_fields: list[str] = []
    missing_fields: list[str] = []

    for fid, title in FIELD_TITLE_MAP.items():
        blocks = tuple(field_blocks[fid])
        parsed_fields[fid] = ParsedField(field_id=fid, field_title=title, blocks=blocks)

        if fid in detected_headings:
            if not blocks:
                empty_fields.append(fid)
        else:
            missing_fields.append(fid)

    return ParsedDocument(
        fields=parsed_fields,
        empty_fields=empty_fields,
        missing_fields=missing_fields,
        unknown_headings=unknown_headings,
    )
