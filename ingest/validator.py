"""
ingest/validator.py
===================
Validates parsed documents and generated chunks against quality rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from retrieval.models import ParsedDocument, ReferenceChunk


@dataclass
class ValidationReport:
    document_key: str
    is_valid: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_ingest(
    document_meta: dict,
    parsed: ParsedDocument,
    chunks: Sequence[ReferenceChunk],
) -> ValidationReport:
    document_key = document_meta.get("document_key") or document_meta.get("document_id", "unknown")
    report = ValidationReport(document_key=document_key)

    if parsed.empty_fields:
        for fid in parsed.empty_fields:
            report.warnings.append(f"Empty field (heading present, no content): {fid}")

    if parsed.missing_fields:
        for fid in parsed.missing_fields:
            report.warnings.append(f"Missing field (heading not detected): {fid}")

    if parsed.unknown_headings:
        for heading in parsed.unknown_headings:
            report.warnings.append(f"Unrecognized heading: '{heading}'")

    if not chunks and any(parsed.fields[fid].blocks for fid in parsed.fields):
        report.errors.append("No chunks generated despite non-empty parsed fields.")
        report.is_valid = False

    for chunk in chunks:
        if not chunk.content or not chunk.content.strip():
            report.errors.append(f"Empty chunk content found for field {chunk.field_id} index {chunk.chunk_index}.")
            report.is_valid = False

    return report
