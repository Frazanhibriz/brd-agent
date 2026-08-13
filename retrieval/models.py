"""
retrieval/models.py
===================
Unified Data Models for BRD Reference & Retrieval Subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Ingestion Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadedBlock:
    kind: Literal["paragraph", "table"]
    text: str
    style: str | None = None


@dataclass(frozen=True)
class LoadedDocument:
    path: Path
    filename: str
    checksum: str
    blocks: tuple[LoadedBlock, ...]


@dataclass(frozen=True)
class ParsedField:
    field_id: str
    field_title: str
    blocks: tuple[LoadedBlock, ...]


@dataclass
class ParsedDocument:
    """Result of parse_document(). Consumed by chunker and validator."""
    fields: dict[str, ParsedField] = field(default_factory=dict)
    empty_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    unknown_headings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReferenceChunk:
    document_key: str
    field_id: str
    field_title: str
    chunk_index: int
    content: str
    char_count: int


# ---------------------------------------------------------------------------
# Retrieval Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchResult:
    document_key: str
    document_title: str
    field_id: str
    field_title: str
    chunk_index: int
    content: str
    similarity_score: float


@dataclass(frozen=True)
class LexicalDocumentChunk:
    document_key: str
    document_title: str
    field_id: str
    field_title: str
    chunk_index: int
    content: str
