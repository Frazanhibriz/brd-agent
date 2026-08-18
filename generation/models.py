"""
generation/models.py
====================
Structured Data Models for AI2-5 Document & Section Generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from retrieval.models import SearchResult


@dataclass(frozen=True)
class ReferenceCitation:
    """
    Provenance tracking for a reference chunk provided to or cited by the LLM.
    """
    citation_id: str  
    document_key: str
    document_title: str
    field_id: str
    chunk_index: int
    similarity_score: float
    content: str

    @classmethod
    def from_search_result(cls, citation_id: str, result: SearchResult) -> ReferenceCitation:
        return cls(
            citation_id=citation_id,
            document_key=result.document_key,
            document_title=result.document_title,
            field_id=result.field_id,
            chunk_index=result.chunk_index,
            similarity_score=result.similarity_score,
            content=result.content,
        )


@dataclass(frozen=True)
class GeneratedSection:
    """
    Generated content and reference provenance for a single canonical BRD section.
    """
    field_id: str
    field_title: str
    content: str
    retrieved_references: tuple[ReferenceCitation, ...] = ()
    cited_references: tuple[ReferenceCitation, ...] = ()
    is_unresolved: bool = False


@dataclass(frozen=True)
class GeneratedDocument:
    """
    Assembled BRD document consisting of canonical field-aligned GeneratedSections.
    """
    sections: tuple[GeneratedSection, ...] = ()
