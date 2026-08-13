"""
generation/renderer.py
======================
Renders structured GeneratedSection and GeneratedDocument objects into readable Markdown.
"""

from __future__ import annotations

from .models import GeneratedDocument, GeneratedSection


def render_section_to_markdown(
    section: GeneratedSection,
    include_citations_footer: bool = True,
) -> str:
    """
    Renders a single GeneratedSection into formatted Markdown text.
    """
    lines: list[str] = []
    lines.append(f"### Section {section.field_id}: {section.field_title}")
    lines.append("")
    lines.append(section.content)
    lines.append("")

    if include_citations_footer and section.cited_references:
        lines.append("**Grounding References:**")
        for ref in section.cited_references:
            lines.append(
                f"- [{ref.citation_id}] {ref.document_title} (`{ref.document_key}`, Chunk {ref.chunk_index}) - Section {ref.field_id}"
            )
        lines.append("")

    return "\n".join(lines)


def render_document_to_markdown(
    document: GeneratedDocument,
    title: str = "Business Requirements Document (BRD)",
    include_citations_footer: bool = True,
) -> str:
    """
    Renders a complete GeneratedDocument into a structured Markdown document.
    """
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("---")
    lines.append("")

    for sec in document.sections:
        lines.append(render_section_to_markdown(sec, include_citations_footer=include_citations_footer))
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
