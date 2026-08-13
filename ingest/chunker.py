"""
ingest/chunker.py
=================
Splits parsed field content blocks into field-aligned reference chunks.
Ensures ONE chunk belongs to ONE canonical field only.
"""

from __future__ import annotations

from retrieval.models import ParsedDocument, ReferenceChunk

TARGET_CHUNK_CHARS = 1200
MAX_CHUNK_CHARS = 2000


def create_chunks(
    document_key: str,
    parsed: ParsedDocument,
    target_chars: int = TARGET_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[ReferenceChunk]:
    """
    Generate field-aligned chunks from a ParsedDocument.
    Guarantees no chunk crosses a field boundary.
    """
    chunks: list[ReferenceChunk] = []

    for fid in sorted(parsed.fields, key=lambda v: [int(p) for p in v.split(".")]):
        field_obj = parsed.fields[fid]
        if not field_obj.blocks:
            continue

        accumulated_text_blocks: list[str] = []
        accumulated_char_count = 0
        chunk_idx = 0

        for block in field_obj.blocks:
            block_text = block.text.strip()
            if not block_text:
                continue

            block_len = len(block_text)

            if accumulated_char_count + block_len > max_chars and accumulated_text_blocks:
                content = "\n\n".join(accumulated_text_blocks)
                chunks.append(
                    ReferenceChunk(
                        document_key=document_key,
                        field_id=fid,
                        field_title=field_obj.field_title,
                        chunk_index=chunk_idx,
                        content=content,
                        char_count=len(content),
                    )
                )
                chunk_idx += 1
                accumulated_text_blocks = []
                accumulated_char_count = 0

            accumulated_text_blocks.append(block_text)
            accumulated_char_count += block_len

            if accumulated_char_count >= target_chars:
                content = "\n\n".join(accumulated_text_blocks)
                chunks.append(
                    ReferenceChunk(
                        document_key=document_key,
                        field_id=fid,
                        field_title=field_obj.field_title,
                        chunk_index=chunk_idx,
                        content=content,
                        char_count=len(content),
                    )
                )
                chunk_idx += 1
                accumulated_text_blocks = []
                accumulated_char_count = 0

        if accumulated_text_blocks:
            content = "\n\n".join(accumulated_text_blocks)
            chunks.append(
                ReferenceChunk(
                    document_key=document_key,
                    field_id=fid,
                    field_title=field_obj.field_title,
                    chunk_index=chunk_idx,
                    content=content,
                    char_count=len(content),
                )
            )

    return chunks
