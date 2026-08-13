"""
ingest/loader.py
================
Reads paragraphs and tables from reference BRD .docx files.
Filters Table of Contents entries.
"""

from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from retrieval.models import LoadedBlock, LoadedDocument


def calculate_checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_to_text(table: Table) -> str:
    rows: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def load_document(file_path: str | Path) -> LoadedDocument:
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"BRD file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() != ".docx":
        raise ValueError(f"Only .docx files are supported, got: {path.suffix}")

    try:
        document = Document(str(path))
    except Exception as exc:
        raise ValueError(f"Unable to read DOCX: {path.name}") from exc

    _TOC_TEXT_RE = re.compile(r"\t\d+\s*$")
    blocks: list[LoadedBlock] = []

    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue

            style_name = item.style.name if item.style is not None else None

            if style_name and style_name.upper().startswith("TOC"):
                continue
            if _TOC_TEXT_RE.search(text):
                continue

            blocks.append(LoadedBlock(kind="paragraph", text=text, style=style_name))

        elif isinstance(item, Table):
            text = table_to_text(item)
            if text:
                blocks.append(LoadedBlock(kind="table", text=text))

    return LoadedDocument(
        path=path,
        filename=path.name,
        checksum=calculate_checksum(path),
        blocks=tuple(blocks),
    )
