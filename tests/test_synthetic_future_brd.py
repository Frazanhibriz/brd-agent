"""
tests/test_synthetic_future_brd.py
==================================
Tests ingestion scaling when corpus document count exceeds 15.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from retrieval.models import LoadedBlock, LoadedDocument, ParsedDocument, ParsedField
from ingest.chunker import create_chunks
from ingest.parser import parse_document
from ingest.validator import validate_ingest
from ingest.repository import ReferenceRepository


def test_synthetic_future_brd_sequence_16():
    meta = {
        "sequence": 16,
        "document_id": "brd_future_system_v16",
        "title": "Synthetic Future AI Operations System",
        "approval_status": "approved",
    }
    loaded = LoadedDocument(
        path=Path("synthetic.docx"),
        filename="16 - BRD_Future_AI_Operations.docx",
        checksum="synthetic_hash_16",
        blocks=(
            LoadedBlock(kind="paragraph", text="1.1.1 Background"),
            LoadedBlock(kind="paragraph", text="Synthetic future AI ops background details."),
            LoadedBlock(kind="paragraph", text="1.2 Business Objective"),
            LoadedBlock(kind="paragraph", text="Scale reference corpus beyond initial 15 seed documents."),
        ),
    )

    parsed = parse_document(loaded)
    chunks = create_chunks(meta["document_id"], parsed)
    validation = validate_ingest(meta, parsed, chunks)

    assert validation.is_valid
    assert len(chunks) == 2

    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchone.return_value = (16,)

    repo = ReferenceRepository(conn=mock_conn)
    result = repo.save_document_with_chunks(meta, loaded, chunks)

    assert result["document_id"] == "brd_future_system_v16"
    assert result["inserted_chunks"] == 2
