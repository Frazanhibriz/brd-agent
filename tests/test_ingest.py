"""
tests/test_ingest.py
=====================
Consolidated unit and integration tests for Ingestion Submodule (loader, parser, chunker, validator, repository).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from retrieval.models import LoadedBlock, LoadedDocument, ParsedDocument, ParsedField
from ingest.loader import load_document
from ingest.parser import parse_document, match_heading, FIELD_TITLE_MAP
from ingest.chunker import create_chunks
from ingest.validator import validate_ingest
from ingest.repository import ReferenceRepository
from scripts.ingest_references import load_corpus, find_source_file


def test_canonical_field_contract_loaded():
    assert len(FIELD_TITLE_MAP) == 26
    assert "1.1.1" in FIELD_TITLE_MAP
    assert "3.2" in FIELD_TITLE_MAP
    assert "5.1" in FIELD_TITLE_MAP


def test_heading_matching():
    fid, title = match_heading("1.1.1 Background")
    assert fid == "1.1.1"
    assert title == "Background"

    fid2, title2 = match_heading("3.2 Product / Service Specification")
    assert fid2 == "3.2"
    assert title2 == "Product / Service Specification"


def test_load_corpus_manifest():
    corpus = load_corpus()
    assert isinstance(corpus, list)
    assert len(corpus) == 15


def test_find_source_file_resolution():
    file_path = find_source_file(1)
    assert file_path.exists()
    assert "Guest_Checkout_Payment_Flow" in file_path.name


def test_parser_and_chunker():
    loaded = LoadedDocument(
        path=Path("test.docx"),
        filename="test.docx",
        checksum="abc123hash",
        blocks=(
            LoadedBlock(kind="paragraph", text="1.1.1 Background"),
            LoadedBlock(kind="paragraph", text="This is the background section text."),
            LoadedBlock(kind="paragraph", text="1.2 Business Objective"),
            LoadedBlock(kind="paragraph", text="This is the business objective section text."),
        ),
    )

    parsed = parse_document(loaded)
    assert "1.1.1" in parsed.fields
    assert "1.2" in parsed.fields

    chunks = create_chunks("test_doc", parsed)
    assert len(chunks) == 2
    assert chunks[0].field_id == "1.1.1"
    assert chunks[1].field_id == "1.2"


def test_repository_save_document_with_chunks_mock():
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur
    mock_cur.fetchone.return_value = (42,)

    repo = ReferenceRepository(conn=mock_conn)

    doc_meta = {
        "sequence": 1,
        "document_id": "test_doc",
        "title": "Test Title",
        "approval_status": "approved",
    }
    loaded = LoadedDocument(
        path=Path("test.docx"),
        filename="test.docx",
        checksum="abc123hash",
        blocks=(),
    )
    parsed = ParsedDocument(
        fields={
            "1.1.1": ParsedField(
                field_id="1.1.1",
                field_title="Background",
                blocks=(LoadedBlock(kind="paragraph", text="Background content goes here."),),
            ),
        },
        empty_fields=[],
        missing_fields=[],
    )
    chunks = create_chunks("test_doc", parsed)
    embeddings = [[0.1] * 384]

    result = repo.save_document_with_chunks(
        document_meta=doc_meta,
        loaded=loaded,
        chunks=chunks,
        embeddings=embeddings,
    )

    assert result["document_id"] == "test_doc"
    assert result["inserted_chunks"] == len(chunks)
    assert result["embedded_chunks"] == 1
    assert mock_conn.commit.called


def test_nested_heading_preservation_and_chapter_boundary():
    loaded = LoadedDocument(
        path=Path("test_nested.docx"),
        filename="test_nested.docx",
        checksum="hash123",
        blocks=(
            LoadedBlock(kind="paragraph", text="3.3.2 Description", style="Heading 3"),
            LoadedBlock(kind="paragraph", text="Existing Process", style="Heading 4"),
            LoadedBlock(kind="paragraph", text="Legacy manual steps here.", style="Normal"),
            LoadedBlock(kind="paragraph", text="Proposed Process", style="Heading 4"),
            LoadedBlock(kind="paragraph", text="New automated workflow steps here.", style="Normal"),
            LoadedBlock(kind="paragraph", text="3.3.3 Security", style="Heading 3"),
            LoadedBlock(kind="paragraph", text="Security requirements text.", style="Normal"),
            LoadedBlock(kind="paragraph", text="CHAPTER IV RELEASE PLAN", style="Heading 1"),
            LoadedBlock(kind="paragraph", text="Release phase text.", style="Normal"),
        ),
    )

    parsed = parse_document(loaded)

    # 1. 3.3.2 is not empty and has content
    assert "3.3.2" not in parsed.empty_fields
    b_332_texts = [b.text for b in parsed.fields["3.3.2"].blocks]
    assert "Existing Process" in b_332_texts
    assert "Legacy manual steps here." in b_332_texts
    assert "Proposed Process" in b_332_texts
    assert "New automated workflow steps here." in b_332_texts

    # 2. Next canonical field 3.3.3 owns its text
    b_333_texts = [b.text for b in parsed.fields["3.3.3"].blocks]
    assert "Security requirements text." in b_333_texts
    assert "Legacy manual steps here." not in b_333_texts

    # 3. Chapter IV text did not leak into 3.3.3 or 3.3.2
    assert "Release phase text." not in b_332_texts
    assert "Release phase text." not in b_333_texts

    # 4. Unknown headings diagnostics captured nested and chapter headings
    assert "Existing Process" in parsed.unknown_headings
    assert "Proposed Process" in parsed.unknown_headings
    assert "CHAPTER IV RELEASE PLAN" in parsed.unknown_headings

