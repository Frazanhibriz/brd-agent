"""
ingest/repository.py
====================
Handles persistence of reference BRD documents, chunks, and vector embeddings
to PostgreSQL via psycopg (v3).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import psycopg

from retrieval.models import LoadedDocument, ReferenceChunk


class ReferenceRepository:
    """
    Handles persistence of reference BRD documents and their chunks/embeddings to PostgreSQL.
    """

    _REQUIRED_VARS = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    @classmethod
    def from_env(cls) -> ReferenceRepository:
        required_non_empty = [k for k in cls._REQUIRED_VARS if k != "DB_PASSWORD"]
        missing = [k for k in required_non_empty if not os.environ.get(k)]
        if missing:
            raise EnvironmentError(f"Missing required environment variable(s): {', '.join(missing)}")

        password = os.environ.get("DB_PASSWORD") or None

        conn = psycopg.connect(
            host=os.environ["DB_HOST"],
            port=int(os.environ["DB_PORT"]),
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=password,
            autocommit=False,
        )
        return cls(conn)

    @classmethod
    @contextmanager
    def connect(cls) -> Generator[ReferenceRepository, None, None]:
        repo = cls.from_env()
        try:
            yield repo
            repo._conn.commit()
        except Exception:
            repo._conn.rollback()
            raise
        finally:
            repo._conn.close()

    def save_document_with_chunks(
        self,
        document_meta: dict,
        loaded: LoadedDocument,
        chunks: list[ReferenceChunk],
        embeddings: list[list[float]] | None = None,
    ) -> dict:
        """
        Upsert reference document and replace chunks + embeddings atomically.
        Independent transaction with explicit rollback on error.
        """
        document_key = document_meta.get("document_key") or document_meta["document_id"]
        sequence_no: int = document_meta["sequence"]
        title: str = document_meta["title"]
        approval_status: str = document_meta.get("approval_status", "approved")

        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO reference_documents
                        (document_key, sequence_no, title, source_filename, source_checksum, approval_status)
                    VALUES
                        (%(document_key)s, %(sequence_no)s, %(title)s, %(source_filename)s, %(source_checksum)s, %(approval_status)s)
                    ON CONFLICT (document_key) DO UPDATE
                        SET title = EXCLUDED.title,
                            source_filename = EXCLUDED.source_filename,
                            source_checksum = EXCLUDED.source_checksum,
                            approval_status = EXCLUDED.approval_status,
                            updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                    """,
                    {
                        "document_key": document_key,
                        "sequence_no": sequence_no,
                        "title": title,
                        "source_filename": loaded.filename,
                        "source_checksum": loaded.checksum,
                        "approval_status": approval_status,
                    },
                )
                row = cur.fetchone()
                document_id: int = row[0]  # type: ignore[index]

                cur.execute("DELETE FROM reference_chunks WHERE document_id = %s", (document_id,))
                deleted_chunks: int = cur.rowcount

                embedded_count = 0
                if chunks:
                    chunk_params = []
                    for idx, c in enumerate(chunks):
                        emb_str = None
                        if embeddings and idx < len(embeddings) and embeddings[idx]:
                            emb_str = "[" + ",".join(str(x) for x in embeddings[idx]) + "]"
                            embedded_count += 1

                        chunk_params.append({
                            "document_id": document_id,
                            "field_id": c.field_id,
                            "field_title": c.field_title,
                            "chunk_index": c.chunk_index,
                            "content": c.content,
                            "char_count": c.char_count,
                            "embedding": emb_str,
                        })

                    cur.executemany(
                        """
                        INSERT INTO reference_chunks
                            (document_id, field_id, field_title, chunk_index, content, char_count, embedding)
                        VALUES
                            (%(document_id)s, %(field_id)s, %(field_title)s, %(chunk_index)s, %(content)s, %(char_count)s, %(embedding)s::vector)
                        """,
                        chunk_params,
                    )

            self._conn.commit()

            return {
                "document_id": document_key,
                "deleted_chunks": deleted_chunks,
                "inserted_chunks": len(chunks),
                "embedded_chunks": embedded_count,
            }
        except Exception:
            self._conn.rollback()
            raise

