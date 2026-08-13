-- 002_add_pgvector.sql
-- Enables vector extension and adds 384-dimensional vector embedding column to reference_chunks

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE reference_chunks
    ADD COLUMN IF NOT EXISTS embedding vector(384);

CREATE INDEX IF NOT EXISTS idx_reference_chunks_embedding
    ON reference_chunks USING hnsw (embedding vector_cosine_ops);
