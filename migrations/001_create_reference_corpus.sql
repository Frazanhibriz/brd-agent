CREATE TABLE reference_documents (
    id BIGSERIAL PRIMARY KEY,

    document_key VARCHAR(150) NOT NULL UNIQUE,
    sequence_no SMALLINT NOT NULL UNIQUE,

    title TEXT NOT NULL,
    source_filename TEXT NOT NULL,

    approval_status VARCHAR(20)
        NOT NULL DEFAULT 'approved',

    source_checksum VARCHAR(64),

    created_at TIMESTAMPTZ
        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMPTZ
        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_reference_document_sequence
        CHECK (sequence_no >= 1),

    CONSTRAINT chk_reference_document_approval
        CHECK (approval_status = 'approved')
);


CREATE TABLE reference_chunks (
    id BIGSERIAL PRIMARY KEY,

    document_id BIGINT NOT NULL,

    field_id VARCHAR(20) NOT NULL,
    field_title TEXT NOT NULL,

    chunk_index INTEGER
        NOT NULL DEFAULT 0,

    content TEXT NOT NULL,

    char_count INTEGER NOT NULL,

    metadata JSONB
        NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ
        NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_reference_chunk_document
        FOREIGN KEY (document_id)
        REFERENCES reference_documents(id)
        ON DELETE CASCADE,

    CONSTRAINT chk_reference_chunk_index
        CHECK (chunk_index >= 0),

    CONSTRAINT uq_reference_document_field_chunk
        UNIQUE (
            document_id,
            field_id,
            chunk_index
        )
);


CREATE INDEX idx_reference_chunks_field_id
    ON reference_chunks(field_id);

CREATE INDEX idx_reference_chunks_document_id
    ON reference_chunks(document_id);