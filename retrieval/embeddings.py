"""
retrieval/embeddings.py
========================
Generates 384-dimensional dense vector embeddings using fastembed.
"""

from __future__ import annotations

from typing import Sequence
from fastembed import TextEmbedding

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class EmbeddingGenerator:
    """Generates 384-dimensional dense vector embeddings using FastEmbed."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            return [0.0] * EMBEDDING_DIM
        embeddings = list(self.model.embed([text]))
        return [float(x) for x in embeddings[0]]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        non_empty_texts = [t if t and t.strip() else "empty" for t in texts]
        embeddings = list(self.model.embed(non_empty_texts))
        return [[float(x) for x in emb] for emb in embeddings]
