"""
retrieval/lexical_baseline.py
=============================
AI2-3 Standalone Lexical Baseline Search (BM25 Keyword Matching).

IMPORTANT:
Strictly an AI2-3 baseline for comparison evidence and offline testing.
It is NOT used by the production semantic retrieval engine (AI2-4).
Do NOT fuse, fallback, or rerank with pgvector search in production runtime.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from .models import LexicalDocumentChunk, SearchResult


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


class LexicalBaseline:
    """Standalone BM25 Lexical Matching Engine for AI2-3 Baseline."""

    def __init__(
        self,
        chunks: Sequence[LexicalDocumentChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b

        self._doc_tokens: list[list[str]] = []
        self._doc_lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._num_docs = len(self.chunks)
        self._avg_dl = 0.0

        self._index_corpus()

    def _index_corpus(self) -> None:
        if not self.chunks:
            return

        total_length = 0
        for chunk in self.chunks:
            text = f"{chunk.field_title} {chunk.content}"
            tokens = _tokenize(text)
            self._doc_tokens.append(tokens)
            doc_len = len(tokens)
            self._doc_lengths.append(doc_len)
            total_length += doc_len

            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._df[token] += 1

        self._avg_dl = total_length / self._num_docs if self._num_docs > 0 else 0.0

    def _idf(self, term: str) -> float:
        n_q = self._df.get(term, 0)
        if n_q == 0:
            return 0.0
        return math.log(1.0 + (self._num_docs - n_q + 0.5) / (n_q + 0.5))

    def search_lexical(
        self,
        query: str,
        field_id: str | None = None,
        top_k: int = 3,
    ) -> list[SearchResult]:
        query_tokens = _tokenize(query)
        if not query_tokens or self._num_docs == 0:
            return []

        scores: list[tuple[float, int]] = []

        for idx, chunk in enumerate(self.chunks):
            if field_id is not None and chunk.field_id != field_id:
                continue

            doc_tokens = self._doc_tokens[idx]
            doc_len = self._doc_lengths[idx]
            if doc_len == 0:
                continue

            tf_counter = Counter(doc_tokens)
            score = 0.0

            for q_term in query_tokens:
                tf = tf_counter.get(q_term, 0)
                if tf == 0:
                    continue

                idf = self._idf(q_term)
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._avg_dl))
                score += idf * (num / den)

            if score > 0:
                scores.append((score, idx))

        scores.sort(key=lambda item: item[0], reverse=True)

        results: list[SearchResult] = []
        for score, idx in scores[:top_k]:
            c = self.chunks[idx]
            results.append(
                SearchResult(
                    document_key=c.document_key,
                    document_title=c.document_title,
                    field_id=c.field_id,
                    field_title=c.field_title,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    similarity_score=round(score, 4),
                )
            )

        return results
