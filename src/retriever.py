"""BM25 retrieval utilities for the support agent project."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def simple_tokenize(text: str) -> list[str]:
    """Lowercase and split text into simple alphanumeric tokens."""

    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class RetrievalResult:
    record: dict[str, Any]
    score: float


class BM25Retriever:
    """
    BM25 is a classic and strong Information Retrieval baseline.

    It scores documents using term matching, term frequency, inverse document
    frequency, and document length normalization. Compared with a very basic
    TF-IDF cosine baseline, BM25 is usually a stronger and more standard choice
    for a Master-level IR project.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        text_builder: Callable[[dict[str, Any]], str],
    ) -> None:
        self.records = records
        self.text_builder = text_builder
        self.tokenized_corpus = [simple_tokenize(text_builder(record)) for record in records]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        """Return the top-k records with BM25 scores."""

        if not self.records or not self.bm25:
            return []

        query_tokens = simple_tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_items = sorted(
            zip(self.records, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[RetrievalResult] = []
        for record, score in ranked_items[:top_k]:
            results.append(RetrievalResult(record=record, score=float(score)))

        return results
