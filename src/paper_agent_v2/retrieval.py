from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from paper_agent_v2.parser import PaperChunk


@dataclass(slots=True)
class RetrievalHit:
    chunk: PaperChunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


class HybridRetriever:
    """Deterministic BM25 + cosine RRF reference implementation.

    Production persistence is backed by pgvector/FTS; this implementation also
    keeps tests and local inspection independent from a running database.
    """

    def __init__(self, chunks: list[PaperChunk], embeddings: list[list[float]] | None = None) -> None:
        self.chunks = chunks
        self.embeddings = embeddings
        self.documents = [Counter(_tokens(chunk.text)) for chunk in chunks]
        self.document_frequency = Counter(token for document in self.documents for token in document)
        self.average_length = sum(sum(document.values()) for document in self.documents) / max(len(chunks), 1)

    def _bm25(self, query: str) -> list[tuple[int, float]]:
        scores: list[tuple[int, float]] = []
        query_tokens = _tokens(query)
        total = len(self.documents)
        for index, document in enumerate(self.documents):
            length = sum(document.values())
            score = 0.0
            for token in query_tokens:
                tf = document[token]
                if not tf:
                    continue
                df = self.document_frequency[token]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * length / max(self.average_length, 1)))
            scores.append((index, score))
        return sorted(scores, key=lambda item: (-item[1], item[0]))

    def search(
        self,
        query: str,
        *,
        query_embedding: list[float] | None = None,
        limit: int = 8,
        rrf_k: int = 60,
    ) -> list[RetrievalHit]:
        lexical = self._bm25(query)
        dense: list[tuple[int, float]] = []
        if query_embedding is not None and self.embeddings:
            qnorm = math.sqrt(sum(value * value for value in query_embedding)) or 1.0
            for index, vector in enumerate(self.embeddings):
                norm = math.sqrt(sum(value * value for value in vector)) or 1.0
                similarity = sum(a * b for a, b in zip(query_embedding, vector, strict=True)) / (qnorm * norm)
                dense.append((index, similarity))
            dense.sort(key=lambda item: (-item[1], item[0]))

        ranks: dict[int, dict[str, int]] = {}
        for rank, (index, _) in enumerate(lexical, start=1):
            ranks.setdefault(index, {})["lexical"] = rank
        for rank, (index, _) in enumerate(dense, start=1):
            ranks.setdefault(index, {})["dense"] = rank
        hits = []
        for index, item_ranks in ranks.items():
            score = sum(1 / (rrf_k + rank) for rank in item_ranks.values())
            hits.append(
                RetrievalHit(
                    chunk=self.chunks[index],
                    score=score,
                    dense_rank=item_ranks.get("dense"),
                    lexical_rank=item_ranks.get("lexical"),
                )
            )
        return sorted(hits, key=lambda item: (-item.score, item.chunk.id))[:limit]


def postgres_rrf_chunk_ids(
    session: Session,
    document_id: str,
    query: str,
    query_embedding: list[float],
    *,
    limit: int = 8,
    rrf_k: int = 60,
) -> list[str]:
    """Rank persisted pgvector and PostgreSQL FTS results with reciprocal-rank fusion."""
    if not session.bind or session.bind.dialect.name != "postgresql":
        raise RuntimeError("PostgreSQL RRF is only available on PostgreSQL")
    sql = text(
        """
        WITH dense AS (
          SELECT id, row_number() OVER (ORDER BY embedding <=> CAST(:embedding AS vector)) AS rank
          FROM chunks WHERE document_id = :document_id AND embedding IS NOT NULL LIMIT 50
        ), lexical AS (
          SELECT id, row_number() OVER (
            ORDER BY ts_rank_cd(to_tsvector('english', text), websearch_to_tsquery('english', :query)) DESC
          ) AS rank
          FROM chunks
          WHERE document_id = :document_id
            AND to_tsvector('english', text) @@ websearch_to_tsquery('english', :query)
          LIMIT 50
        )
        SELECT coalesce(dense.id, lexical.id) AS id,
               coalesce(1.0 / (:rrf_k + dense.rank), 0.0) +
               coalesce(1.0 / (:rrf_k + lexical.rank), 0.0) AS score
        FROM dense FULL OUTER JOIN lexical ON dense.id = lexical.id
        ORDER BY score DESC, id LIMIT :limit
        """
    )
    rows = session.execute(
        sql,
        {
            "document_id": document_id,
            "query": query,
            "embedding": str(query_embedding),
            "rrf_k": rrf_k,
            "limit": limit,
        },
    )
    return [str(row.id) for row in rows]
