"""Vector similarity search + MMR diversification (pure NumPy).

At ~12k L2-normalized vectors, exact cosine search is one matrix-vector
product — no ANN library needed. Keeping FAISS out of the serving process
also sidesteps a real crash: on x86-64 Linux, FAISS and torch each bundle
their own OpenMP runtime and the duplicates segfault at thread teardown.
The offline build still writes ``faiss.index``; serving reads
``embeddings.npy``.

Owned here: cosine top-``top_k`` → MMR (λ) selects a diverse ``select_k``
subset. The cross-encoder reranker downstream cuts that to the final 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings

from patchcontext.config import settings
from patchcontext.index.embedder import embed_texts

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with its score and metadata."""

    text: str
    score: float  # cosine similarity (retriever) or cross-encoder score (reranker)
    metadata: dict[str, Any]  # source_type, ref_id, url, author, date, title, section


class _QwenEmbeddingsAdapter(Embeddings):
    """LangChain Embeddings interface over our in-process Qwen3 embedder
    (used by the RAGAs eval; retrieval itself calls embed_texts directly)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts, show_progress=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text], show_progress=False, prompt_name="query")[0].tolist()


def _mmr(
    query: "np.ndarray",
    candidate_vecs: "np.ndarray",
    candidate_scores: "np.ndarray",
    k: int,
    lambda_mult: float,
) -> list[int]:
    """Maximal Marginal Relevance over pre-fetched candidates.

    Greedily selects ``k`` local indices (into ``candidate_vecs``) balancing
    relevance to ``query`` against redundancy with already-selected items —
    the same objective LangChain's MMR uses.
    """
    import numpy as np

    selected: list[int] = [int(np.argmax(candidate_scores))]
    while len(selected) < min(k, len(candidate_vecs)):
        selected_vecs = candidate_vecs[selected]
        redundancy = (candidate_vecs @ selected_vecs.T).max(axis=1)  # max sim to any selected
        mmr = lambda_mult * candidate_scores - (1 - lambda_mult) * redundancy
        mmr[selected] = -np.inf  # never re-pick
        selected.append(int(np.argmax(mmr)))
    return selected


class Retriever:
    """Loads ``embeddings.npy`` + ``metadata.parquet`` and serves cosine + MMR search."""

    def __init__(self, index_dir: Path) -> None:
        import numpy as np
        import pandas as pd

        self._vectors: np.ndarray = np.load(index_dir / "embeddings.npy")  # (n, dim), L2-normalized
        self._meta = pd.read_parquet(index_dir / "metadata.parquet")
        if len(self._vectors) != len(self._meta):
            raise ValueError(
                f"index/metadata mismatch: {len(self._vectors)} vectors vs {len(self._meta)} rows"
            )
        logger.info("retriever ready: %d chunks from %s", len(self._vectors), index_dir)

    def _row_to_chunk(self, row_id: int, score: float) -> RetrievedChunk:
        row = self._meta.iloc[row_id]
        return RetrievedChunk(
            text=row["text"],
            score=float(score),
            metadata={
                "row_id": int(row_id),
                "source_type": row["source_type"],
                "ref_id": row["ref_id"],
                "url": row["url"],
                "author": row["author"],
                "date": row["date"],
                "title": row["title"],
                "section": row["section"],
            },
        )

    def _query_vector(self, query: str) -> "np.ndarray":
        return embed_texts([query], show_progress=False, prompt_name="query")[0]

    def similarity_top(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Plain cosine top-``k`` — the pre-MMR/pre-rerank baseline."""
        import numpy as np

        scores = self._vectors @ self._query_vector(query)
        top = np.argsort(-scores)[:k]
        return [self._row_to_chunk(int(i), scores[i]) for i in top]

    def search(
        self,
        query: str,
        top_k: int = 20,
        mmr_lambda: float = 0.6,
        select_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Cosine top-``top_k``, then MMR (``mmr_lambda``) selects a diverse
        ``select_k`` subset (default ``settings.mmr_candidates``). Returned
        scores are the original cosine similarities.
        """
        import numpy as np

        select_k = select_k or settings.mmr_candidates
        qvec = self._query_vector(query)
        scores = self._vectors @ qvec
        pool = np.argsort(-scores)[:top_k]  # global row ids of the top_k
        pool_vecs = self._vectors[pool]
        pool_scores = scores[pool]
        chosen_local = _mmr(qvec, pool_vecs, pool_scores, select_k, mmr_lambda)
        return [self._row_to_chunk(int(pool[i]), pool_scores[i]) for i in chosen_local]
