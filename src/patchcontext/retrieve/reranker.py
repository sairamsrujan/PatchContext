"""Cross-encoder reranking with BAAI/bge-reranker-base.

Scores (query, chunk-text) pairs jointly — far more precise than bi-encoder
cosine — and cuts the MMR-diversified candidates down to the top-5 fed to the
LLM. The model is lazy-loaded once per process.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from patchcontext.config import settings
from patchcontext.retrieve.retriever import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_model: "CrossEncoder | None" = None


def get_model() -> "CrossEncoder":
    """Load the reranker once per process (downloads weights on first use)."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        logger.info(
            "Loading reranker %s (device: %s) ...",
            settings.reranker_model, settings.model_device,
        )
        _model = CrossEncoder(settings.reranker_model, device=settings.model_device)
        logger.info("Reranker ready")
    return _model


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
    """Re-score ``candidates`` against ``query`` with the cross-encoder and
    return the top ``top_k``, with cross-encoder scores replacing cosine.
    """
    if not candidates:
        return []
    from patchcontext.inference import run_inference

    model = get_model()
    pairs = [(query, c.text) for c in candidates]
    scores = run_inference(lambda: model.predict(pairs, show_progress_bar=False))
    ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)
    return [
        RetrievedChunk(text=c.text, score=float(s), metadata=c.metadata)
        for c, s in ranked[:top_k]
    ]
