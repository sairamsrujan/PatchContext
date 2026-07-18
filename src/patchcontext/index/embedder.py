"""Batch embedding with Qwen/Qwen3-Embedding-0.6B via sentence-transformers.

The model runs locally/in-process; embeddings are L2-normalized so FAISS inner
product equals cosine similarity. Heavy imports stay inside functions so
importing this module is cheap (and Streamlit caching stays clean).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from patchcontext.config import settings

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: "SentenceTransformer | None" = None


def get_model() -> "SentenceTransformer":
    """Load the embedding model once per process (downloads weights on first use)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        # torch threading stays at default: with FAISS out of the serving
        # process it's the only OpenMP user (OMP_NUM_THREADS caps it in the
        # container).
        logger.info(
            "Loading embedding model %s (device: %s) ...",
            settings.embedding_model, settings.model_device,
        )
        _model = SentenceTransformer(settings.embedding_model, device=settings.model_device)
        _model.max_seq_length = settings.embed_max_seq_tokens
        logger.info(
            "Embedding model ready (device: %s, dim: %d, max_seq: %d)",
            _model.device,
            _model.get_sentence_embedding_dimension(),
            _model.max_seq_length,
        )
    return _model


def embed_texts(
    texts: Sequence[str],
    batch_size: int | None = None,
    show_progress: bool = True,
    prompt_name: str | None = None,
) -> "np.ndarray":
    """Embed ``texts`` in batches; returns an L2-normalized float32 array (n, dim).

    Pass ``prompt_name="query"`` for search queries — Qwen3 embedding models
    ship an instruction prompt for queries while documents are embedded plain.
    """
    from patchcontext.inference import run_inference

    model = get_model()
    if batch_size is None:
        batch_size = settings.embed_batch_size
    if prompt_name is not None and prompt_name not in getattr(model, "prompts", {}):
        logger.warning("Model has no '%s' prompt; embedding without one", prompt_name)
        prompt_name = None
    start = time.perf_counter()
    embeddings = run_inference(lambda: model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        prompt_name=prompt_name,
    ))
    elapsed = time.perf_counter() - start
    logger.info(
        "Embedded %d texts in %.1f s (%.1f texts/s)",
        len(texts), elapsed, len(texts) / max(elapsed, 1e-9),
    )
    return embeddings.astype("float32")
