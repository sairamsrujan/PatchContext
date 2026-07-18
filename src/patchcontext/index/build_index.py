"""FAISS IndexFlatIP build + parquet metadata sidecar.

Artifacts written to ``index_dir``:
- ``faiss.index``        — inner-product index over L2-normalized vectors (= cosine)
- ``metadata.parquet``   — one row per chunk, row order == FAISS vector id;
                           columns: chunk metadata + the chunk ``text`` itself
- ``stats.json``         — chunk counts, date range, model, build timestamp
                           (rendered in the Streamlit sidebar)

Embedding is checkpointed: texts are processed in slices, each slice saved to
``index_dir/embed_parts/`` as it completes, so an interrupted multi-hour CPU
run resumes instead of restarting. Texts are embedded in length-sorted order
(minimizes padding waste) and un-permuted before indexing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from patchcontext.config import settings
from patchcontext.index.chunker import Chunk, approx_tokens
from patchcontext.index.embedder import embed_texts

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Approx tokens per checkpoint slice. Sized so one slice embeds in ~5-10 min
# on CPU, bounding how much work any interruption can lose.
SLICE_TOKEN_BUDGET = 60_000


def _corpus_fingerprint(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8", errors="replace"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _slice_boundaries(sorted_texts: list[str], token_budget: int) -> list[int]:
    """Start indices of token-budgeted slices, plus the final end index."""
    boundaries = [0]
    running = 0
    for idx, text in enumerate(sorted_texts):
        tokens = approx_tokens(text)
        if running + tokens > token_budget and idx > boundaries[-1]:
            boundaries.append(idx)
            running = 0
        running += tokens
    boundaries.append(len(sorted_texts))
    return boundaries


def embed_with_checkpoints(
    texts: list[str], index_dir: Path, token_budget: int = SLICE_TOKEN_BUDGET
) -> "np.ndarray":
    """Embed ``texts`` in resumable slices; returns rows aligned to ``texts`` order.

    Texts embed in length-sorted order (padding efficiency), packed into
    slices of ~``token_budget`` approx tokens each, saved to
    ``index_dir/embed_parts/`` as they complete. A manifest fingerprints the
    corpus and the budget: if either changes, stale parts are discarded; a
    cached part with an unexpected row count is re-embedded. The caller
    removes the parts directory after the index is written.
    """
    import numpy as np

    part_dir = index_dir / "embed_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = part_dir / "manifest.json"
    manifest = {"fingerprint": _corpus_fingerprint(texts), "token_budget": token_budget}
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        logger.warning("corpus or slice budget changed; discarding old embed parts")
        shutil.rmtree(part_dir)
        part_dir.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest))

    order = np.argsort([-len(t) for t in texts], kind="stable")  # longest first
    sorted_texts = [texts[i] for i in order]
    boundaries = _slice_boundaries(sorted_texts, token_budget)

    parts: list[np.ndarray] = []
    for start, end in zip(boundaries, boundaries[1:]):
        part_path = part_dir / f"part_{start:06d}.npy"
        if part_path.exists():
            cached = np.load(part_path)
            if len(cached) == end - start:
                parts.append(cached)
                continue
            logger.warning("part %s has %d rows, expected %d; re-embedding", part_path.name, len(cached), end - start)
        embedded = embed_texts(sorted_texts[start:end], show_progress=False)
        tmp = part_path.with_suffix(".npy.tmp")
        with open(tmp, "wb") as fh:  # file handle: np.save must not append ".npy"
            np.save(fh, embedded)
        tmp.rename(part_path)
        parts.append(embedded)
        logger.info("embedding checkpoint: %d/%d texts done", end, len(sorted_texts))

    stacked = np.vstack(parts)
    unsorted = np.empty_like(stacked)
    unsorted[order] = stacked  # row i of the result corresponds to texts[i]
    return unsorted


def build_index(chunks: list[Chunk], index_dir: Path, batch_size: int | None = None) -> None:
    """Embed ``chunks`` and write ``faiss.index`` + ``metadata.parquet`` + ``stats.json``."""
    # torch MUST load before faiss: both bundle libomp.dylib on macOS, and
    # faiss-first ordering segfaults CPU inference at an OpenMP barrier.
    import torch  # noqa: F401
    import faiss
    import pandas as pd

    index_dir.mkdir(parents=True, exist_ok=True)

    embeddings = embed_with_checkpoints([c.text for c in chunks], index_dir)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    index_path = index_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    # The serving Retriever reads embeddings.npy (pure NumPy search, no FAISS in
    # the query path — see retriever.py). faiss.index is kept for compatibility.
    import numpy as np

    np.save(index_dir / "embeddings.npy", embeddings.astype("float32"))

    frame = pd.DataFrame([{**c.to_metadata(), "text": c.text} for c in chunks])
    metadata_path = index_dir / "metadata.parquet"
    frame.to_parquet(metadata_path, index=False)

    dates = sorted(d for d in frame["date"] if d)
    stats = {
        "chunks": len(chunks),
        "dim": int(embeddings.shape[1]),
        "by_source_type": frame["source_type"].value_counts().to_dict(),
        "date_min": dates[0][:10] if dates else None,
        "date_max": dates[-1][:10] if dates else None,
        "embedding_model": settings.embedding_model,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (index_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    shutil.rmtree(index_dir / "embed_parts", ignore_errors=True)  # checkpoints no longer needed

    logger.info(
        "index built: %d vectors (dim %d) | faiss.index %.1f MB | metadata.parquet %.1f MB",
        index.ntotal,
        embeddings.shape[1],
        index_path.stat().st_size / 1e6,
        metadata_path.stat().st_size / 1e6,
    )
