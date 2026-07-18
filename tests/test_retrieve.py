"""Retrieval tests: retriever loading, MMR diversification, reranker ordering.

No model downloads: query embedding is monkeypatched and the cross-encoder is
replaced with a keyword-overlap fake.
"""

import numpy as np
import pandas as pd
import pytest

from patchcontext.retrieve import reranker as reranker_module
from patchcontext.retrieve.retriever import RetrievedChunk, Retriever
from patchcontext.retrieve.reranker import rerank

DIM = 8


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.zeros(DIM, dtype="float32")
    arr[: len(vec)] = vec
    return arr / np.linalg.norm(arr)


# row 0 and row 1 are exact duplicates; row 2 is moderately relevant & distinct
VECS = np.stack([
    _unit([1, 0, 0]),      # 0: "pydantic v2 upgrade"
    _unit([1, 0, 0]),      # 1: duplicate of 0
    _unit([0, 1, 0]),      # 2: "pydantic validation change"
    _unit([0, 0, 1]),      # 3: unrelated
    _unit([0.1, 0, 1]),    # 4: unrelated
])
QUERY_VEC = _unit([1, 0.6, 0])  # relevant to 0/1, moderately to 2


@pytest.fixture()
def index_dir(tmp_path):
    np.save(tmp_path / "embeddings.npy", VECS)
    rows = [
        {"source_type": "pr", "ref_id": str(100 + i), "url": f"https://x/{i}",
         "author": "a", "date": "2026-01-01", "title": f"t{i}", "section": "body",
         "text": f"chunk text {i}"}
        for i in range(len(VECS))
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "metadata.parquet", index=False)
    return tmp_path


@pytest.fixture()
def fixed_query_embedding(monkeypatch):
    monkeypatch.setattr(
        "patchcontext.retrieve.retriever.embed_texts",
        lambda texts, **kw: np.stack([QUERY_VEC for _ in texts]),
    )


def test_similarity_top_returns_scores_and_metadata(index_dir, fixed_query_embedding):
    results = Retriever(index_dir).similarity_top("q", k=3)
    assert len(results) == 3
    assert results[0].score >= results[1].score >= results[2].score
    assert {results[0].metadata["ref_id"], results[1].metadata["ref_id"]} == {"100", "101"}
    assert set(results[0].metadata) >= {"source_type", "ref_id", "url", "title", "section"}


def test_mmr_search_demotes_duplicates(index_dir, fixed_query_embedding):
    results = Retriever(index_dir).search("q", top_k=5, mmr_lambda=0.6, select_k=3)
    assert len(results) == 3
    picked = [r.metadata["row_id"] for r in results]
    # Pure similarity order is [0, 1, 2] (rows 0/1 identical). MMR must pick a
    # duplicate first, then prefer the distinct-but-relevant row 2 over the copy.
    assert picked[0] in (0, 1)
    assert picked[1] == 2
    assert all(r.score > 0 for r in results)  # cosine scores carried through


def test_retriever_rejects_mismatched_artifacts(index_dir):
    frame = pd.read_parquet(index_dir / "metadata.parquet")
    frame.iloc[:3].to_parquet(index_dir / "metadata.parquet", index=False)
    with pytest.raises(ValueError, match="mismatch"):
        Retriever(index_dir)


class _FakeCrossEncoder:
    """Scores by word overlap with the query — deterministic and offline."""

    def predict(self, pairs, show_progress_bar=False):
        return [
            len(set(query.split()) & set(text.split())) for query, text in pairs
        ]


def test_rerank_reorders_and_cuts(monkeypatch):
    monkeypatch.setattr(reranker_module, "_model", _FakeCrossEncoder())
    candidates = [
        RetrievedChunk(text="nothing relevant here", score=0.9, metadata={"ref_id": "1"}),
        RetrievedChunk(text="pydantic v2 adopted for validation", score=0.5, metadata={"ref_id": "2"}),
        RetrievedChunk(text="pydantic v2 mentioned", score=0.4, metadata={"ref_id": "3"}),
    ]
    result = rerank("why pydantic v2 adopted", candidates, top_k=2)
    assert [r.metadata["ref_id"] for r in result] == ["2", "3"]  # cosine order was 1,2,3
    assert result[0].score > result[1].score  # cross-encoder scores replaced cosine


def test_rerank_empty_is_empty():
    assert rerank("q", [], top_k=5) == []
