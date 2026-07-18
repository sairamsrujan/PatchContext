"""Checkpointed embedding: resume, invalidation, order restoration, budgets."""

import numpy as np
import pytest

from patchcontext.index import build_index as bi


@pytest.fixture()
def fake_embed(monkeypatch):
    """Deterministic 4-dim 'embedding': encodes text length; counts calls."""
    calls = {"texts": 0}

    def embed(texts, batch_size=None, show_progress=True, prompt_name=None):
        calls["texts"] += len(texts)
        return np.array([[len(t), 1.0, 0.0, 0.0] for t in texts], dtype="float32")

    monkeypatch.setattr(bi, "embed_texts", embed)
    return calls


# Single-'word' texts: approx_tokens == chars//4 (min 1).
# Sorted desc by length: 300(75tok), 120(30), 60(15), 40(10), 7(1), 5(1), 2(1).
# With token_budget=40 the slices are [0:1], [1:2], [2:7].
TEXTS = ["a" * n for n in (5, 300, 40, 7, 120, 60, 2)]
BUDGET = 40


def test_slice_boundaries_respect_budget() -> None:
    sorted_texts = sorted(TEXTS, key=len, reverse=True)
    assert bi._slice_boundaries(sorted_texts, BUDGET) == [0, 1, 2, 7]


def test_rows_align_to_input_order(tmp_path, fake_embed):
    result = bi.embed_with_checkpoints(TEXTS, tmp_path, token_budget=BUDGET)
    assert result.shape == (len(TEXTS), 4)
    # row i encodes len(TEXTS[i]) despite length-sorted processing
    assert [int(r[0]) for r in result] == [len(t) for t in TEXTS]


def test_resume_skips_completed_slices(tmp_path, fake_embed):
    bi.embed_with_checkpoints(TEXTS, tmp_path, token_budget=BUDGET)
    assert fake_embed["texts"] == len(TEXTS)
    # second run: all slices cached -> zero new embedding work
    result = bi.embed_with_checkpoints(TEXTS, tmp_path, token_budget=BUDGET)
    assert fake_embed["texts"] == len(TEXTS)
    assert [int(r[0]) for r in result] == [len(t) for t in TEXTS]


def test_partial_resume(tmp_path, fake_embed):
    bi.embed_with_checkpoints(TEXTS, tmp_path, token_budget=BUDGET)
    # simulate a crash that lost the last slice ([2:7] -> 5 texts)
    parts = sorted((tmp_path / "embed_parts").glob("part_*.npy"))
    parts[-1].unlink()
    before = fake_embed["texts"]
    bi.embed_with_checkpoints(TEXTS, tmp_path, token_budget=BUDGET)
    assert fake_embed["texts"] == before + 5


def test_corpus_change_invalidates(tmp_path, fake_embed):
    bi.embed_with_checkpoints(TEXTS, tmp_path, token_budget=BUDGET)
    changed = TEXTS[:-1] + ["something new"]
    result = bi.embed_with_checkpoints(changed, tmp_path, token_budget=BUDGET)
    assert fake_embed["texts"] == len(TEXTS) + len(changed)  # all slices redone
    assert [int(r[0]) for r in result] == [len(t) for t in changed]


def test_budget_change_invalidates(tmp_path, fake_embed):
    bi.embed_with_checkpoints(TEXTS, tmp_path, token_budget=BUDGET)
    bi.embed_with_checkpoints(TEXTS, tmp_path, token_budget=BUDGET * 4)
    assert fake_embed["texts"] == 2 * len(TEXTS)  # boundaries shifted -> redone
