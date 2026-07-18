"""UI helper tests: citation linkification, claim flagging, eval-results loading."""

import json

import pandas as pd

from patchcontext.ui_helpers import (
    flag_unsupported,
    linkify_citations,
    load_eval_results,
    load_ref_urls,
)

REF_URLS = {
    "pr:1276": "https://github.com/fastapi/fastapi/pull/1276",
    "issue:4433": "https://github.com/fastapi/fastapi/issues/4433",
    "commit:ab12cd3": "https://github.com/fastapi/fastapi/commit/ab12cd3",
}


def test_linkify_all_citation_forms() -> None:
    text = "Adopted in [PR #1276], see [Issue #4433] and [commit ab12cd3ff00]."
    out = linkify_citations(text, REF_URLS)
    assert "[[PR #1276]](https://github.com/fastapi/fastapi/pull/1276)" in out
    assert "[[Issue #4433]](https://github.com/fastapi/fastapi/issues/4433)" in out
    assert "[[commit ab12cd3ff00]](https://github.com/fastapi/fastapi/commit/ab12cd3)" in out


def test_linkify_leaves_unknown_refs_plain() -> None:
    text = "Mentioned in [PR #99999]."
    assert linkify_citations(text, REF_URLS) == text


def test_flag_unsupported_marks_claims() -> None:
    text = "First claim [PR #1276]. Second claim [Issue #4433]."
    out = flag_unsupported(text, ["Second claim [Issue #4433]."])
    assert out == "First claim [PR #1276]. **⚠️ Second claim [Issue #4433].**"


def test_load_ref_urls_normalizes(tmp_path) -> None:
    pd.DataFrame([
        {"source_type": "pr", "ref_id": "10", "url": "https://x/pull/10"},
        {"source_type": "pr", "ref_id": "10", "url": "https://x/pull/10"},  # dup row
        {"source_type": "commit", "ref_id": "AB12CD3", "url": "https://x/commit/ab"},
    ]).to_parquet(tmp_path / "metadata.parquet", index=False)
    urls = load_ref_urls(tmp_path)
    assert urls == {"pr:10": "https://x/pull/10", "commit:ab12cd3": "https://x/commit/ab"}


def test_load_eval_results_empty_and_populated(tmp_path) -> None:
    assert load_eval_results(tmp_path) == []
    (tmp_path / "ragas_1.json").write_text(json.dumps(
        {"metrics": {"faithfulness": 0.9}, "judge_model": "gemini", "n_questions": 50}
    ))
    (tmp_path / "junk.json").write_text("not json")
    (tmp_path / "other.json").write_text(json.dumps({"no_metrics": True}))
    results = load_eval_results(tmp_path)
    assert len(results) == 1
    assert results[0]["name"] == "ragas_1"
    assert results[0]["metrics"]["faithfulness"] == 0.9
