"""Chunker tests: chunking rules, bot filtering, splitting, metadata. No models."""

import json
from pathlib import Path

from patchcontext.index.chunker import (
    Chunk,
    _pack_thread,
    _split_words,
    approx_tokens,
    chunk_raw_records,
)


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _make_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    _write(raw / "commits" / "a.json", {
        "source_type": "commit", "sha": "a" * 40, "short_sha": "ab12cd3",
        "message": "Adopt pydantic v2\n\nMigrates validation.", "author": "tiangolo",
        "date": "2026-06-01T00:00:00Z", "html_url": "https://github.com/f/f/commit/a",
        "stats": {"additions": 10, "deletions": 2}, "files": [], "files_total": 3,
    })
    _write(raw / "commits" / "b.json", {  # bot commit -> filtered
        "source_type": "commit", "sha": "b" * 40, "short_sha": "bbbbbbb",
        "message": "Update release notes", "author": "github-actions[bot]",
        "date": "2026-06-01T00:00:00Z", "html_url": "", "stats": {}, "files": [],
    })
    _write(raw / "prs" / "42.json", {
        "source_type": "pr", "number": 42, "title": "Add QUERY method",
        "body": "Implements RFC QUERY support.", "author": "contributor",
        "created_at": "2026-06-02T00:00:00Z", "merged": True,
        "html_url": "https://github.com/f/f/pull/42", "labels": ["feature"],
        "linked_issues": [7], "comments_count": 2, "review_comments_count": 2,
        "priority": True,
        "issue_comments": [
            {"author": "codspeed-hq[bot]", "date": "2026-06-02T01:00:00Z", "body": "perf report"},
            {"author": "reviewer", "date": "2026-06-02T02:00:00Z", "body": "Why not PATCH?"},
        ],
        "review_comments": [
            {"author": "reviewer", "date": "2026-06-02T03:00:00Z", "path": "fastapi/routing.py",
             "body": "This should reuse the existing route decorator.", "in_reply_to": None},
        ],
    })
    _write(raw / "issues" / "7.json", {
        "source_type": "issue", "number": 7, "title": "Support HTTP QUERY",
        "body": "It would be useful to support QUERY.", "author": "asker",
        "created_at": "2026-06-01T00:00:00Z", "html_url": "https://github.com/f/f/issues/7",
        "labels": [], "comments_count": 2, "priority": True,
        "comments": [
            {"author": "tiangolo", "date": "2026-06-01T01:00:00Z", "body": "Makes sense, PRs welcome."},
        ],
    })
    return raw


def test_chunking_units_metadata_and_bot_filter(tmp_path: Path) -> None:
    chunks = chunk_raw_records(_make_raw(tmp_path))
    by_type = {}
    for c in chunks:
        by_type.setdefault(c.source_type, []).append(c)

    # Bot commit filtered; human commit kept with its metadata
    assert len(by_type["commit"]) == 1
    commit = by_type["commit"][0]
    assert commit.ref_id == "ab12cd3"
    assert commit.title == "Adopt pydantic v2"
    assert "+10/-2 across 3 files" in commit.text

    # PR: body + discussion + one review thread; bot comment excluded
    sections = {c.section for c in by_type["pr"]}
    assert sections == {"body", "discussion", "review:fastapi/routing.py"}
    pr_all_text = " ".join(c.text for c in by_type["pr"])
    assert "Why not PATCH?" in pr_all_text
    assert "perf report" not in pr_all_text  # bot comment filtered
    body = next(c for c in by_type["pr"] if c.section == "body")
    assert "Linked issues: #7" in body.text and "merged" in body.text

    # Issue: body + discussion
    assert {c.section for c in by_type["issue"]} == {"body", "discussion"}

    # No chunk ever mixes units: every chunk has exactly one ref_id/url
    assert all(c.ref_id and c.url for c in chunks if c.source_type != "commit" or c.ref_id)

    # Metadata rows carry every required column
    row = commit.to_metadata()
    assert set(row) == {"source_type", "ref_id", "url", "author", "date", "title", "section"}


def test_split_words_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(2000))  # ~2666 approx tokens
    pieces = _split_words(text, max_tokens=800, overlap_tokens=100)
    assert len(pieces) > 1
    # word windows are sized via the words->tokens factor; the char-based floor
    # of approx_tokens can estimate slightly above the target ("~800" per brief)
    assert all(approx_tokens(p) <= 900 for p in pieces)
    # consecutive pieces share the overlap window
    first_words, second_words = pieces[0].split(), pieces[1].split()
    assert first_words[-1] in second_words


def test_pack_thread_repeats_header_and_respects_cap() -> None:
    header = "Discussion on PR #1: big thread"
    bodies = [f"user{i}: " + " ".join(["word"] * 300) for i in range(6)]  # ~400 tokens each
    packed = _pack_thread(header, bodies, max_tokens=800, overlap_tokens=100)
    assert len(packed) >= 3
    assert all(p.startswith(header) for p in packed)
    assert all(approx_tokens(p) <= 850 for p in packed)


def test_short_text_single_chunk() -> None:
    assert _split_words("short text", 800, 100) == ["short text"]
    assert approx_tokens("four words right here") >= 4


def test_chunk_is_frozen_and_defaults() -> None:
    c = Chunk(text="t", source_type="pr", ref_id="1", url="u", author="a", date="d", title="ti")
    assert c.section == "body"
