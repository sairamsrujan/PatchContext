"""Semantic-unit chunking of the raw GitHub cache.

Chunking rules:
- One semantic unit per document: a commit (message + stats), a PR (title +
  body; each review-comment thread becomes its own chunk), an issue (body +
  comment thread). Unrelated units are never merged into one chunk.
- Max ~800 tokens per chunk; 100-token word overlap only when splitting a
  single long text. Thread chunks repeat their header line for continuity.
- Every chunk carries: source_type, ref_id, url, author, date, title, section.

Filtering: bot-authored records and comments (``…[bot]``) are dropped —
release-notes commits, dependency bumps, and CI bot comments carry no design
history and would pollute the index.

Token counts are approximated (~4/3 tokens per word, floor of chars/4) to keep
the chunker dependency-free; ~800 tokens is a retrieval-quality target, not a
model limit (Qwen3 accepts far longer inputs).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SourceType = Literal["commit", "pr", "issue"]


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit with the metadata every chunk must carry."""

    text: str
    source_type: SourceType
    ref_id: str  # commit short SHA, PR number, or issue number (as string)
    url: str  # direct GitHub link
    author: str
    date: str  # ISO 8601
    title: str
    section: str = "body"  # body | discussion | review:<path> (+ "#N" split suffix)

    def to_metadata(self) -> dict[str, Any]:
        """Metadata row for the parquet sidecar (everything except ``text``)."""
        return {
            "source_type": self.source_type,
            "ref_id": self.ref_id,
            "url": self.url,
            "author": self.author,
            "date": self.date,
            "title": self.title,
            "section": self.section,
        }


def approx_tokens(text: str) -> int:
    """Cheap token estimate: max of ~4/3 tokens per word and chars/4."""
    return max(len(text.split()) * 4 // 3, len(text) // 4, 1)


def _is_bot(author: str) -> bool:
    return author.endswith("[bot]")


def _split_words(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split one long text into word windows with overlap; short text passes through.

    The window is sized by the text's measured tokens-per-word density, so
    code-heavy content (long identifiers, URLs) gets proportionally smaller
    word windows than prose and still lands near ``max_tokens``.
    """
    total = approx_tokens(text)
    if total <= max_tokens:
        return [text]
    words = text.split()
    density = total / max(len(words), 1)
    max_words = max(int(max_tokens / density), 25)
    overlap_words = min(int(overlap_tokens / density), max_words // 4)
    step = max_words - overlap_words
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), step)]


def _sectioned(pieces: list[str], base: str) -> list[tuple[str, str]]:
    """Pair each piece with its section label (``base``, ``base#2``, ...)."""
    return [(piece, base if i == 0 else f"{base}#{i + 1}") for i, piece in enumerate(pieces)]


def _format_comment(comment: dict[str, Any]) -> str:
    return f"{comment.get('author', '')} ({comment.get('date', '')[:10]}): {comment.get('body', '').strip()}"


def _pack_thread(
    header: str, bodies: list[str], max_tokens: int, overlap_tokens: int
) -> list[str]:
    """Pack comment bodies into chunks of <= ``max_tokens``, repeating ``header``.

    Chunks break at comment boundaries; only an individually oversized comment
    is word-split (with overlap).
    """
    header_tokens = approx_tokens(header)
    entries: list[str] = []
    for body in bodies:
        entries.extend(_split_words(body, max_tokens - header_tokens, overlap_tokens))
    packed: list[str] = []
    current: list[str] = []
    current_tokens = header_tokens
    for entry in entries:
        tokens = approx_tokens(entry)
        if current and current_tokens + tokens > max_tokens:
            packed.append("\n\n".join([header, *current]))
            current, current_tokens = [], header_tokens
        current.append(entry)
        current_tokens += tokens
    if current:
        packed.append("\n\n".join([header, *current]))
    return packed


def _chunk_commit(rec: dict[str, Any], max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    if _is_bot(rec.get("author", "")):
        return []
    message = (rec.get("message") or "").strip()
    if not message:
        return []
    stats = rec.get("stats") or {}
    files = rec.get("files") or []
    lines = [
        f"commit {rec['short_sha']}: {message}",
        "",
        f"Author: {rec.get('author', '')} | Date: {rec.get('date', '')[:10]}",
        f"Changes: +{stats.get('additions', 0)}/-{stats.get('deletions', 0)}"
        f" across {rec.get('files_total', len(files))} files",
    ]
    if files:
        lines.append("Files: " + ", ".join(f["filename"] for f in files[:10]))
    meta = {
        "source_type": "commit",
        "ref_id": rec["short_sha"],
        "url": rec.get("html_url", ""),
        "author": rec.get("author", ""),
        "date": rec.get("date", ""),
        "title": message.splitlines()[0][:200],
    }
    pieces = _split_words("\n".join(lines), max_tokens, overlap_tokens)
    return [Chunk(text=t, section=s, **meta) for t, s in _sectioned(pieces, "body")]


def _chunk_pr(rec: dict[str, Any], max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    if _is_bot(rec.get("author", "")):
        return []
    number = rec["number"]
    title = rec.get("title", "")
    meta = {
        "source_type": "pr",
        "ref_id": str(number),
        "url": rec.get("html_url", ""),
        "author": rec.get("author", ""),
        "date": rec.get("created_at", ""),
        "title": title,
    }
    chunks: list[Chunk] = []

    status = "merged" if rec.get("merged") else "closed unmerged"
    header_lines = [
        f"PR #{number}: {title}",
        f"Author: {rec.get('author', '')} | created {rec.get('created_at', '')[:10]} | {status}",
    ]
    if rec.get("labels"):
        header_lines.append("Labels: " + ", ".join(rec["labels"]))
    if rec.get("linked_issues"):
        header_lines.append("Linked issues: " + ", ".join(f"#{n}" for n in rec["linked_issues"]))
    body = (rec.get("body") or "").strip()
    main_text = "\n".join(header_lines) + ("\n\n" + body if body else "")
    for text, section in _sectioned(_split_words(main_text, max_tokens, overlap_tokens), "body"):
        chunks.append(Chunk(text=text, section=section, **meta))

    discussion = [_format_comment(c) for c in rec.get("issue_comments", []) if not _is_bot(c.get("author", ""))]
    if discussion:
        header = f"Discussion on PR #{number}: {title}"
        for text, section in _sectioned(
            _pack_thread(header, discussion, max_tokens, overlap_tokens), "discussion"
        ):
            chunks.append(Chunk(text=text, section=section, **meta))

    by_path: dict[str, list[str]] = {}
    for c in rec.get("review_comments", []):
        if _is_bot(c.get("author", "")):
            continue
        by_path.setdefault(c.get("path") or "general", []).append(_format_comment(c))
    for path, bodies in by_path.items():
        header = f"Code review of {path} in PR #{number}: {title}"
        for text, section in _sectioned(
            _pack_thread(header, bodies, max_tokens, overlap_tokens), f"review:{path}"
        ):
            chunks.append(Chunk(text=text, section=section, **meta))
    return chunks


def _chunk_issue(rec: dict[str, Any], max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    if _is_bot(rec.get("author", "")):
        return []
    number = rec["number"]
    title = rec.get("title", "")
    meta = {
        "source_type": "issue",
        "ref_id": str(number),
        "url": rec.get("html_url", ""),
        "author": rec.get("author", ""),
        "date": rec.get("created_at", ""),
        "title": title,
    }
    chunks: list[Chunk] = []
    body = (rec.get("body") or "").strip()
    main_text = (
        f"Issue #{number}: {title}\n"
        f"Author: {rec.get('author', '')} | created {rec.get('created_at', '')[:10]}"
        + ("\n\n" + body if body else "")
    )
    for text, section in _sectioned(_split_words(main_text, max_tokens, overlap_tokens), "body"):
        chunks.append(Chunk(text=text, section=section, **meta))

    thread = [_format_comment(c) for c in rec.get("comments", []) if not _is_bot(c.get("author", ""))]
    if thread:
        header = f"Discussion on Issue #{number}: {title}"
        for text, section in _sectioned(
            _pack_thread(header, thread, max_tokens, overlap_tokens), "discussion"
        ):
            chunks.append(Chunk(text=text, section=section, **meta))
    return chunks


def chunk_raw_records(
    raw_dir: Path, max_tokens: int = 800, overlap_tokens: int = 100
) -> list[Chunk]:
    """Convert every cached JSON record under ``raw_dir`` into chunks."""
    chunkers = {"commits": _chunk_commit, "prs": _chunk_pr, "issues": _chunk_issue}
    chunks: list[Chunk] = []
    for kind, chunk_fn in chunkers.items():
        records = filtered = 0
        produced = len(chunks)
        for path in sorted((raw_dir / kind).glob("*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            records += 1
            unit_chunks = chunk_fn(rec, max_tokens, overlap_tokens)
            if not unit_chunks:
                filtered += 1
            chunks.extend(unit_chunks)
        logger.info(
            "%s: %d records -> %d chunks (%d records filtered as bot/empty)",
            kind, records, len(chunks) - produced, filtered,
        )
    logger.info("chunking done: %d chunks total", len(chunks))
    return chunks
