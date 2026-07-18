"""Fetch closed issues (body + comment threads) into ``data/raw/issues/``.

One JSON file per issue, keyed by number. A rerun skips every issue whose
file already exists. Comment threads are fetched only for issues with
>= ``min_comments`` comments (those are prioritized); quieter issues
still get a body-only record.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from patchcontext.ingest.github_client import GitHubClient, write_json_atomic

logger = logging.getLogger(__name__)


def fetch_issues(client: GitHubClient, since_iso: str, out_dir: Path, min_comments: int = 2) -> int:
    """Fetch closed issues created since ``since_iso``; returns the newly fetched count.

    Uses the issues list endpoint (which interleaves PRs — those are skipped;
    PRs are handled by :mod:`patchcontext.ingest.fetch_prs`).
    """
    issues_dir = out_dir / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    fetched = skipped = 0
    for item in client.paginate(
        f"/repos/{client.repo}/issues",
        {"state": "closed", "since": since_iso, "sort": "created", "direction": "desc"},
        checkpoint_key="issues",
    ):
        if "pull_request" in item:
            continue
        if item["created_at"] < since_iso:  # sorted desc: everything after is older
            break
        number = item["number"]
        target = issues_dir / f"{number}.json"
        if target.exists():
            skipped += 1
            continue
        comments: list[dict[str, Any]] = []
        if item.get("comments", 0) >= min_comments:
            comments = list(client.paginate(f"/repos/{client.repo}/issues/{number}/comments"))
        write_json_atomic(target, _trim_issue(item, comments, min_comments))
        fetched += 1
        if fetched % 100 == 0:
            logger.info("issues: %d fetched (%d cached) so far", fetched, skipped)
    client.clear_checkpoint("issues")  # loop may exit via break at the window edge
    logger.info("issues: done — %d fetched, %d skipped (already cached)", fetched, skipped)
    return fetched


def _trim_issue(
    item: dict[str, Any], comments: list[dict[str, Any]], min_comments: int
) -> dict[str, Any]:
    """Reduce an issue list entry (+ comment thread) to the fields chunking needs."""
    return {
        "source_type": "issue",
        "number": item["number"],
        "title": item.get("title", ""),
        "body": item.get("body") or "",
        "author": (item.get("user") or {}).get("login", ""),
        "created_at": item.get("created_at", ""),
        "closed_at": item.get("closed_at"),
        "html_url": item.get("html_url", ""),
        "labels": [label["name"] for label in item.get("labels") or []],
        "comments_count": item.get("comments", 0),
        "priority": item.get("comments", 0) >= min_comments,
        "comments": [
            {
                "author": (c.get("user") or {}).get("login", ""),
                "date": c.get("created_at", ""),
                "body": c.get("body") or "",
            }
            for c in comments
        ],
    }
