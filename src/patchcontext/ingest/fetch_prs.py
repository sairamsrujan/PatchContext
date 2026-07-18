"""Fetch closed pull requests (body + review comments + linked issues) into
``data/raw/prs/``.

One JSON file per PR, keyed by number. A rerun skips every PR whose file
already exists. Comment threads (both review comments and issue-style
comments) are fetched whenever they exist — review discussions are the core
value of PR history; the ``min_comments`` threshold only sets the record's
``priority`` flag used for chunk prioritization at index time.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from patchcontext.ingest.github_client import GitHubClient, write_json_atomic

logger = logging.getLogger(__name__)

_ISSUE_REF = re.compile(r"(?:#|/issues/)(\d+)")


def fetch_prs(client: GitHubClient, since_iso: str, out_dir: Path, min_comments: int = 2) -> int:
    """Fetch closed PRs created since ``since_iso``; returns the newly fetched count.

    Lists ``state=closed`` sorted by creation date (newest first) and stops at
    the window edge, so old history is never paginated needlessly.
    """
    prs_dir = out_dir / "prs"
    prs_dir.mkdir(parents=True, exist_ok=True)
    fetched = skipped = 0
    for item in client.paginate(
        f"/repos/{client.repo}/pulls",
        {"state": "closed", "sort": "created", "direction": "desc"},
        checkpoint_key="prs",
    ):
        if item["created_at"] < since_iso:  # sorted desc: everything after is older
            break
        number = item["number"]
        target = prs_dir / f"{number}.json"
        if target.exists():
            skipped += 1
            continue
        detail = client.get(f"/repos/{client.repo}/pulls/{number}")
        review_comments: list[dict[str, Any]] = []
        if detail.get("review_comments", 0) > 0:
            review_comments = list(
                client.paginate(f"/repos/{client.repo}/pulls/{number}/comments")
            )
        issue_comments: list[dict[str, Any]] = []
        if detail.get("comments", 0) > 0:
            issue_comments = list(
                client.paginate(f"/repos/{client.repo}/issues/{number}/comments")
            )
        write_json_atomic(target, _trim_pr(detail, review_comments, issue_comments, min_comments))
        fetched += 1
        if fetched % 50 == 0:
            logger.info("prs: %d fetched (%d cached) so far", fetched, skipped)
    client.clear_checkpoint("prs")  # loop may exit via break at the window edge
    logger.info("prs: done — %d fetched, %d skipped (already cached)", fetched, skipped)
    return fetched


def _linked_issue_numbers(text: str) -> list[int]:
    """Issue/PR numbers referenced in ``text`` via ``#123`` or ``.../issues/123``."""
    return sorted({int(n) for n in _ISSUE_REF.findall(text)})


def _trim_pr(
    detail: dict[str, Any],
    review_comments: list[dict[str, Any]],
    issue_comments: list[dict[str, Any]],
    min_comments: int,
) -> dict[str, Any]:
    """Reduce a PR-detail response (+ comment threads) to the fields chunking needs."""
    body = detail.get("body") or ""
    n_comments = detail.get("comments", 0) + detail.get("review_comments", 0)
    return {
        "source_type": "pr",
        "number": detail["number"],
        "title": detail.get("title", ""),
        "body": body,
        "author": (detail.get("user") or {}).get("login", ""),
        "created_at": detail.get("created_at", ""),
        "closed_at": detail.get("closed_at"),
        "merged_at": detail.get("merged_at"),
        "merged": bool(detail.get("merged")),
        "html_url": detail.get("html_url", ""),
        "labels": [label["name"] for label in detail.get("labels") or []],
        "comments_count": detail.get("comments", 0),
        "review_comments_count": detail.get("review_comments", 0),
        "priority": n_comments >= min_comments,
        "linked_issues": _linked_issue_numbers(body),
        "issue_comments": [
            {
                "author": (c.get("user") or {}).get("login", ""),
                "date": c.get("created_at", ""),
                "body": c.get("body") or "",
            }
            for c in issue_comments
        ],
        "review_comments": [
            {
                "author": (c.get("user") or {}).get("login", ""),
                "date": c.get("created_at", ""),
                "path": c.get("path"),
                "body": c.get("body") or "",
                "in_reply_to": c.get("in_reply_to_id"),
            }
            for c in review_comments
        ],
    }
