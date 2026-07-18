"""Fetch commit history (message + stats) into ``data/raw/commits/``.

One JSON file per commit, keyed by SHA. A rerun skips every commit whose file
already exists, so only the (cheap) list pages are re-read.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from patchcontext.ingest.github_client import GitHubClient, write_json_atomic

logger = logging.getLogger(__name__)

MAX_FILES_PER_COMMIT = 30  # keep records compact; full file lists add no retrieval value


def fetch_commits(client: GitHubClient, since_iso: str, out_dir: Path) -> int:
    """Fetch all commits since ``since_iso`` (ISO 8601) and cache them as JSON.

    The list endpoint has no stats, so each uncached commit costs one extra
    detail request. Returns the number of newly fetched commits.
    """
    commits_dir = out_dir / "commits"
    commits_dir.mkdir(parents=True, exist_ok=True)
    fetched = skipped = 0
    for item in client.paginate(
        f"/repos/{client.repo}/commits", {"since": since_iso}, checkpoint_key="commits"
    ):
        sha = item["sha"]
        target = commits_dir / f"{sha}.json"
        if target.exists():
            skipped += 1
            continue
        detail = client.get(f"/repos/{client.repo}/commits/{sha}")
        write_json_atomic(target, _trim_commit(detail))
        fetched += 1
        if fetched % 100 == 0:
            logger.info("commits: %d fetched (%d cached) so far", fetched, skipped)
    logger.info("commits: done — %d fetched, %d skipped (already cached)", fetched, skipped)
    return fetched


def _trim_commit(detail: dict[str, Any]) -> dict[str, Any]:
    """Reduce a commit-detail response to the fields chunking needs."""
    commit = detail["commit"]
    git_author = commit.get("author") or {}
    github_user = detail.get("author") or {}
    files = detail.get("files") or []
    return {
        "source_type": "commit",
        "sha": detail["sha"],
        "short_sha": detail["sha"][:7],
        "message": commit.get("message", ""),
        "author": github_user.get("login") or git_author.get("name") or "",
        "date": git_author.get("date") or (commit.get("committer") or {}).get("date") or "",
        "html_url": detail.get("html_url", ""),
        "stats": detail.get("stats") or {},
        "files": [
            {"filename": f["filename"], "additions": f["additions"], "deletions": f["deletions"]}
            for f in files[:MAX_FILES_PER_COMMIT]
        ],
        "files_total": len(files),
    }
