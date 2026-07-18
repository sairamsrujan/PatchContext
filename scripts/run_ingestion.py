"""One-command pipeline: fetch -> chunk -> embed -> index.

Usage:
    python scripts/run_ingestion.py [--since YYYY-MM-DD] [--skip-fetch]
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone

from patchcontext.config import settings
from patchcontext.index.build_index import build_index
from patchcontext.index.chunker import chunk_raw_records
from patchcontext.ingest.fetch_commits import fetch_commits
from patchcontext.ingest.fetch_issues import fetch_issues
from patchcontext.ingest.fetch_prs import fetch_prs
from patchcontext.ingest.github_client import GitHubClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        help="Ingestion window start, YYYY-MM-DD (default: settings.ingest_window_years ago)",
    )
    parser.add_argument(
        "--skip-fetch", action="store_true", help="Reuse cached raw data; only chunk/embed/index"
    )
    return parser.parse_args()


def window_start_iso(since: str | None) -> str:
    if since:
        return f"{since}T00:00:00Z"
    start = datetime.now(timezone.utc) - timedelta(days=365 * settings.ingest_window_years)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    since_iso = window_start_iso(args.since)

    if args.skip_fetch:
        print("--skip-fetch: reusing cached raw data")
    else:
        print(f"fetching {settings.github_repo} history since {since_iso}")
        with GitHubClient(
            settings.github_repo, settings.github_token, settings.raw_dir / "checkpoints"
        ) as client:
            n_commits = fetch_commits(client, since_iso, settings.raw_dir)
            n_prs = fetch_prs(client, since_iso, settings.raw_dir, settings.min_issue_comments)
            n_issues = fetch_issues(client, since_iso, settings.raw_dir, settings.min_issue_comments)
        print(
            f"fetch complete: {n_commits} commits, {n_prs} PRs, {n_issues} issues newly fetched "
            f"(cached records were skipped; see log lines above for skip counts)"
        )

    chunks = chunk_raw_records(
        settings.raw_dir, settings.chunk_max_tokens, settings.chunk_overlap_tokens
    )
    print(f"chunked: {len(chunks)} chunks")
    build_index(chunks, settings.index_dir)
    print(f"index written to {settings.index_dir}")


if __name__ == "__main__":
    main()
