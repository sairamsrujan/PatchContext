"""Authenticated, rate-limit-aware GitHub REST client with resumable pagination.

- Authenticates with a personal access token (5000 req/hr).
- Respects ``X-RateLimit-Remaining``: when exhausted (or on a secondary rate
  limit), sleeps until ``X-RateLimit-Reset`` / ``Retry-After`` instead of failing.
- Persists pagination checkpoints to disk so an interrupted run resumes from
  the last completed page instead of restarting.

Note for fetchers: if you ``break`` out of :meth:`GitHubClient.paginate` early
(e.g. at the ingestion-window edge), call :meth:`GitHubClient.clear_checkpoint`
afterwards — a leftover checkpoint would make the next full rerun skip the
newest pages.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

API_BASE = "https://api.github.com"
MAX_TRANSIENT_RETRIES = 5


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON via temp file + rename so interrupted runs never leave corrupt cache."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class GitHubClient:
    """Thin wrapper over the GitHub REST API scoped to one repository."""

    def __init__(self, repo: str, token: str | None, checkpoint_dir: Path) -> None:
        if not token:
            raise ValueError(
                "GITHUB_TOKEN is required: unauthenticated access is limited to 60 req/hr."
            )
        self.repo = repo
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._resume_at: float = 0.0  # epoch seconds; set when the quota hits zero
        self._http = httpx.Client(
            base_url=API_BASE,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "patchcontext-ingestion",
            },
            timeout=30.0,
            follow_redirects=True,
        )

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # --- requests ----------------------------------------------------------

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET one resource, honoring rate limits and retrying transient failures.

        Transient = HTTP 5xx *or* transport-level errors (connection drops,
        timeouts, server disconnects) — both get exponential backoff.
        """
        transient_failures = 0
        while True:
            self._wait_if_quota_exhausted()
            try:
                response = self._http.get(path, params=params)
            except httpx.TransportError as exc:
                transient_failures += 1
                if transient_failures > MAX_TRANSIENT_RETRIES:
                    raise
                wait = 2**transient_failures
                logger.warning(
                    "Transport error on %s (%s); retry %d/%d in %d s",
                    path, exc, transient_failures, MAX_TRANSIENT_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            if self._is_rate_limited(response):
                self._sleep_until_reset(response)
                continue
            if response.status_code >= 500:
                transient_failures += 1
                if transient_failures > MAX_TRANSIENT_RETRIES:
                    response.raise_for_status()
                wait = 2**transient_failures
                logger.warning(
                    "HTTP %d from %s; retry %d/%d in %d s",
                    response.status_code, path, transient_failures, MAX_TRANSIENT_RETRIES, wait,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            if response.headers.get("X-RateLimit-Remaining") == "0":
                self._resume_at = float(response.headers.get("X-RateLimit-Reset", "0")) + 1
            return response.json()

    def paginate(
        self, path: str, params: dict[str, Any] | None = None, checkpoint_key: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield every item across all pages of a list endpoint, resumably.

        With ``checkpoint_key``, the next page number is persisted after each
        completed page and the checkpoint is removed when pagination finishes.
        """
        params = dict(params or {})
        params.setdefault("per_page", 100)
        page = self._load_checkpoint(checkpoint_key)
        while True:
            items = self.get(path, {**params, "page": page})
            if not items:
                break
            yield from items
            page += 1
            self._save_checkpoint(checkpoint_key, page)
        self.clear_checkpoint(checkpoint_key)

    # --- rate limiting -------------------------------------------------------

    def _wait_if_quota_exhausted(self) -> None:
        wait = self._resume_at - time.time()
        if wait > 0:
            logger.warning("Rate-limit quota exhausted; sleeping %.0f s until reset", wait)
            time.sleep(wait)
        self._resume_at = 0.0

    @staticmethod
    def _is_rate_limited(response: httpx.Response) -> bool:
        if response.status_code == 429:
            return True
        if response.status_code != 403:
            return False
        # Primary limit: remaining == 0. Secondary limit: 403 with Retry-After.
        return (
            response.headers.get("X-RateLimit-Remaining") == "0"
            or "Retry-After" in response.headers
        )

    @staticmethod
    def _sleep_until_reset(response: httpx.Response) -> None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            wait = float(retry_after) + 1
        else:
            reset = float(response.headers.get("X-RateLimit-Reset", "0"))
            wait = max(reset - time.time(), 0) + 1
        logger.warning("Rate limited (HTTP %d); sleeping %.0f s", response.status_code, wait)
        time.sleep(wait)

    # --- checkpoints ----------------------------------------------------------

    def _checkpoint_path(self, key: str) -> Path:
        return self.checkpoint_dir / f"{key}.json"

    def _load_checkpoint(self, key: str | None) -> int:
        if key is None:
            return 1
        path = self._checkpoint_path(key)
        if path.exists():
            page = int(json.loads(path.read_text(encoding="utf-8"))["next_page"])
            logger.info("Resuming '%s' pagination from page %d", key, page)
            return page
        return 1

    def _save_checkpoint(self, key: str | None, next_page: int) -> None:
        if key is None:
            return
        write_json_atomic(self._checkpoint_path(key), {"next_page": next_page})

    def clear_checkpoint(self, key: str | None) -> None:
        if key is None:
            return
        self._checkpoint_path(key).unlink(missing_ok=True)
