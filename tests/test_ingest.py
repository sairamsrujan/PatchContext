"""Ingestion tests: cache-skip, checkpointing, window edge, record trimming. No network."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from patchcontext.ingest.fetch_commits import _trim_commit, fetch_commits
from patchcontext.ingest.fetch_issues import fetch_issues
from patchcontext.ingest.fetch_prs import _linked_issue_numbers, fetch_prs
from patchcontext.ingest.github_client import GitHubClient, write_json_atomic


class FakeClient:
    """Offline stand-in for GitHubClient: canned list items and detail responses."""

    repo = "fastapi/fastapi"

    def __init__(
        self,
        list_items: dict[str, list[dict[str, Any]]],
        details: dict[str, Any] | None = None,
    ) -> None:
        self._list_items = list_items  # path -> items
        self._details = details or {}
        self.detail_calls = 0
        self.cleared: list[str | None] = []

    def paginate(
        self, path: str, params: dict[str, Any] | None = None, checkpoint_key: str | None = None
    ) -> Iterator[dict[str, Any]]:
        yield from self._list_items.get(path, [])

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        self.detail_calls += 1
        return self._details[path]

    def clear_checkpoint(self, key: str | None) -> None:
        self.cleared.append(key)


SHA = "ab12cd3" + "0" * 33

COMMIT_DETAIL = {
    "sha": SHA,
    "html_url": f"https://github.com/fastapi/fastapi/commit/{SHA}",
    "author": {"login": "tiangolo"},
    "commit": {
        "message": "Adopt pydantic v2",
        "author": {"name": "Sebastián Ramírez", "date": "2026-06-01T00:00:00Z"},
    },
    "stats": {"additions": 10, "deletions": 2, "total": 12},
    "files": [{"filename": "fastapi/main.py", "additions": 10, "deletions": 2}],
}


def _commit_client() -> FakeClient:
    return FakeClient(
        {"/repos/fastapi/fastapi/commits": [{"sha": SHA}]},
        {f"/repos/fastapi/fastapi/commits/{SHA}": COMMIT_DETAIL},
    )


def test_fetch_commits_then_rerun_skips_cache(tmp_path: Path) -> None:
    client = _commit_client()
    assert fetch_commits(client, "2026-01-01T00:00:00Z", tmp_path) == 1  # type: ignore[arg-type]
    assert client.detail_calls == 1
    # Rerun: record is cached -> no new fetch, no new detail request.
    assert fetch_commits(client, "2026-01-01T00:00:00Z", tmp_path) == 0  # type: ignore[arg-type]
    assert client.detail_calls == 1


def test_trim_commit_fields() -> None:
    record = _trim_commit(COMMIT_DETAIL)
    assert record["source_type"] == "commit"
    assert record["short_sha"] == "ab12cd3"
    assert record["author"] == "tiangolo"
    assert record["stats"]["total"] == 12
    assert record["files_total"] == 1


def test_fetch_prs_window_edge_and_cache(tmp_path: Path) -> None:
    in_window = {"number": 2, "created_at": "2026-06-01T00:00:00Z"}
    too_old = {"number": 1, "created_at": "2020-01-01T00:00:00Z"}
    detail = {
        "number": 2,
        "title": "Fix dependency cache",
        "body": "Closes #1276 and relates to /issues/4433",
        "user": {"login": "tiangolo"},
        "created_at": "2026-06-01T00:00:00Z",
        "merged": True,
        "comments": 0,
        "review_comments": 0,
        "html_url": "https://github.com/fastapi/fastapi/pull/2",
        "labels": [],
    }
    client = FakeClient(
        {"/repos/fastapi/fastapi/pulls": [in_window, too_old]},
        {"/repos/fastapi/fastapi/pulls/2": detail},
    )
    # Only the in-window PR is fetched; the loop breaks at the window edge.
    assert fetch_prs(client, "2026-01-01T00:00:00Z", tmp_path) == 1  # type: ignore[arg-type]
    assert (tmp_path / "prs" / "2.json").exists()
    assert not (tmp_path / "prs" / "1.json").exists()
    assert "prs" in client.cleared  # checkpoint cleared after early break
    # Rerun skips the cached PR entirely.
    assert fetch_prs(client, "2026-01-01T00:00:00Z", tmp_path) == 0  # type: ignore[arg-type]
    assert client.detail_calls == 1


def test_fetch_issues_skips_prs_and_thread_threshold(tmp_path: Path) -> None:
    pr_entry = {"number": 9, "created_at": "2026-06-02T00:00:00Z", "pull_request": {}}
    quiet = {"number": 10, "created_at": "2026-06-01T00:00:00Z", "comments": 1, "user": {}}
    busy = {"number": 11, "created_at": "2026-06-01T00:00:00Z", "comments": 3, "user": {}}
    client = FakeClient(
        {
            "/repos/fastapi/fastapi/issues": [pr_entry, busy, quiet],
            "/repos/fastapi/fastapi/issues/11/comments": [
                {"user": {"login": "a"}, "created_at": "2026-06-02T00:00:00Z", "body": "same here"}
            ],
        }
    )
    assert fetch_issues(client, "2026-01-01T00:00:00Z", tmp_path, min_comments=2) == 2  # type: ignore[arg-type]
    assert not (tmp_path / "issues" / "9.json").exists()  # PR entry skipped
    import json

    busy_record = json.loads((tmp_path / "issues" / "11.json").read_text())
    quiet_record = json.loads((tmp_path / "issues" / "10.json").read_text())
    assert busy_record["priority"] and len(busy_record["comments"]) == 1
    assert not quiet_record["priority"] and quiet_record["comments"] == []


def test_linked_issue_numbers() -> None:
    text = "Closes #1276, see #1276 again and https://github.com/fastapi/fastapi/issues/4433"
    assert _linked_issue_numbers(text) == [1276, 4433]


def test_get_retries_transport_errors(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    """Connection drops mid-run must be retried, not crash the run."""
    import time

    import httpx

    client = GitHubClient("o/r", "test-token", tmp_path)
    calls = {"n": 0}

    def flaky_get(path: str, params: dict | None = None) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return httpx.Response(
            200, json={"ok": True}, request=httpx.Request("GET", "https://api.github.com" + path)
        )

    monkeypatch.setattr(client._http, "get", flaky_get)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    assert client.get("/rate_limit") == {"ok": True}
    assert calls["n"] == 3
    client.close()


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    client = GitHubClient("o/r", "test-token", tmp_path)
    assert client._load_checkpoint("commits") == 1
    client._save_checkpoint("commits", 7)
    assert client._load_checkpoint("commits") == 7
    client.clear_checkpoint("commits")
    assert client._load_checkpoint("commits") == 1
    client.close()


def test_write_json_atomic_leaves_no_tmp(tmp_path: Path) -> None:
    target = tmp_path / "record.json"
    write_json_atomic(target, {"ok": True})
    assert target.exists()
    assert list(tmp_path.glob("*.tmp")) == []
