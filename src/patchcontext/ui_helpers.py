"""Pure helpers for the Streamlit UI.

Kept streamlit-free so tests import them without a Streamlit runtime, and
``app.py`` stays a thin layout layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from patchcontext.guard.ref_validator import CITATION_RE, citation_ref


def load_ref_urls(index_dir: Path) -> dict[str, str]:
    """Map every normalized reference (``pr:N`` / ``issue:N`` / ``commit:sha7``)
    to its GitHub URL, from the metadata sidecar."""
    import pandas as pd

    frame = pd.read_parquet(index_dir / "metadata.parquet", columns=["source_type", "ref_id", "url"])
    urls: dict[str, str] = {}
    for row in frame.drop_duplicates().itertuples(index=False):
        ref_id = str(row.ref_id).lower()
        key = f"commit:{ref_id[:7]}" if row.source_type == "commit" else f"{row.source_type}:{ref_id}"
        urls.setdefault(key, row.url)
    return urls


def linkify_citations(text: str, ref_urls: dict[str, str]) -> str:
    """Render each ``[PR #N]`` / ``[Issue #N]`` / ``[commit sha]`` citation as a
    markdown link to the real GitHub URL. Unknown citations stay plain text."""

    def replace(match: Any) -> str:
        url = ref_urls.get(citation_ref(match))
        return f"[{match.group(0)}]({url})" if url else match.group(0)

    return CITATION_RE.sub(replace, text)


def flag_unsupported(text: str, unsupported_claims: list[str]) -> str:
    """Visibly mark each guard-flagged claim inside the answer text."""
    for claim in unsupported_claims:
        text = text.replace(claim, f"**⚠️ {claim}**")
    return text


def load_eval_results(results_dir: Path) -> list[dict[str, Any]]:
    """Parsed RAGAs result files (newest first); empty list if none exist yet."""
    results = []
    for path in sorted(results_dir.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and "metrics" in payload:
            results.append({"name": path.stem, **payload})
    return results
