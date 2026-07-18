"""Deterministic citation validation.

Regex-extracts every cited reference (``[PR #N]``, ``[Issue #N]``,
``[commit sha]``) from an answer and verifies each exists in the index
metadata. Any citation that does not exist is a hard failure (guard step 1).

References are normalized to ``pr:N`` / ``issue:N`` / ``commit:<7-char sha>``
so validation is type-aware: citing an existing PR number as an Issue fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CITATION_RE = re.compile(
    r"\[\s*(?:(PR)\s*#\s*(\d+)|(Issue)\s*#\s*(\d+)|(commit)\s+([0-9a-fA-F]{7,40}))\s*\]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RefValidationResult:
    valid: bool
    cited_refs: list[str]  # normalized: "pr:1276", "issue:4433", "commit:ab12cd3"
    unknown_refs: list[str]  # citations not present in the index metadata


def citation_ref(match: "re.Match[str]") -> str:
    """Normalized form (``pr:N`` / ``issue:N`` / ``commit:sha7``) of one CITATION_RE match."""
    if match.group(2):
        return f"pr:{match.group(2)}"
    if match.group(4):
        return f"issue:{match.group(4)}"
    return f"commit:{match.group(6)[:7].lower()}"


def extract_citations(answer: str) -> list[str]:
    """Every citation in ``answer``, normalized, in order, de-duplicated."""
    refs: list[str] = []
    for match in CITATION_RE.finditer(answer):
        ref = citation_ref(match)
        if ref not in refs:
            refs.append(ref)
    return refs


def validate_refs(answer: str, known_refs: set[str]) -> RefValidationResult:
    """Check every citation in ``answer`` against the normalized set from metadata."""
    cited = extract_citations(answer)
    unknown = [ref for ref in cited if ref not in known_refs]
    return RefValidationResult(valid=not unknown, cited_refs=cited, unknown_refs=unknown)


def known_refs_from_metadata(rows: list[dict]) -> set[str]:
    """Build the normalized reference set from metadata rows
    (dicts with ``source_type`` and ``ref_id``)."""
    refs: set[str] = set()
    for row in rows:
        source_type, ref_id = row["source_type"], str(row["ref_id"]).lower()
        if source_type == "commit":
            refs.add(f"commit:{ref_id[:7]}")
        else:
            refs.add(f"{source_type}:{ref_id}")
    return refs
