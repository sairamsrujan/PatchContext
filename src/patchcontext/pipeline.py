"""End-to-end answer pipeline: retrieve -> rerank -> generate -> guard.

Shared by ``scripts/smoke_test.py`` and ``app.py`` so orchestration lives in
exactly one place. Guard behavior:

1. ``ref_validator``: every cited reference must exist in the index metadata.
2. ``nli_guard``: every cited sentence must be entailed by a retrieved chunk.
3. On failure: regenerate once with the failure reason appended; if it fails
   again, return the answer flagged with the unsupported claims. All guard
   decisions are logged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from patchcontext.config import settings
from patchcontext.generate.llm_client import LLMClient
from patchcontext.generate.prompts import REGENERATION_SUFFIX, SYSTEM_PROMPT, build_answer_prompt
from patchcontext.guard.nli_guard import NLIResult, check_entailment
from patchcontext.guard.ref_validator import (
    RefValidationResult,
    known_refs_from_metadata,
    validate_refs,
)
from patchcontext.retrieve.reranker import rerank
from patchcontext.retrieve.retriever import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardReport:
    verdict: str  # "verified" | "flagged"
    regenerated: bool
    ref_result: RefValidationResult
    nli_result: NLIResult
    failure_reason: str = ""


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    chunks: list[RetrievedChunk]
    guard: GuardReport
    provider: str = ""
    model: str = ""


def load_known_refs(index_dir: Path) -> set[str]:
    """Normalized set of every citable reference in the index metadata."""
    import pandas as pd

    frame = pd.read_parquet(index_dir / "metadata.parquet", columns=["source_type", "ref_id"])
    return known_refs_from_metadata(frame.to_dict("records"))


def _failure_reason(ref: RefValidationResult, nli: NLIResult) -> str:
    parts = []
    if not ref.valid:
        parts.append(
            "these cited references do not exist in the fastapi history: "
            + ", ".join(ref.unknown_refs)
        )
    if not nli.passed:
        parts.append(
            "the following claims were unsupported by the context: "
            + " | ".join(nli.unsupported_claims)
        )
    return "; ".join(parts)


def _run_guard(text: str, chunks: list[RetrievedChunk], known_refs: set[str]) -> tuple[RefValidationResult, NLIResult]:
    ref = validate_refs(text, known_refs)
    nli = check_entailment(text, chunks)
    logger.info(
        "guard: refs %s (%d cited, %d unknown) | nli %s",
        "PASS" if ref.valid else "FAIL", len(ref.cited_refs), len(ref.unknown_refs),
        "PASS" if nli.passed else f"FAIL ({len(nli.unsupported_claims)} unsupported)",
    )
    return ref, nli


def answer_question(
    question: str,
    retriever: Retriever,
    llm: LLMClient,
    known_refs: set[str],
) -> Answer:
    """Run the full pipeline on one question, including the regeneration loop."""
    candidates = retriever.search(
        question, top_k=settings.retrieval_top_k, mmr_lambda=settings.mmr_lambda
    )
    chunks = rerank(question, candidates, top_k=settings.rerank_top_k)
    user_prompt = build_answer_prompt(question, chunks)

    text = llm.chat(SYSTEM_PROMPT, user_prompt)
    ref, nli = _run_guard(text, chunks, known_refs)

    if ref.valid and nli.passed:
        guard = GuardReport("verified", regenerated=False, ref_result=ref, nli_result=nli)
        return Answer(question, text, chunks, guard, llm.active or "", llm.active_model or "")

    reason = _failure_reason(ref, nli)
    logger.warning("guard failed; regenerating once (%s)", reason)
    retry_prompt = user_prompt + REGENERATION_SUFFIX.format(failure_reason=reason)
    text = llm.chat(SYSTEM_PROMPT, retry_prompt)
    ref, nli = _run_guard(text, chunks, known_refs)

    if ref.valid and nli.passed:
        guard = GuardReport("verified", regenerated=True, ref_result=ref, nli_result=nli)
    else:
        guard = GuardReport(
            "flagged", regenerated=True, ref_result=ref, nli_result=nli,
            failure_reason=_failure_reason(ref, nli),
        )
        logger.warning("guard failed after regeneration; flagging answer")
    return Answer(question, text, chunks, guard, llm.active or "", llm.active_model or "")
