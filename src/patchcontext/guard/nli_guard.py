"""NLI entailment guard with cross-encoder/nli-deberta-v3-base (guard step 2).

Every answer sentence containing a citation must be entailed (entailment
probability > threshold, default 0.5) by at least one retrieved chunk.
The chunk text is the premise; the answer sentence is the hypothesis.
The model is lazy-loaded once per process.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from patchcontext.config import settings
from patchcontext.guard.ref_validator import CITATION_RE
from patchcontext.retrieve.retriever import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_model: "CrossEncoder | None" = None
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'`0-9])")
_FOOTNOTE = re.compile(r"\[\d+\]")


@dataclass(frozen=True)
class NLIResult:
    passed: bool
    unsupported_claims: list[str]  # cited sentences no chunk entails
    scores: dict[str, float]  # cited sentence -> best entailment probability


def get_model() -> "CrossEncoder":
    """Load the NLI cross-encoder once per process (downloads on first use)."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading NLI model %s (device: %s) ...", settings.nli_model, settings.model_device)
        _model = CrossEncoder(settings.nli_model, device=settings.model_device)
        logger.info("NLI model ready")
    return _model


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def strip_citation_markers(sentence: str) -> str:
    """Remove citation/footnote markers before NLI — they are meta-text, not
    claims, and pollute the hypothesis."""
    cleaned = CITATION_RE.sub("", sentence)
    cleaned = _FOOTNOTE.sub("", cleaned)
    return " ".join(cleaned.split())


def premise_windows(text: str, max_words: int = 60, overlap: int = 20) -> list[str]:
    """Split a chunk into overlapping word windows sized for the NLI model.

    nli-deberta-v3-base is trained on short premise/hypothesis pairs and
    truncates at 512 tokens: long premises both hide chunk tails and dilute
    entailment. Calibrated 2026-07-12 on known-true vs invented claims:
    60-word windows separate cleanly (true ≥0.95, invented ≤0.01) where
    250-word windows scored some true claims below invented ones.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max_words - overlap
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), step)]


def _entailment_index(model: "CrossEncoder") -> int:
    """Index of the 'entailment' class in the model's output logits."""
    id2label = getattr(model.model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        if str(label).lower() == "entailment":
            return int(idx)
    return 1  # documented order for cross-encoder/nli-deberta-v3-base


def check_entailment(
    answer: str,
    chunks: list[RetrievedChunk],
    threshold: float | None = None,
) -> NLIResult:
    """Verify every cited sentence in ``answer`` is entailed by some chunk."""
    import numpy as np

    threshold = settings.nli_threshold if threshold is None else threshold
    # A sentence that is ONLY citation markers (e.g. a trailing "[PR #123]")
    # carries no claim: after stripping it would be an empty hypothesis that
    # NLI can never entail, falsely flagging good answers. Skip those.
    cited = [
        (sentence, strip_citation_markers(sentence))
        for sentence in split_sentences(answer)
        if CITATION_RE.search(sentence)
    ]
    cited = [
        (sentence, hypothesis)
        for sentence, hypothesis in cited
        if any(ch.isalnum() for ch in hypothesis)  # punctuation-only leftovers are not claims
    ]
    if not cited or not chunks:
        return NLIResult(passed=True, unsupported_claims=[], scores={})
    cited_sentences = [sentence for sentence, _ in cited]

    model = get_model()
    windows = [w for chunk in chunks for w in premise_windows(chunk.text)]
    hypotheses = [hypothesis for _, hypothesis in cited]
    pairs = [(window, hypothesis) for hypothesis in hypotheses for window in windows]
    logits = np.asarray(model.predict(pairs, show_progress_bar=False))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    entail = probs[:, _entailment_index(model)].reshape(len(cited_sentences), len(windows))

    scores = {
        sentence: float(entail[i].max()) for i, sentence in enumerate(cited_sentences)
    }
    unsupported = [s for s, best in scores.items() if best <= threshold]
    result = NLIResult(passed=not unsupported, unsupported_claims=unsupported, scores=scores)
    logger.info(
        "nli_guard: %d cited sentences, %d unsupported (threshold %.2f)",
        len(cited_sentences), len(unsupported), threshold,
    )
    return result
