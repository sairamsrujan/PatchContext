"""Phase 4 tests: citation regex/validation, NLI plumbing, prompts, failover.

Offline: the NLI model and OpenAI clients are replaced with fakes.
"""

import numpy as np
import pytest

from patchcontext.generate.llm_client import Endpoint, LLMClient
from patchcontext.generate.prompts import SYSTEM_PROMPT, build_answer_prompt, format_ref
from patchcontext.guard import nli_guard
from patchcontext.guard.nli_guard import check_entailment, split_sentences
from patchcontext.guard.ref_validator import (
    extract_citations,
    known_refs_from_metadata,
    validate_refs,
)
from patchcontext.retrieve.retriever import RetrievedChunk

# --- ref_validator ----------------------------------------------------------

KNOWN = known_refs_from_metadata([
    {"source_type": "pr", "ref_id": "1276"},
    {"source_type": "issue", "ref_id": "4433"},
    {"source_type": "commit", "ref_id": "ab12cd3"},
])


def test_extract_citations_all_formats() -> None:
    answer = (
        "Adopted in [PR #1276] and discussed in [Issue #4433]; "
        "landed via [commit ab12cd3ff00] and again [PR #1276]."
    )
    assert extract_citations(answer) == ["pr:1276", "issue:4433", "commit:ab12cd3"]


def test_validate_refs_pass_and_fail() -> None:
    ok = validate_refs("See [PR #1276] and [commit AB12CD3].", KNOWN)
    assert ok.valid and ok.unknown_refs == []
    bad = validate_refs("See [PR #99999] and [Issue #1276].", KNOWN)
    assert not bad.valid
    # nonexistent PR caught; PR number cited as Issue is also caught (type-aware)
    assert bad.unknown_refs == ["pr:99999", "issue:1276"]


def test_no_citations_is_valid() -> None:
    assert validate_refs("The history does not contain this.", KNOWN).valid


# --- nli_guard ---------------------------------------------------------------

class _FakeNLI:
    """Entails only sentences that share a keyword with the chunk text."""

    class _Cfg:
        id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

    class _Inner:
        config = None

    def __init__(self):
        self.model = self._Inner()
        self.model.config = self._Cfg()

    def predict(self, pairs, show_progress_bar=False):
        logits = []
        for premise, hypothesis in pairs:
            entailed = bool(set(premise.lower().split()) & set(hypothesis.lower().split()) - {"the", "a", "in"})
            logits.append([0.0, 6.0, 0.0] if entailed else [0.0, -6.0, 6.0])
        return np.array(logits)


@pytest.fixture()
def fake_nli(monkeypatch):
    monkeypatch.setattr(nli_guard, "_model", _FakeNLI())


CHUNKS = [RetrievedChunk(text="pydantic validation rewritten for speed", score=0.9,
                         metadata={"source_type": "pr", "ref_id": "1276", "url": "u",
                                   "author": "a", "date": "2026-01-01", "title": "t", "section": "body"})]


def test_entailed_citation_passes(fake_nli) -> None:
    result = check_entailment("Pydantic validation was rewritten [PR #1276].", CHUNKS, threshold=0.5)
    assert result.passed and not result.unsupported_claims


def test_unsupported_citation_fails(fake_nli) -> None:
    result = check_entailment(
        "Websockets were removed entirely [PR #1276]. Also, pydantic validation changed [PR #1276].",
        CHUNKS, threshold=0.5,
    )
    assert not result.passed
    assert result.unsupported_claims == ["Websockets were removed entirely [PR #1276]."]


def test_uncited_sentences_are_not_checked(fake_nli) -> None:
    result = check_entailment("Some general chatter without citations.", CHUNKS)
    assert result.passed and result.scores == {}


def test_bare_citation_sentence_is_not_a_claim(fake_nli) -> None:
    """A trailing sentence that is ONLY a citation marker must not be flagged."""
    result = check_entailment(
        "Pydantic validation was rewritten [PR #1276]. [PR #1276].",
        CHUNKS, threshold=0.5,
    )
    assert result.passed
    assert "[PR #1276]." not in result.scores  # bare marker skipped, not judged


def test_split_sentences_handles_citations() -> None:
    text = "First claim [PR #1]. Second claim [commit ab12cd3]! Third?"
    assert len(split_sentences(text)) == 3


def test_strip_citation_markers() -> None:
    from patchcontext.guard.nli_guard import strip_citation_markers

    sentence = "The CLI was added [PR #11522][1] to help users [commit ab12cd3]."
    assert strip_citation_markers(sentence) == "The CLI was added to help users ."


def test_premise_windows_cover_long_chunks() -> None:
    from patchcontext.guard.nli_guard import premise_windows

    short = "only a few words here"
    assert premise_windows(short) == [short]
    long_text = " ".join(f"w{i}" for i in range(600))
    windows = premise_windows(long_text, max_words=250, overlap=50)
    assert len(windows) == 3
    assert windows[0].split()[0] == "w0"
    assert windows[-1].split()[-1] == "w599"  # tail is never lost
    assert windows[1].split()[0] == "w200"  # overlap preserved


# --- prompts -----------------------------------------------------------------

def test_build_answer_prompt_numbers_and_refs() -> None:
    prompt = build_answer_prompt("why?", CHUNKS)
    assert "[1] PR #1276" in prompt
    assert "Question: why?" in prompt
    assert "citation" in SYSTEM_PROMPT.lower() or "cite" in SYSTEM_PROMPT.lower()
    assert format_ref("commit", "ab12cd3") == "commit ab12cd3"


# --- llm_client failover -----------------------------------------------------

class _FakeCompletion:
    def __init__(self, text, finish_reason="stop"):
        message = type("M", (), {"content": text})
        self.choices = [type("C", (), {"message": message, "finish_reason": finish_reason})]


class _FakeChatAPI:
    def __init__(self, behavior):
        self._behavior = behavior

    def create(self, **kwargs):
        return self._behavior()


def _fake_endpoint(name, behavior, model="m"):
    completions = _FakeChatAPI(behavior)
    chat = type("Chat", (), {"completions": completions})
    client = type("Client", (), {"chat": chat})
    return Endpoint(name=name, client=client, model=model)


def test_failover_to_fallback(monkeypatch) -> None:
    import openai

    def primary_fails():
        raise openai.APIConnectionError(request=None)

    calls = {"n": 0}

    def fallback_succeeds():
        calls["n"] += 1
        return _FakeCompletion("answer from fallback")

    monkeypatch.setattr("time.sleep", lambda s: None)  # skip tenacity backoff waits
    client = LLMClient(endpoints=[
        _fake_endpoint("primary", primary_fails),
        _fake_endpoint("fallback", fallback_succeeds),
    ])
    assert client.chat("sys", "user") == "answer from fallback"
    assert client.active == "fallback"
    assert calls["n"] == 1


def test_no_endpoints_raises() -> None:
    with pytest.raises(ValueError, match="No LLM endpoint"):
        LLMClient(endpoints=[])


def test_clean_content_strips_think_blocks() -> None:
    from patchcontext.generate.llm_client import clean_content

    assert clean_content("<think>step 1... step 2...</think>The answer [PR #9816].") == "The answer [PR #9816]."
    # truncated think block (generation hit the token cap mid-reasoning)
    assert clean_content("<think>endless deliberation that never closes") == ""
    assert clean_content("plain answer, no reasoning") == "plain answer, no reasoning"


def test_truncated_output_discarded_and_retried(monkeypatch) -> None:
    caps = []

    def fake_completion(**kwargs):
        caps.append(kwargs["max_tokens"])
        if len(caps) == 1:  # truncated: partial reasoning, no real answer
            return _FakeCompletion("We need to consider the instruction...", finish_reason="length")
        return _FakeCompletion("clean answer")

    client = LLMClient(endpoints=[_fake_endpoint("primary", None)])
    client._endpoints[0].client.chat.completions.create = fake_completion
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert client.chat("sys", "user", max_tokens=1000) == "clean answer"
    assert caps == [1000, 2000]


def test_empty_answer_retries_with_more_tokens(monkeypatch) -> None:
    caps = []

    def fake_completion(**kwargs):
        caps.append(kwargs["max_tokens"])
        text = "" if len(caps) == 1 else "real answer"
        return _FakeCompletion(text)

    client = LLMClient(endpoints=[_fake_endpoint("primary", None)])
    client._endpoints[0].client.chat.completions.create = fake_completion
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert client.chat("sys", "user", max_tokens=1000) == "real answer"
    assert caps == [1000, 2000]  # second attempt doubled the budget
