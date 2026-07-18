"""Eval runner tests: benchmark loading, resumable batched answer collection."""

import json

import pytest

from patchcontext.eval.run_ragas import collect_answers, load_benchmark

BENCHMARK = [
    {"question": f"q{i}", "ground_truth": f"g{i}", "type": "direct"} for i in range(5)
]


def _answer_fn_factory(calls):
    def answer_fn(question):
        calls.append(question)
        return {"answer": f"a-{question}", "contexts": ["ctx"], "guard_verdict": "verified"}
    return answer_fn


def test_load_benchmark_roundtrip(tmp_path) -> None:
    path = tmp_path / "benchmark.jsonl"
    path.write_text("\n".join(json.dumps(b) for b in BENCHMARK))
    assert load_benchmark(path) == BENCHMARK
    path.write_text(json.dumps({"ground_truth": "no question"}))
    with pytest.raises(ValueError, match="missing 'question'"):
        load_benchmark(path)


def test_collect_answers_batches_and_sleeps(tmp_path, monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr("patchcontext.eval.run_ragas.time.sleep", sleeps.append)
    calls = []
    records = collect_answers(
        BENCHMARK, _answer_fn_factory(calls), tmp_path / "partial.jsonl",
        batch_size=2, sleep_seconds=30,
    )
    assert len(records) == 5 and calls == [f"q{i}" for i in range(5)]
    assert records[3]["answer"] == "a-q3" and records[3]["ground_truth"] == "g3"
    assert sleeps == [30, 30]  # between batches (2,2,1) — not after the last


def test_collect_answers_resumes_from_partial(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("patchcontext.eval.run_ragas.time.sleep", lambda s: None)
    partial = tmp_path / "partial.jsonl"
    first_calls = []
    # simulate an interrupted run: only the first 3 questions answered
    collect_answers(BENCHMARK[:3], _answer_fn_factory(first_calls), partial,
                    batch_size=5, sleep_seconds=0)
    assert len(first_calls) == 3
    # full rerun answers only the missing 2
    second_calls = []
    records = collect_answers(BENCHMARK, _answer_fn_factory(second_calls), partial,
                              batch_size=5, sleep_seconds=0)
    assert second_calls == ["q3", "q4"]
    assert [r["question"] for r in records] == [b["question"] for b in BENCHMARK]
