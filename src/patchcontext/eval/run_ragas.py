"""Batched, resumable RAGAs evaluation runner.

Two stages, both resumable:

1. **Answer collection** — every benchmark question runs through the full
   retrieve → rerank → generate → guard pipeline in batches of
   ``settings.ragas_batch_size`` with ``settings.ragas_batch_sleep_seconds``
   between batches (free-tier LLM rate limits). Each answer is appended to a
   partial JSONL immediately, so an interrupted run resumes where it stopped.
2. **RAGAs scoring** — the collected answers are scored by an independent
   judge LLM configured via ``RAGAS_JUDGE_*`` env vars (default: Google AI
   Studio free Gemini tier). Results are written to ``eval/results/``.

Usage:
    python -m patchcontext.eval.run_ragas [--benchmark PATH] [--answers-only]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from patchcontext.config import settings

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
BENCHMARK_PATH = EVAL_DIR / "benchmark.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
PARTIAL_ANSWERS = RESULTS_DIR / "answers.partial.jsonl"


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    """Benchmark JSONL: {"question", "ground_truth", "type": direct|multihop|unanswerable}."""
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for item in items:
        if "question" not in item:
            raise ValueError(f"benchmark item missing 'question': {item}")
    return items


def collect_answers(
    benchmark: list[dict[str, Any]],
    answer_fn: Callable[[str], dict[str, Any]],
    partial_path: Path,
    batch_size: int,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    """Run ``answer_fn`` over every question, resumably.

    Answers already present in ``partial_path`` (matched by question text) are
    reused; new ones are appended to the file as they complete.
    """
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done[record["question"]] = record
        if done:
            logger.info("resuming: %d answers already collected", len(done))

    pending = [item for item in benchmark if item["question"] not in done]
    with open(partial_path, "a", encoding="utf-8") as sink:
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            for item in batch:
                record = {**item, **answer_fn(item["question"])}
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                done[item["question"]] = record
            logger.info("collected %d/%d answers", len(done), len(benchmark))
            if batch_start + batch_size < len(pending) and sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return [done[item["question"]] for item in benchmark]


def make_pipeline_answer_fn() -> Callable[[str], dict[str, Any]]:
    """Build the real answer function (loads the retriever, LLM, and guard once)."""
    from patchcontext.generate.llm_client import LLMClient
    from patchcontext.pipeline import answer_question, load_known_refs
    from patchcontext.retrieve.retriever import Retriever

    retriever = Retriever(settings.index_dir)
    llm = LLMClient()
    known_refs = load_known_refs(settings.index_dir)

    def answer_fn(question: str) -> dict[str, Any]:
        answer = answer_question(question, retriever, llm, known_refs)
        return {
            "answer": answer.text,
            "contexts": [c.text for c in answer.chunks],
            "guard_verdict": answer.guard.verdict,
            "provider": answer.provider,
        }

    return answer_fn


def _shim_langchain_vertexai() -> None:
    """ragas 0.4.3 unconditionally imports langchain_community.chat_models.vertexai,
    which langchain-community 0.4 removed. Register a placeholder module (we
    never use a VertexAI judge) so ragas imports cleanly."""
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ModuleNotFoundError:
        stub = types.ModuleType(name)

        class ChatVertexAI:  # pragma: no cover — placeholder, never instantiated
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("VertexAI judge is not supported in PatchContext")

        stub.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
        sys.modules[name] = stub


def evaluate_with_ragas(records: list[dict[str, Any]]) -> dict[str, float]:
    """Score collected answers with RAGAs using the configured judge LLM.

    Judge: any OpenAI-compatible endpoint via ``RAGAS_JUDGE_*`` (default:
    Google AI Studio free tier). Embeddings for answer_relevancy: our local
    Qwen3 embedder. Worker count kept low for free-tier rate limits.
    """
    if not settings.ragas_judge_api_key:
        raise SystemExit("RAGAS_JUDGE_API_KEY is not set — required for scoring.")

    _shim_langchain_vertexai()
    import math

    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig

    from patchcontext.retrieve.retriever import _QwenEmbeddingsAdapter

    judge = LangchainLLMWrapper(ChatOpenAI(
        base_url=settings.ragas_judge_base_url,
        api_key=settings.ragas_judge_api_key,
        model=settings.ragas_judge_model,
        temperature=0,
    ))
    embeddings = LangchainEmbeddingsWrapper(_QwenEmbeddingsAdapter())
    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference": r["ground_truth"],
        }
        for r in records
    ])
    metrics = [
        Faithfulness(),
        # strictness=1: single completion per question. The default (3) asks the
        # judge for n=3 candidates in one call, which Gemini's OpenAI-compatible
        # endpoint rejects ("Multiple candidates is not enabled for this model").
        ResponseRelevancy(strictness=1),
        ContextPrecision(),
        ContextRecall(),
    ]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=2, max_retries=10, max_wait=60, timeout=180),
    )
    frame = result.to_pandas()
    names = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    scores: dict[str, float] = {}
    for col in (c for c in frame.columns if c in names):
        mean = float(frame[col].mean(skipna=True))
        if not math.isnan(mean):  # drop metrics that failed on every row
            scores[col] = mean
        else:
            logger.warning("metric %s produced no values (all NaN); omitting", col)
    return scores


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument(
        "--answers-only", action="store_true",
        help="Collect pipeline answers only; skip RAGAs judge scoring",
    )
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)
    records = collect_answers(
        benchmark,
        make_pipeline_answer_fn(),
        PARTIAL_ANSWERS,
        settings.ragas_batch_size,
        settings.ragas_batch_sleep_seconds,
    )
    print(f"answers collected: {len(records)} (partials in {PARTIAL_ANSWERS})")
    if args.answers_only:
        return

    metrics = evaluate_with_ragas(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"ragas_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    out.write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge_model": settings.ragas_judge_model,
        "n_questions": len(records),
        "metrics": metrics,
        "per_question": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results written to {out}")


if __name__ == "__main__":
    main()
