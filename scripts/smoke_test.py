"""End-to-end sanity check: retrieve -> rerank -> generate -> guard.

Usage:
    python scripts/smoke_test.py                     # runs the 5 gate questions
    python scripts/smoke_test.py --question "..."    # runs one custom question

The default set includes one question about a nonexistent PR to demonstrate
the hallucination guard catching invented references.
"""

from __future__ import annotations

import argparse
import logging

from patchcontext.config import settings
from patchcontext.generate.llm_client import LLMClient
from patchcontext.pipeline import answer_question, load_known_refs
from patchcontext.retrieve.retriever import Retriever

GATE_QUESTIONS = [
    "Why was pydantic v2 adopted and what did the migration involve?",
    "Why was the fastapi-cli command line tool introduced?",
    "How did FastAPI replace startup and shutdown events with lifespan?",
    "What was needed to support Python 3.14?",
    # Deliberately unanswerable: this PR number does not exist in the corpus.
    "What performance improvements did PR #99999 introduce?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="Single question (default: the 5 gate questions)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    questions = [args.question] if args.question else GATE_QUESTIONS

    retriever = Retriever(settings.index_dir)
    llm = LLMClient()
    known_refs = load_known_refs(settings.index_dir)

    for question in questions:
        print(f"\n{'=' * 80}\nQ: {question}")
        answer = answer_question(question, retriever, llm, known_refs)
        badge = "✅ verified" if answer.guard.verdict == "verified" else "⚠️ flagged"
        regen = " (after regeneration)" if answer.guard.regenerated else ""
        print(f"[{badge}{regen} | {answer.provider}:{answer.model}]")
        print(answer.text)
        if answer.guard.verdict == "flagged":
            print(f"--- guard failure: {answer.guard.failure_reason}")
        print("--- top chunks: " + ", ".join(
            f"{c.metadata['source_type']}:{c.metadata['ref_id']}({c.score:+.2f})"
            for c in answer.chunks
        ))


if __name__ == "__main__":
    main()
