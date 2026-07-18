"""Phase 0 smoke tests: every module must be importable."""

import importlib

import pytest

MODULES = [
    "patchcontext",
    "patchcontext.config",
    "patchcontext.ingest.github_client",
    "patchcontext.ingest.fetch_commits",
    "patchcontext.ingest.fetch_prs",
    "patchcontext.ingest.fetch_issues",
    "patchcontext.index.chunker",
    "patchcontext.index.embedder",
    "patchcontext.index.build_index",
    "patchcontext.retrieve.retriever",
    "patchcontext.retrieve.reranker",
    "patchcontext.generate.llm_client",
    "patchcontext.generate.prompts",
    "patchcontext.guard.nli_guard",
    "patchcontext.guard.ref_validator",
    "patchcontext.eval.run_ragas",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)
