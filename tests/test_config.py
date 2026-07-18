"""Phase 0 smoke tests: Settings loads without secrets and honors env overrides."""

import pytest

from patchcontext.config import REPO_ROOT, Settings, settings


def test_singleton_importable_with_defaults() -> None:
    assert settings.github_repo == "fastapi/fastapi"
    assert settings.retrieval_top_k == 20
    assert settings.mmr_lambda == 0.6
    assert settings.rerank_top_k == 5
    assert settings.llm_temperature == 0.2
    assert settings.llm_max_tokens == 3000  # covers reasoning headroom (see config.py)
    assert settings.nli_threshold == 0.5


def test_model_execution_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults (not the local .env, which may opt into MPS) are low-RAM CPU."""
    for var in ("MODEL_DEVICE", "EMBED_BATCH_SIZE", "EMBED_MAX_SEQ_TOKENS"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.model_device == "cpu"  # matches HF Spaces CPU tier
    assert s.embed_batch_size == 8
    assert s.embed_max_seq_tokens == 1024


def test_no_secrets_required(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("GITHUB_TOKEN", "LLM_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.github_token is None
    assert s.llm_api_key is None
    assert s.llm_fallback_api_key is None


def test_provider_agnostic_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard constraint: works unchanged with e.g. OpenAI via env vars."""
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    s = Settings(_env_file=None)
    assert s.llm_base_url == "https://api.openai.com/v1"
    assert s.llm_api_key == "sk-test"
    assert s.llm_model == "gpt-4o-mini"


def test_named_key_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    """NVIDIA_API_KEY / OPENROUTER_API_KEY work as aliases per the brief."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    s = Settings(_env_file=None)
    assert s.llm_api_key == "nvapi-test"
    assert s.llm_fallback_api_key == "sk-or-test"


def test_paths_anchor_to_repo_root() -> None:
    assert settings.data_dir == REPO_ROOT / "data"
    assert settings.raw_dir == REPO_ROOT / "data" / "raw"
    assert settings.index_dir == REPO_ROOT / "data" / "index"
