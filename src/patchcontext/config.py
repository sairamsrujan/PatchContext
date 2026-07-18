"""Central configuration for PatchContext.

Every environment variable used anywhere in the project is defined here,
loaded via pydantic-settings from the process environment and an optional
`.env` file at the repo root. Secrets default to ``None`` so the package
imports (and non-networked code runs) without any credentials.

Provider-agnostic LLM layer (hard constraint): the generation client reads
``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL`` and works unchanged with
any OpenAI-compatible endpoint. ``NVIDIA_API_KEY`` and ``OPENROUTER_API_KEY``
are accepted as aliases for the primary and fallback keys respectively.
"""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor paths to the repo root so behaviour does not depend on the CWD.
# Layout: <repo>/src/patchcontext/config.py -> parents[2] == <repo>.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All PatchContext settings. Field names map to env vars case-insensitively."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- GitHub ingestion ---
    github_token: str | None = None
    github_repo: str = "fastapi/fastapi"
    ingest_window_years: int = 3
    min_issue_comments: int = 2  # closed issues/PRs with >= this many comments are prioritized

    # --- Primary LLM endpoint (OpenAI-compatible; default: NVIDIA) ---
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("LLM_API_KEY", "NVIDIA_API_KEY")
    )
    llm_model: str = "nvidia/nemotron-3-nano-30b-a3b"

    # --- Fallback LLM endpoint (default: OpenRouter free tier) ---
    llm_fallback_base_url: str = "https://openrouter.ai/api/v1"
    llm_fallback_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_FALLBACK_API_KEY", "OPENROUTER_API_KEY"),
    )
    llm_fallback_model: str = "nvidia/nemotron-3-nano-30b-a3b:free"

    # --- Generation parameters ---
    # max_tokens covers reasoning headroom: a smaller cap truncated
    # nemotron-3-nano MID-REASONING (it thinks before answering), dumping raw
    # chain-of-thought into the answer; the cap now covers reasoning headroom
    # while the prompt keeps the visible answer concise.
    llm_temperature: float = 0.2
    llm_max_tokens: int = 3000

    # --- Local, in-process models ---
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model: str = "BAAI/bge-reranker-base"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_threshold: float = 0.5
    # Low-RAM CPU defaults. "mps" is the
    # fast opt-in on Apple Silicon but balloons unified memory on big batches.
    model_device: str = "cpu"  # device for embedder, reranker, and NLI model
    embed_batch_size: int = 8
    embed_max_seq_tokens: int = 1024  # cap activation memory; p99 chunk ~800 tokens

    # --- Chunking ---
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 100

    # --- Retrieval ---
    retrieval_top_k: int = 20
    mmr_lambda: float = 0.6
    mmr_candidates: int = 12  # MMR-selected diverse subset of top_k fed to the reranker
    rerank_top_k: int = 5

    # --- RAGAs evaluation judge (default: Google AI Studio free tier) ---
    ragas_judge_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    ragas_judge_api_key: str | None = None
    ragas_judge_model: str = "gemini-2.0-flash"
    ragas_batch_size: int = 5
    ragas_batch_sleep_seconds: float = 30.0

    # --- Paths ---
    data_dir: Path = REPO_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        """Cached GitHub JSON (gitignored)."""
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        """Embeddings, metadata parquet, and index artifacts."""
        return self.data_dir / "index"


settings = Settings()
