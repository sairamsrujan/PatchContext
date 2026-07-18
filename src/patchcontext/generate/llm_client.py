"""OpenAI-compatible LLM client with automatic provider failover.

- Primary endpoint: ``settings.llm_base_url`` / ``llm_api_key`` / ``llm_model``
  (defaults: NVIDIA nemotron). Fallback: ``settings.llm_fallback_*`` (defaults:
  OpenRouter free tier). Endpoints without a key are skipped.
- Each endpoint gets tenacity retries with exponential backoff on 429/5xx and
  connection/timeout errors; when the primary's retries are exhausted, the
  client fails over to the next endpoint.
- Provider-agnostic: works unchanged with any OpenAI-compatible endpoint
  (e.g. OpenAI gpt-4o-mini) supplied via env vars.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from patchcontext.config import settings

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_ATTEMPTS_PER_ENDPOINT = 4
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def clean_content(content: str) -> str:
    """Strip reasoning-model think blocks (closed or truncated) from an answer."""
    return _THINK_BLOCK.sub("", content).strip()


@dataclass
class Endpoint:
    name: str  # "primary" | "fallback" (or custom, in tests)
    client: "OpenAI"
    model: str


def _default_endpoints() -> list[Endpoint]:
    from openai import OpenAI

    endpoints: list[Endpoint] = []
    if settings.llm_api_key:
        endpoints.append(Endpoint(
            "primary",
            OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key),
            settings.llm_model,
        ))
    if settings.llm_fallback_api_key:
        endpoints.append(Endpoint(
            "fallback",
            OpenAI(base_url=settings.llm_fallback_base_url, api_key=settings.llm_fallback_api_key),
            settings.llm_fallback_model,
        ))
    return endpoints


class LLMClient:
    """Chat-completion client with primary -> fallback failover."""

    def __init__(self, endpoints: list[Endpoint] | None = None) -> None:
        self._endpoints = endpoints if endpoints is not None else _default_endpoints()
        if not self._endpoints:
            raise ValueError(
                "No LLM endpoint configured: set LLM_API_KEY (or NVIDIA_API_KEY) "
                "and/or LLM_FALLBACK_API_KEY (or OPENROUTER_API_KEY) in .env"
            )
        self.active: str | None = None  # last endpoint that answered (for the UI sidebar)
        self.active_model: str | None = None

    def chat(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant text for a single system+user exchange.

        Tries each endpoint in order; raises the last error if all fail.
        """
        last_error: Exception | None = None
        for endpoint in self._endpoints:
            try:
                text = self._chat_with_retries(endpoint, system, user, temperature, max_tokens)
                if not text:
                    # Reasoning models can burn the whole budget thinking;
                    # one retry with double headroom before failing over.
                    logger.warning("endpoint '%s' returned empty answer; retrying with 2x tokens", endpoint.name)
                    doubled = 2 * (settings.llm_max_tokens if max_tokens is None else max_tokens)
                    text = self._chat_with_retries(endpoint, system, user, temperature, doubled)
                if not text:
                    raise RuntimeError("empty answer after token-budget retry")
            except Exception as exc:  # noqa: BLE001 — any endpoint failure triggers failover
                logger.warning("endpoint '%s' failed after retries: %s", endpoint.name, exc)
                last_error = exc
                continue
            self.active, self.active_model = endpoint.name, endpoint.model
            return text
        raise RuntimeError("All LLM endpoints failed") from last_error

    def _chat_with_retries(
        self,
        endpoint: Endpoint,
        system: str,
        user: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        import openai
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        retryable = (
            openai.RateLimitError,        # 429
            openai.InternalServerError,   # 5xx
            openai.APIConnectionError,
            openai.APITimeoutError,
        )

        @retry(
            retry=retry_if_exception_type(retryable),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(MAX_ATTEMPTS_PER_ENDPOINT),
            reraise=True,
            before_sleep=lambda rs: logger.warning(
                "endpoint '%s' attempt %d failed (%s); backing off",
                endpoint.name, rs.attempt_number, rs.outcome.exception(),
            ),
        )
        def _call() -> str:
            response: Any = endpoint.client.chat.completions.create(
                model=endpoint.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=settings.llm_temperature if temperature is None else temperature,
                max_tokens=settings.llm_max_tokens if max_tokens is None else max_tokens,
            )
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                # Truncated mid-generation: for a reasoning model the visible
                # content is partial chain-of-thought, not an answer. Discard
                # so the caller's double-budget retry kicks in.
                logger.warning("endpoint '%s' hit the token cap; discarding truncated output", endpoint.name)
                return ""
            return clean_content(choice.message.content or "")

        return _call()
