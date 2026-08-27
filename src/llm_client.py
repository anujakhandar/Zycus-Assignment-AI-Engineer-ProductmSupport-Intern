"""Thin wrapper around the Anthropic Messages API.

Everything in this project that talks to a model goes through :class:`LLMClient`
so the model id, sampling settings, caching and error handling live in one
place.

Two behaviours are worth calling out because the brief asks for them directly:

* **Determinism.** ``temperature`` defaults to 0.0 and every request is keyed
  into :class:`~src.cache.ResponseCache`, so an identical input replays an
  identical response rather than re-sampling.
* **Offline execution.** With a populated cache the pipelines need no network
  and no API key, which is how CI and the eval harness run.

Configuration comes from ``.env`` (see ``.env.example``):

    ANTHROPIC_API_KEY     required for live calls; not needed on a cache hit
    ANTHROPIC_MODEL       default: claude-sonnet-4-6
    LLM_TEMPERATURE       default: 0.0
    LLM_MAX_TOKENS        default: 4096
    LLM_TIMEOUT_SECONDS   default: 60
    LLM_CACHE             on (default) | off - bypass cache reads
    LLM_OFFLINE           0 (default) | 1 - never call the API, cache only
    LLM_MAX_CALLS         default: 60 - hard cap on live calls per process

Usage::

    from src.llm_client import LLMClient

    llm = LLMClient()
    text, meta = llm.complete("Classify this ticket...", system="You are ...")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from dotenv import load_dotenv

from src.cache import ResponseCache, cache_key

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
DEFAULT_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
DEFAULT_MAX_RETRIES = 3

# Cache hits are free and uncounted; this caps only calls that reach the API.
# Sized for a demo run plus a full eval suite (~44 calls). Triaging the whole
# ticket file would be ~500 calls, so an accidental loop over the dataset stops
# here instead of quietly spending the budget.
DEFAULT_MAX_CALLS = int(os.getenv("LLM_MAX_CALLS", "60"))


class LLMError(RuntimeError):
    """Raised for any failure while talking to the API."""


class LLMConfigError(LLMError):
    """Raised when required configuration (an API key) is missing."""


class LLMOfflineError(LLMError):
    """Raised when a live call is needed but the client is running offline."""


class LLMBudgetError(LLMError):
    """Raised when a run would exceed its live-call budget."""


@dataclass
class CallMeta:
    """What happened on a call, for logging and for the eval report."""

    model: str = ""
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: Optional[str] = None
    cache_key: str = ""
    tags: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """A small, synchronous client for single-turn and multi-turn requests."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        *,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        cache: Optional[ResponseCache] = None,
        offline: Optional[bool] = None,
        max_calls: int = DEFAULT_MAX_CALLS,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = cache if cache is not None else ResponseCache()
        self.offline = (
            offline if offline is not None else os.getenv("LLM_OFFLINE", "0").lower() in {"1", "true", "yes"}
        )

        self.max_calls = max_calls
        self.live_calls = 0

        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None  # constructed lazily, so a cache-only run needs no key
        self.last_meta: Optional[CallMeta] = None

    # -- lazy client -------------------------------------------------------

    def _anthropic(self):
        """Build the SDK client on first live call."""
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise LLMConfigError(
                "ANTHROPIC_API_KEY is not set and this request is not in the cache. "
                "Copy .env.example to .env and add a key, or run against a populated cache."
            )

        import anthropic  # imported here so offline runs do not need the package loaded

        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        return self._client

    # -- requests ----------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tags: Optional[dict[str, Any]] = None,
    ) -> tuple[str, CallMeta]:
        """Send one user message. Returns ``(text, meta)``."""
        return self.send(
            [{"role": "user", "content": prompt}],
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            tags=tags,
        )

    def send(
        self,
        messages: Sequence[dict],
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[Sequence[str]] = None,
        tags: Optional[dict[str, Any]] = None,
    ) -> tuple[str, CallMeta]:
        """Send a full message list. Returns ``(text, meta)``.

        On a cache hit no network call is made and ``meta.cached`` is True.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": list(messages),
        }

        effective_temperature = self.temperature if temperature is None else temperature
        if effective_temperature is not None:
            params["temperature"] = effective_temperature
        if system:
            params["system"] = system
        if stop_sequences:
            params["stop_sequences"] = list(stop_sequences)

        key = cache_key({**params, "_tags": tags or {}})
        record = self.cache.get(key)
        if record is not None:
            meta = CallMeta(
                model=record.get("model", self.model),
                cached=True,
                input_tokens=record.get("input_tokens", 0),
                output_tokens=record.get("output_tokens", 0),
                stop_reason=record.get("stop_reason"),
                cache_key=key,
                tags=tags or {},
            )
            self.last_meta = meta
            logger.debug("cache hit %s", key[:12])
            return record.get("text", ""), meta

        if self.offline:
            raise LLMOfflineError(
                f"LLM_OFFLINE is set and request {key[:12]} is not cached. "
                "Run once with a key to populate fixtures/llm_cache/."
            )

        text, meta = self._live_call(params, key, tags or {})
        self.cache.put(
            key,
            {
                "text": text,
                "model": meta.model,
                "input_tokens": meta.input_tokens,
                "output_tokens": meta.output_tokens,
                "stop_reason": meta.stop_reason,
                "request": _redact_request(params),
                "tags": tags or {},
            },
        )
        return text, meta

    def _live_call(self, params: dict[str, Any], key: str, tags: dict[str, Any]) -> tuple[str, CallMeta]:
        import anthropic

        if self.live_calls >= self.max_calls:
            raise LLMBudgetError(
                f"Live-call budget exhausted ({self.max_calls} calls). "
                "Raise LLM_MAX_CALLS deliberately if this run really needs more - "
                "the cap exists so a loop over the dataset cannot drain an API balance."
            )

        client = self._anthropic()
        self.live_calls += 1
        try:
            message = client.messages.create(**params)
        except anthropic.AuthenticationError as exc:
            raise LLMConfigError("Anthropic rejected the API key.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Unknown model or endpoint: {self.model}") from exc
        except anthropic.RateLimitError as exc:
            retry_after = exc.response.headers.get("retry-after", "unknown")
            raise LLMError(f"Rate limited by the API (retry-after: {retry_after}s).") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError(f"Request rejected: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the API - check the network connection.") from exc

        meta = CallMeta(
            model=message.model,
            cached=False,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            stop_reason=message.stop_reason,
            cache_key=key,
            tags=tags,
        )
        self.last_meta = meta

        if message.stop_reason == "max_tokens":
            logger.warning(
                "Response hit the max_tokens limit (%s) and may be truncated.",
                params["max_tokens"],
            )
        return extract_text(message), meta

    def count_tokens(self, prompt: str, *, system: Optional[str] = None) -> int:
        """Input token count for a prompt, for budgeting retrieved KB context."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            params["system"] = system
        return self._anthropic().messages.count_tokens(**params).input_tokens


def extract_text(message) -> str:
    """Concatenate the text blocks of a response, ignoring any other block type."""
    return "\n".join(block.text for block in message.content if block.type == "text").strip()


def _redact_request(params: dict[str, Any]) -> dict[str, Any]:
    """Store request shape in the cache without duplicating full prompt bodies."""
    return {
        "model": params.get("model"),
        "temperature": params.get("temperature"),
        "max_tokens": params.get("max_tokens"),
        "system_chars": len(params.get("system") or ""),
        "message_chars": sum(len(str(m.get("content", ""))) for m in params.get("messages", [])),
    }
