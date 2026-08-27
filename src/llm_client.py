"""Thin wrapper around the Gemini API.

Everything in this project that talks to a model goes through :class:`LLMClient`
so the model id, determinism settings, caching, cost control and error handling
live in one place. Nothing else in the codebase imports a provider SDK, which is
what kept swapping providers down to a single file.

Three behaviours matter, because the brief asks for them directly:

* **Determinism.** ``temperature=0`` and a fixed ``seed`` make generation
  repeatable at the provider. The response cache then guarantees it regardless,
  by replaying a stored response for an identical request rather than generating
  again. The brief allows "control temperature, seed, or post-process to ensure
  this"; this does all three.
* **Offline execution.** With a populated cache the pipelines need no network
  and no API key, which is how CI and the evaluation harness run.
* **Cost control.** Live calls are capped per process. Cache hits are free and
  are not counted against the cap.

Configuration comes from ``.env`` (see ``.env.example``):

    GEMINI_API_KEY        required for live calls; not needed on a cache hit
    GEMINI_MODEL          default: gemini-3.5-flash-lite
    LLM_TEMPERATURE       default: 0.0
    LLM_SEED              default: 42
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
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from dotenv import load_dotenv

from src.cache import ResponseCache, cache_key

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
DEFAULT_SEED = int(os.getenv("LLM_SEED", "42"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
DEFAULT_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
DEFAULT_MAX_RETRIES = 3

# Cache hits are free and uncounted; this caps only calls that reach the API.
# Sized for a demo run plus a full eval suite (~44 calls). Triaging the whole
# ticket file would be ~500 calls, so an accidental loop over the dataset stops
# here instead of quietly draining a quota.
DEFAULT_MAX_CALLS = int(os.getenv("LLM_MAX_CALLS", "60"))


class LLMError(RuntimeError):
    """Raised for any failure while talking to the API."""


class LLMConfigError(LLMError):
    """Raised when required configuration (an API key) is missing or rejected."""


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
        seed: Optional[int] = DEFAULT_SEED,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        cache: Optional[ResponseCache] = None,
        offline: Optional[bool] = None,
        max_calls: int = DEFAULT_MAX_CALLS,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = cache if cache is not None else ResponseCache()
        self.offline = (
            offline if offline is not None else os.getenv("LLM_OFFLINE", "0").lower() in {"1", "true", "yes"}
        )

        self.max_calls = max_calls
        self.live_calls = 0

        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._client = None  # constructed lazily, so a cache-only run needs no key
        self.last_meta: Optional[CallMeta] = None

    # -- lazy client -------------------------------------------------------

    def _gemini(self):
        """Build the SDK client on first live call."""
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise LLMConfigError(
                "GEMINI_API_KEY is not set and this request is not in the cache. "
                "Copy .env.example to .env and add a key from https://aistudio.google.com, "
                "or run against a populated cache."
            )

        # Imported here so an offline run never loads the SDK at all.
        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=self._api_key,
            http_options=types.HttpOptions(timeout=int(self.timeout * 1000)),
        )
        return self._client

    def available_models(self) -> list[str]:
        """Model ids this key can actually call, for diagnosing a bad model id."""
        client = self._gemini()
        names = []
        for model in client.models.list():
            name = (model.name or "").removeprefix("models/")
            actions = getattr(model, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                names.append(name)
        return sorted(names)

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
        effective_temperature = self.temperature if temperature is None else temperature

        # Everything that can change the response goes into the key, so a
        # changed prompt, model, seed or temperature produces a fresh call
        # rather than replaying a stale answer.
        request = {
            "model": self.model,
            "messages": list(messages),
            "system": system or "",
            "temperature": effective_temperature,
            "seed": self.seed,
            "max_output_tokens": max_tokens or self.max_tokens,
            "stop_sequences": list(stop_sequences) if stop_sequences else [],
        }

        key = cache_key({**request, "_tags": tags or {}})
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

        text, meta = self._live_call(request, key, tags or {})
        self.cache.put(
            key,
            {
                "text": text,
                "model": meta.model,
                "input_tokens": meta.input_tokens,
                "output_tokens": meta.output_tokens,
                "stop_reason": meta.stop_reason,
                "request": _redact_request(request),
                "tags": tags or {},
            },
        )
        return text, meta

    def _live_call(self, request: dict[str, Any], key: str, tags: dict[str, Any]) -> tuple[str, CallMeta]:
        from google.genai import errors, types

        if self.live_calls >= self.max_calls:
            raise LLMBudgetError(
                f"Live-call budget exhausted ({self.max_calls} calls). "
                "Raise LLM_MAX_CALLS deliberately if this run really needs more - "
                "the cap exists so a loop over the dataset cannot drain a quota."
            )

        client = self._gemini()

        config = types.GenerateContentConfig(
            system_instruction=request["system"] or None,
            temperature=request["temperature"],
            seed=request["seed"],
            max_output_tokens=request["max_output_tokens"],
            stop_sequences=request["stop_sequences"] or None,
            # No tools are declared, so the SDK's automatic function calling has
            # nothing to do. Disabling it silences its warning and removes a
            # code path this project never wants entered.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = [
            types.Content(
                role="model" if m.get("role") == "assistant" else "user",
                parts=[types.Part(text=str(m.get("content", "")))],
            )
            for m in request["messages"]
        ]

        self.live_calls += 1
        response = None

        # The free tier allows 15 requests per minute per model, and a 429 names
        # the exact wait in its retryDelay. Honouring it lets a full evaluation
        # run finish unattended instead of failing partway and reporting quota
        # errors as if they were quality failures.
        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=request["model"], contents=contents, config=config
                )
                break
            except errors.ClientError as exc:
                message = str(exc)
                lowered = message.lower()

                if "api key" in lowered or "unauthenticated" in lowered or "permission" in lowered:
                    raise LLMConfigError(f"The API rejected this key: {message}") from exc
                if "not found" in lowered or "404" in lowered:
                    raise LLMError(
                        f"Model {request['model']!r} is not available to this key. "
                        "Set GEMINI_MODEL in .env to one the key can call."
                    ) from exc

                rate_limited = "resource_exhausted" in lowered or "429" in lowered or "quota" in lowered
                if rate_limited and attempt < self.max_retries:
                    wait = _retry_delay(message)
                    logger.warning(
                        "Rate limited, waiting %.0fs before retry %d of %d.",
                        wait, attempt + 1, self.max_retries,
                    )
                    time.sleep(wait)
                    continue
                if rate_limited:
                    raise LLMError(f"Rate limit or quota exceeded: {message}") from exc
                raise LLMError(f"Request rejected: {message}") from exc
            except errors.ServerError as exc:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise LLMError(f"Provider error, safe to retry: {exc}") from exc
            except (LLMError, KeyboardInterrupt):
                raise
            except Exception as exc:  # network failures surface as plain exceptions
                raise LLMError(f"Could not reach the API: {type(exc).__name__}: {exc}") from exc

        if response is None:
            raise LLMError("Request failed after exhausting retries.")

        usage = getattr(response, "usage_metadata", None)
        finish = None
        if getattr(response, "candidates", None):
            finish = getattr(response.candidates[0], "finish_reason", None)
            finish = getattr(finish, "name", None) or (str(finish) if finish else None)

        meta = CallMeta(
            model=request["model"],
            cached=False,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            stop_reason=finish,
            cache_key=key,
            tags=tags,
        )
        self.last_meta = meta

        if finish and finish.upper() not in ("STOP", "FINISH_REASON_STOP"):
            logger.warning("Response stopped for reason %s and may be incomplete.", finish)

        return extract_text(response), meta


def _retry_delay(message: str, default: float = 30.0) -> float:
    """Seconds to wait after a 429, taken from the API's own retryDelay."""
    match = re.search(r"retry in ([\d.]+)s", message) or re.search(r"'retryDelay': '(\d+)s'", message)
    if match:
        try:
            return min(float(match.group(1)) + 2.0, 90.0)
        except ValueError:
            pass
    return default


def extract_text(response) -> str:
    """Pull the text out of a response, tolerating a multi-part candidate."""
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                parts.append(part.text)
    return "\n".join(parts).strip()


def _redact_request(request: dict[str, Any]) -> dict[str, Any]:
    """Store request shape in the cache without duplicating full prompt bodies."""
    return {
        "model": request.get("model"),
        "temperature": request.get("temperature"),
        "seed": request.get("seed"),
        "max_output_tokens": request.get("max_output_tokens"),
        "system_chars": len(request.get("system") or ""),
        "message_chars": sum(len(str(m.get("content", ""))) for m in request.get("messages", [])),
    }
