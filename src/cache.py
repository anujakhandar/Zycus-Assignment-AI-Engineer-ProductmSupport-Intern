"""Content-addressed cache for model responses.

Task 2 requires that the same input always produces the same output. The
Messages API no longer exposes sampling controls at all - ``temperature``,
``top_p`` and ``top_k`` were removed - so there is no setting that makes
generation repeatable. This cache is what provides it: every request is hashed
by everything that affects the result, meaning model id, system prompt,
messages, max_tokens and prompt version, and the stored response is replayed
whenever that key is seen again.

Two consequences worth knowing:

* The eval harness and the CLI are reproducible run to run, which is what makes
  a regression test meaningful.
* Once the cache is populated and committed, the pipelines run with no network
  access at all, so CI and a reviewer without an API key can still execute
  everything end to end.

Set ``LLM_CACHE=off`` to bypass reads (writes still happen) when deliberately
regenerating responses.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / "fixtures" / "llm_cache"


def cache_key(payload: dict[str, Any]) -> str:
    """Stable sha256 over a request payload.

    ``sort_keys`` matters: a dict that serialises in a different order would
    otherwise produce a different key for an identical request.
    """
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    """A flat directory of ``<sha256>.json`` files, one per request."""

    def __init__(self, directory: Path | str = DEFAULT_CACHE_DIR, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled and os.getenv("LLM_CACHE", "on").lower() != "off"
        self.hits = 0
        self.misses = 0
        self.writes = 0

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Return the stored record, or None on a miss."""
        if not self.enabled:
            self.misses += 1
            return None

        path = self.path_for(key)
        if not path.exists():
            self.misses += 1
            return None

        try:
            with path.open(encoding="utf-8") as handle:
                record = json.load(handle)
        except (json.JSONDecodeError, OSError):
            # A corrupt entry should degrade to a miss, never crash a run.
            self.misses += 1
            return None

        self.hits += 1
        return record

    def put(self, key: str, record: dict[str, Any]) -> None:
        """Store a record. Always writes, even when reads are disabled."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False, sort_keys=True)
        temporary.replace(path)
        self.writes += 1

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes}

    def __len__(self) -> int:
        if not self.directory.exists():
            return 0
        return len(list(self.directory.glob("*.json")))
