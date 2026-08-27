"""Retrieval over the knowledge base.

A deliberately dependency-free retriever: BM25 lexical scoring plus two signals
that are specific to this corpus and do most of the useful work.

**Why not embeddings.** Every error code that appears in a ticket body
(``SCHEMA_MISMATCH``, ``ERR_CONNECTION_TIMEOUT``, ``AUTH_TOKEN_EXPIRED`` and
seven others) is documented verbatim in the knowledge base. Exact symbol
matching finds those with certainty, where a dense retriever would blur them
into nearby prose. The corpus is also small - 46 chunks, ~38k characters - so
lexical scoring over the whole set is fast, has no index to maintain, no
embedding API to pay for, and stays inspectable when a retrieval looks wrong.

**Why not filter hard on product.** The error codes in tickets are scattered
across products: ``SCHEMA_MISMATCH`` is documented under DataBridge Pro but
turns up in SecureVault and CloudSync tickets. Product is therefore a *boost*,
never a filter, and the two cross-product ``troubleshooting/`` documents stay
reachable from any ticket.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from src.data_loader import KBChunk, load_knowledge_base

# Error codes and other SHOUTING_SYMBOLS. Kept as one token so BM25 treats
# SCHEMA_MISMATCH as a unit rather than "schema" + "mismatch".
_CODE_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+\b")
_WORD_RE = re.compile(r"[a-z0-9_]+")

# Very common words carry no discriminating signal in a corpus this small.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "for", "from",
    "has", "have", "how", "i", "if", "in", "is", "it", "its", "me", "not", "of",
    "on", "or", "our", "so", "that", "the", "then", "this", "to", "we", "what",
    "when", "which", "with", "you", "your",
}

# Scoring weights. Tuned by hand against the ticket templates in the dataset;
# the exact-code bonus dominates because a code match is near-certain evidence.
CODE_MATCH_BONUS = 12.0
PRODUCT_BOOST = 3.0
AREA_BOOST = 4.0
TROUBLESHOOTING_BOOST = 0.5


def extract_error_codes(text: str) -> list[str]:
    """Pull UPPER_SNAKE error codes out of free text, in order, deduplicated."""
    seen: list[str] = []
    for match in _CODE_RE.finditer(text or ""):
        code = match.group(0)
        if code not in seen:
            seen.append(code)
    return seen


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with error codes preserved as single tokens."""
    codes = [code.lower() for code in extract_error_codes(text)]
    words = [w for w in _WORD_RE.findall((text or "").lower()) if w not in _STOPWORDS and len(w) > 1]
    return words + codes


@dataclass
class RetrievalHit:
    """One scored chunk, with the reason it scored."""

    chunk: KBChunk
    score: float
    matched_codes: list[str]
    reasons: list[str]

    def excerpt(self, limit: int = 400) -> str:
        text = " ".join(self.chunk.text.split())
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


class KnowledgeBaseIndex:
    """BM25 index over knowledge-base chunks, built once and reused.

    Construction is cheap (a few milliseconds for this corpus), so the API
    process builds one at startup and every request shares it.
    """

    def __init__(self, chunks: Optional[Sequence[KBChunk]] = None, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks: list[KBChunk] = list(chunks) if chunks is not None else load_knowledge_base()
        self.k1 = k1
        self.b = b

        self._tokens: list[Counter[str]] = []
        self._lengths: list[int] = []
        self._codes: list[set[str]] = []

        document_frequency: Counter[str] = Counter()
        for chunk in self.chunks:
            # Headings describe what a chunk is about, so they are worth more
            # than one mention of a word buried in the body.
            heading_text = " ".join(chunk.heading_path + chunk.headings)
            counts = Counter(tokenize(f"{heading_text} {heading_text} {chunk.text}"))
            self._tokens.append(counts)
            self._lengths.append(sum(counts.values()))
            self._codes.append(set(extract_error_codes(chunk.text)))
            document_frequency.update(counts.keys())

        total = len(self.chunks) or 1
        self.average_length = (sum(self._lengths) / total) if self._lengths else 0.0
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def _bm25(self, query_tokens: Iterable[str], index: int) -> float:
        counts = self._tokens[index]
        length = self._lengths[index] or 1
        score = 0.0
        for term in query_tokens:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = self._idf.get(term, 0.0)
            denominator = frequency + self.k1 * (1 - self.b + self.b * length / (self.average_length or 1))
            score += idf * (frequency * (self.k1 + 1)) / denominator
        return score

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        product: Optional[str] = None,
        product_area: Optional[str] = None,
        min_score: float = 0.5,
    ) -> list[RetrievalHit]:
        """Rank chunks for a query.

        ``product`` and ``product_area`` are hints that boost matching chunks;
        they never exclude anything, so a cross-product troubleshooting doc can
        still win when it is the better answer.
        """
        query_tokens = tokenize(query)
        query_codes = set(extract_error_codes(query))

        hits: list[RetrievalHit] = []
        for index, chunk in enumerate(self.chunks):
            score = self._bm25(query_tokens, index)
            reasons: list[str] = []

            shared_codes = sorted(query_codes & self._codes[index])
            if shared_codes:
                score += CODE_MATCH_BONUS * len(shared_codes)
                reasons.append(f"documents error code {', '.join(shared_codes)}")

            haystack = f"{chunk.source} {chunk.breadcrumb} {' '.join(chunk.headings)}".lower()
            if product and product.lower() in haystack:
                score += PRODUCT_BOOST
                reasons.append(f"product reference for {product}")
            if product_area and product_area.lower() in haystack:
                score += AREA_BOOST
                reasons.append(f"covers the {product_area} area")
            if chunk.category == "troubleshooting":
                score += TROUBLESHOOTING_BOOST

            if score >= min_score:
                hits.append(
                    RetrievalHit(
                        chunk=chunk,
                        score=round(score, 3),
                        matched_codes=shared_codes,
                        reasons=reasons,
                    )
                )

        # Sort by score, then chunk_id, so ties resolve identically every run -
        # the eval harness depends on this being stable.
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        return hits[:top_k]

    def retrieve_for_ticket(
        self,
        subject: str,
        body: str,
        *,
        top_k: int = 4,
        product: Optional[str] = None,
        product_area: Optional[str] = None,
    ) -> list[RetrievalHit]:
        """Convenience wrapper: subject is weighted above body."""
        query = f"{subject} {subject} {body}"
        return self.search(query, top_k=top_k, product=product, product_area=product_area)


def format_hits_for_prompt(hits: Sequence[RetrievalHit], *, excerpt_chars: int = 900) -> str:
    """Render retrieved chunks as prompt context with stable citation ids."""
    if not hits:
        return "(no knowledge-base sections matched this ticket)"

    blocks: list[str] = []
    for position, hit in enumerate(hits, 1):
        text = " ".join(hit.chunk.text.split())
        if len(text) > excerpt_chars:
            text = text[: excerpt_chars - 1].rstrip() + "..."
        blocks.append(
            f"[{position}] chunk_id: {hit.chunk.chunk_id}\n"
            f"    document: {hit.chunk.source}\n"
            f"    section: {hit.chunk.breadcrumb}\n"
            f"    content: {text}"
        )
    return "\n\n".join(blocks)


_DEFAULT_INDEX: Optional[KnowledgeBaseIndex] = None


def get_index() -> KnowledgeBaseIndex:
    """Process-wide index, built on first use."""
    global _DEFAULT_INDEX
    if _DEFAULT_INDEX is None:
        _DEFAULT_INDEX = KnowledgeBaseIndex()
    return _DEFAULT_INDEX


if __name__ == "__main__":
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    index = get_index()
    print(f"indexed {len(index)} chunks, average length {index.average_length:.1f} tokens\n")

    probes = [
        ("Pipeline failing with ERR_CONNECTION_TIMEOUT after 30s", "DataBridge Pro", "Connectors"),
        ("New joiners cannot sign in after SSO migration", "SecureVault", "SSO"),
        ("Dashboard takes over 40 seconds to load", "AnalyticsHub", "Dashboard"),
        ("We want to upgrade from Starter to Business plan", None, None),
    ]
    for query, product, area in probes:
        print(f"--- {query!r}  (product={product}, area={area})")
        for hit in index.search(query, product=product, product_area=area, top_k=3):
            why = "; ".join(hit.reasons) or "lexical overlap"
            print(f"    {hit.score:>7.2f}  {hit.chunk.chunk_id:<48} {why}")
        print()
