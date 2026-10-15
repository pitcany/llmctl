"""Pure helpers: query parsing, parameter inference, quantization detection.

No I/O here — these functions are deterministic and unit-testable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

QUANT_PATTERNS: dict[str, tuple[str, ...]] = {
    "awq": ("awq",),
    "gptq": ("gptq",),
    "fp8": ("fp8", "-fp8", "_fp8"),
    "gguf": ("gguf", "q4_k", "q5_k", "q6_k", "q8_0", "q4_0"),
    "bnb": ("bnb", "bitsandbytes", "-nf4", "-int4", "-int8"),
}

FAMILY_HINTS: tuple[tuple[str, str], ...] = (
    ("llama", "llama"),
    ("qwen", "qwen"),
    ("deepseek", "deepseek"),
    ("mistral", "mistral"),
    ("mixtral", "mixtral"),
    ("gemma", "gemma"),
    ("phi", "phi"),
    ("yi", "yi"),
    ("command-r", "command-r"),
    ("falcon", "falcon"),
)

SIZE_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*[bB](?![a-z])")
MOE_RE = re.compile(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*[bB]", re.I)

# Words/symbols immediately before a size that mean "no larger than".
_MAX_SIZE_RE = re.compile(
    r"(?:under|below|less\s+than|fewer\s+than|max(?:imum)?|up\s+to|<=?)\s*"
    r"(\d+(?:\.\d+)?)\s*[bB]\b",
    re.I,
)


@dataclass
class ParsedQuery:
    """Structured view of a natural-language model query."""

    raw: str
    tokens: list[str]
    family: str | None = None
    param_count_b: float | None = None
    quantization: str | None = None
    moe: tuple[int, float] | None = None  # (experts, expert_size_b)
    keywords: list[str] = field(default_factory=list)
    # True when the size is a ceiling ("under 35B"), False when it's a target ("70B").
    size_is_max: bool = False

    def is_model_id(self) -> bool:
        """Heuristic: 'org/name' shape with no spaces."""
        return "/" in self.raw and " " not in self.raw.strip()


def detect_quantization(text: str) -> str | None:
    """Return canonical quant label (awq/gptq/fp8/gguf/bnb) or None."""
    low = text.lower()
    for label, needles in QUANT_PATTERNS.items():
        if any(n in low for n in needles):
            return label
    return None


def detect_family(text: str) -> str | None:
    low = text.lower()
    for needle, canonical in FAMILY_HINTS:
        if needle in low:
            return canonical
    return None


def detect_param_count(text: str) -> float | None:
    """Pull the first '<N>B' style number out of a string."""
    moe = MOE_RE.search(text)
    if moe:
        # For MoE, return the per-expert size for reference; total handled separately.
        return float(moe.group(2))
    m = SIZE_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def detect_moe(text: str) -> tuple[int, float] | None:
    m = MOE_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2))


# Map user-facing keyword variants to the canonical stem that appears in real HF
# model ids. HF's `search` parameter is a substring filter, so the canonical form
# is what gets sent to HF; the original keyword is still credited during scoring.
_KEYWORD_ALIASES: dict[str, str] = {
    "coding": "coder",
    "code": "coder",
}


def canonicalize_keyword(kw: str) -> str:
    """Return the alias-mapped form of a keyword (or itself if no mapping)."""
    return _KEYWORD_ALIASES.get(kw, kw)


_STOPWORDS = {
    "the",
    "a",
    "an",
    "for",
    "on",
    "of",
    "with",
    "and",
    "or",
    "best",
    "model",
    "models",
    "version",
    "instruct",
    "chat",
    # Comparison / range words that aren't useful as HF search substrings.
    "under",
    "below",
    "above",
    "over",
    "max",
    "min",
    "less",
    "more",
    "than",
    "fewer",
    "around",
    "about",
    "approximately",
    "approx",
}


def parse_query(raw: str) -> ParsedQuery:
    """Break a natural-language query into structured signals."""
    raw = raw.strip()
    tokens = [t for t in re.split(r"[\s,]+", raw.lower()) if t]
    keywords = [t for t in tokens if t not in _STOPWORDS and not SIZE_RE.fullmatch(t)]
    max_match = _MAX_SIZE_RE.search(raw)
    size_is_max = max_match is not None
    param_count_b = float(max_match.group(1)) if max_match else detect_param_count(raw)
    return ParsedQuery(
        raw=raw,
        tokens=tokens,
        family=detect_family(raw),
        param_count_b=param_count_b,
        quantization=detect_quantization(raw),
        moe=detect_moe(raw),
        keywords=keywords,
        size_is_max=size_is_max,
    )


def score_model(
    model_id: str, tags: Iterable[str], parsed: ParsedQuery, downloads: int = 0, likes: int = 0
) -> tuple[float, list[str]]:
    """Score a candidate model against the parsed query.

    Returns (score, reasons). Higher is better. Reasons explain the match.
    """
    reasons: list[str] = []
    score = 0.0
    mid_low = model_id.lower()
    tag_set = {t.lower() for t in tags}

    if parsed.family and parsed.family in mid_low:
        score += 3.0
        reasons.append(f"family match: {parsed.family}")

    if parsed.param_count_b is not None:
        model_size = detect_param_count(model_id)
        if parsed.size_is_max:
            if model_size is not None and model_size <= parsed.param_count_b:
                score += 3.0
                reasons.append(f"size ≤ {parsed.param_count_b:g}B")
            elif model_size is not None:
                # Over the user's ceiling — penalize proportionally.
                score -= 2.0 + (model_size - parsed.param_count_b) * 0.1
        else:
            size_re = re.compile(rf"(?<![a-z0-9]){parsed.param_count_b:g}b", re.I)
            if size_re.search(mid_low):
                score += 4.0
                reasons.append(f"size match: {parsed.param_count_b:g}B")

    if parsed.quantization:
        if parsed.quantization in mid_low or parsed.quantization in tag_set:
            score += 2.5
            reasons.append(f"quant match: {parsed.quantization}")
    else:
        # User did not ask for a quant; mildly prefer non-GGUF for vLLM.
        if "gguf" in mid_low:
            score -= 0.5

    for kw in parsed.keywords:
        if len(kw) < 3:
            continue
        canon = canonicalize_keyword(kw)
        if kw in mid_low or (canon != kw and canon in mid_low):
            score += 0.5
            reasons.append(f"keyword: {kw}")

    # Popularity tiebreaker (log-scaled).
    import math

    if downloads > 0:
        score += min(2.0, math.log10(downloads + 1) * 0.4)
    if likes > 0:
        score += min(1.0, math.log10(likes + 1) * 0.3)

    return score, reasons
