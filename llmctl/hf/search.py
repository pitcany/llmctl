"""Hugging Face model search and ranking."""

from __future__ import annotations

from dataclasses import dataclass

from .parsing import canonicalize_keyword, parse_query, score_model


@dataclass
class Candidate:
    model_id: str
    downloads: int
    likes: int
    tags: list[str]
    pipeline_tag: str | None
    score: float
    reasons: list[str]


def search_models(query: str, limit: int = 10) -> list[Candidate]:
    """Search HF for models matching the query, return ranked candidates.

    Pulls a wider pool from HF, then re-ranks locally with our scoring function.
    """
    from huggingface_hub import HfApi  # imported lazily so the rest of the CLI works offline

    parsed = parse_query(query)
    api = HfApi()

    # HF's `search` is a substring filter; build a compact search string.
    search_terms: list[str] = []
    if parsed.family:
        search_terms.append(parsed.family)
    # Only pin an exact "<N>b" substring when the user asked for a specific size.
    # For "under 35B" we want a wider pool and filter via scoring instead.
    if parsed.param_count_b is not None and not parsed.size_is_max:
        search_terms.append(f"{parsed.param_count_b:g}b")
    if parsed.quantization:
        search_terms.append(parsed.quantization)
    # Always include the user's own keywords so things like 'coder' survive.
    # Map known synonyms (e.g. "coding" → "coder") to the form that actually
    # appears in HF model ids, otherwise the substring search misses the well-
    # known families (Qwen-Coder, DeepSeek-Coder, CodeLlama).
    for kw in parsed.keywords:
        term = canonicalize_keyword(kw)
        if term not in search_terms and len(term) >= 3:
            search_terms.append(term)
    search_string = " ".join(search_terms) or query

    # If the user typed a literal model id, just return it as the single result.
    if parsed.is_model_id():
        try:
            info = api.model_info(parsed.raw)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not fetch model info for '{parsed.raw}': {e}") from e
        tags = list(info.tags or [])
        return [
            Candidate(
                model_id=info.id,
                downloads=info.downloads or 0,
                likes=info.likes or 0,
                tags=tags,
                pipeline_tag=info.pipeline_tag,
                score=10.0,
                reasons=["exact model id"],
            )
        ]

    pool_size = max(50, limit * 5)
    raw_models = list(
        api.list_models(
            search=search_string,
            sort="downloads",  # the Hub sorts descending; hub 1.x removed `direction=`
            limit=pool_size,
            full=False,
        )
    )

    candidates: list[Candidate] = []
    for m in raw_models:
        tags = list(getattr(m, "tags", []) or [])
        downloads = getattr(m, "downloads", 0) or 0
        likes = getattr(m, "likes", 0) or 0
        pipeline_tag = getattr(m, "pipeline_tag", None)
        score, reasons = score_model(m.id, tags, parsed, downloads=downloads, likes=likes)
        candidates.append(
            Candidate(
                model_id=m.id,
                downloads=downloads,
                likes=likes,
                tags=tags,
                pipeline_tag=pipeline_tag,
                score=score,
                reasons=reasons,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:limit]


def best_match(query: str) -> Candidate:
    """Top-ranked candidate, or raise if none found."""
    results = search_models(query, limit=5)
    if not results:
        raise RuntimeError(f"No models matched query: {query!r}")
    return results[0]
