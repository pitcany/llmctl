"""Comparison eligibility for benchmark results (FIXPLAN 2026-11-03 §1D).

A green "+12.3 tok/s" delta is a scientific claim: *this change made it
faster*. That claim is only meaningful when the two runs measured the same
thing. This module owns that judgement once, so the TUI, CLI, and any other
surface render deltas through the same gate instead of each reinventing
(approximations of) it.

What blocks comparison:

* either run failed,
* different execution modes (live vs synthetic),
* different benchmark kinds,
* different workload (manifest hash, or workload size when hashes are absent),
* different generation settings (max_tokens, temperature),
* different token-count methods (estimated vs server usage),
* different serving hardware (when both manifests recorded it).

What deliberately does NOT block comparison: the model, engine, or endpoint
identity -- those are the intended experimental variables. Identical
manifests are not required for a model comparison.

Unknown information (legacy rows without manifests or workload metadata)
does not block comparison when symmetric: if neither side recorded the fact,
there is no evidence of a difference. Asymmetric unknowns (one recorded, one
not) are reported as a reason so the operator can see the gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ComparisonVerdict:
    """Whether two benchmark results may be presented as a comparison."""

    eligible: bool
    reasons: list[str]

    @property
    def summary(self) -> str:
        """One-line human summary (empty when eligible)."""
        return "; ".join(self.reasons)


def _params(result: Any) -> dict[str, Any]:
    return getattr(result, "parameters", None) or {}


def _kind(result: Any) -> str | None:
    kind = getattr(result, "kind", None)
    return kind.value if hasattr(kind, "value") else (str(kind) if kind else None)


def _manifest(result: Any) -> dict[str, Any]:
    return getattr(result, "manifest", None) or {}


def _compare_recorded(
    label: str, left: Any, right: Any, reasons: list[str]
) -> None:
    """Record a reason when one side recorded a fact the other did not."""
    if left is None and right is None:
        return  # symmetric unknown: no evidence of a difference
    if left is None or right is None:
        reasons.append(f"{label} recorded on only one run")


def comparison_verdict(candidate: Any, baseline: Any) -> ComparisonVerdict:
    """Decide whether ``candidate`` may be compared against ``baseline``."""
    reasons: list[str] = []

    # 1. Both runs must have succeeded.
    if not getattr(candidate, "success", False) or not getattr(baseline, "success", False):
        reasons.append("one or both runs failed")

    # 2. Same execution mode (live vs synthetic is never a comparison).
    cand_mode = str(_params(candidate).get("mode", "?"))
    base_mode = str(_params(baseline).get("mode", "?"))
    if cand_mode != base_mode:
        reasons.append(f"{cand_mode} vs {base_mode} execution mode")

    # 3. Same benchmark kind (chat vs health is not a comparison).
    cand_kind, base_kind = _kind(candidate), _kind(baseline)
    if cand_kind != base_kind:
        reasons.append(
            f"different benchmark kinds ({cand_kind or 'unknown'} vs {base_kind or 'unknown'})"
        )

    # 4. Same workload. Prefer manifest hashes; fall back to recorded
    #    workload size for legacy rows.
    cand_manifest, base_manifest = _manifest(candidate), _manifest(baseline)
    cand_hash = (cand_manifest.get("workload") or {}).get("hash")
    base_hash = (base_manifest.get("workload") or {}).get("hash")
    if cand_hash and base_hash:
        if cand_hash != base_hash:
            reasons.append("different workload (manifest hash mismatch)")
    else:
        _compare_recorded("workload hash", cand_hash, base_hash, reasons)
        cand_size = _params(candidate).get("total_requests")
        base_size = _params(baseline).get("total_requests")
        if cand_size is not None and base_size is not None and cand_size != base_size:
            reasons.append(f"different workload ({cand_size} vs {base_size} requests)")

    # 5. Same generation settings (only compare what both recorded).
    for key, label in (("max_tokens", "max_tokens"), ("temperature", "temperature")):
        cand_value = _params(candidate).get(key)
        base_value = _params(baseline).get(key)
        _compare_recorded(label, cand_value, base_value, reasons)
        if (
            cand_value is not None
            and base_value is not None
            and cand_value != base_value
        ):
            reasons.append(f"different {label} ({cand_value} vs {base_value})")

    # 6. Same token-count method (estimated counts are not precise counts).
    cand_source = str(_params(candidate).get("token_count_source", "") or "")
    base_source = str(_params(baseline).get("token_count_source", "") or "")
    if cand_source and base_source and cand_source != base_source:
        reasons.append(f"different token-count methods ({cand_source} vs {base_source})")

    # 7. Same serving hardware, when both manifests recorded it.
    cand_gpus = (cand_manifest.get("serving_host") or {}).get("gpu_identity")
    base_gpus = (base_manifest.get("serving_host") or {}).get("gpu_identity")
    if cand_gpus is not None and base_gpus is not None:
        if sorted(map(str, cand_gpus)) != sorted(map(str, base_gpus)):
            reasons.append("different serving hardware")

    return ComparisonVerdict(eligible=not reasons, reasons=reasons)
