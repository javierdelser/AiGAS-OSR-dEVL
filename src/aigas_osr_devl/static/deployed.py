"""Deployed static rescoring rules for LETTER and PENDIGITS (paper Table 2)."""

from __future__ import annotations

from typing import Any

# (dataset, nominal_openness) -> rule path inside gng_mahalanobis / improvements / gng_variants JSON
DEPLOYED_RULES: dict[tuple[str, float], str] = {
    ("LETTER", 0.0): "knn5__dist5__val",
    ("LETTER", 15.48): "maha_k5_eucl_label__val",
    ("LETTER", 25.46): "maha_k5_eucl_label__train_q95",
    ("PENDIGITS", 0.0): "knn5__dist5__val",
    ("PENDIGITS", 8.71): "relabel_k5_copied",
    ("PENDIGITS", 18.35): "relabel_k5_copied",
}


def deployed_rule_for(dataset: str, nominal_openness: float) -> str:
    key = (dataset.upper(), round(float(nominal_openness), 2))
    if key not in DEPLOYED_RULES:
        raise KeyError(f"No deployed rule for {dataset} O={nominal_openness}")
    return DEPLOYED_RULES[key]


def paper_f1_from_row(row: dict[str, Any], rule_key: str) -> float:
    """Extract paper micro-F1 from a rescoring JSON row (improvements / gng_* modules)."""
    rules = row.get("rules") or {}
    payload = rules.get(rule_key)
    if isinstance(payload, dict) and "paper_f1_percent" in payload:
        return float(payload["paper_f1_percent"])
    raise KeyError(f"Rule {rule_key!r} missing in row; have {list(rules)[:12]}")