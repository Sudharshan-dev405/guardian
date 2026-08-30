"""
core/explain.py

REAL ranked explanation. Uses the EXACT contribution values returned
by core.fusion.fuse() -- never recomputes or re-derives numbers, and
never substitutes raw stream scores for contributions (a contribution
already reflects weight and any suppression fusion applied, which a
raw score does not).

INTERFACE
---------
explain(risk, contributions) -> str

- risk: the fused risk value (already computed by fusion).
- contributions: the exact dict returned by fuse() -- same numbers,
  no recalculation.
"""

LABELS = {
    "motion": "motion evidence",
    "context": "context evidence",
    "physiology": "heart rate elevated",
}


def explain(risk: float, contributions: dict) -> str:
    if not contributions:
        return f"Risk {risk:.2f} -- no reliable evidence available this window"

    ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    parts = ", ".join(f"{LABELS.get(name, name)} {value:.2f}" for name, value in ranked)
    return f"Emergency risk {risk:.2f} -- {parts}"