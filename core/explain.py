"""
core/explain.py

STUB. Real implementation (Stage 7) will sort fusion's actual
contributions descending and render them in plain language, e.g.
"Emergency risk 0.82 -- high impact 0.31, immobility 0.28, ...".
It must use fusion's real numbers, never invent its own.

For now: a placeholder string, just to confirm the explanation panel
is wired into the pipeline and the dashboard.
"""


def explain_stub(risk: float, scores: dict) -> str:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    parts = ", ".join(f"{name} {val:.2f}" for name, val in ranked)
    return f"[STUB] risk {risk:.2f} -- {parts}"