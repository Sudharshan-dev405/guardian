"""
core/fusion.py

STUB. Real implementation (Stage 5) will be:

    contribution_i = quality_i * weight_i * score_i
    risk = sum(contribution_i) / sum(quality_i * weight_i)

with the seated-hand-activity motion-halving rule. That formula is
what produces demo scenario 1 and must not be skipped later.

For now: plain unweighted average of the four stream scores, so the
pipeline has a "risk" value to hand to decision/explain/dashboard.
"""


def fuse_stub(scores: dict) -> float:
    """
    scores: dict like {"motion": 0.6, "physiology": 0.2,
                        "context": 0.3, "quality": 0.5}
    Returns a float in [0,1].
    """
    values = list(scores.values())
    risk = sum(values) / len(values)
    return max(0.0, min(1.0, risk))