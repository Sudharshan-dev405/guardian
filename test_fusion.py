# test_fusion.py -- tests for core/fusion.py
import math
from types import SimpleNamespace
from core.fusion import fuse, WEIGHTS


def mock_context(last_state=None, last_confidence=0.0):
    return SimpleNamespace(last_state=last_state, last_confidence=last_confidence)


def mock_motion(time_since_impact=None):
    return SimpleNamespace(time_since_impact=time_since_impact)


print("=== Case 1: normal weighted calculation, no suppression ===")
scores = {"motion": 0.8, "context": 0.3, "physiology": 0.2}
risk, contrib = fuse(scores, mock_context(), mock_motion())
expected = (0.8 * WEIGHTS["motion"] + 0.3 * WEIGHTS["context"] + 0.2 * WEIGHTS["physiology"]) / sum(WEIGHTS.values())
print(f"risk={risk:.4f} expected={expected:.4f} contrib={contrib}")
assert math.isclose(risk, expected, abs_tol=1e-9)
assert math.isclose(contrib["motion"], 0.8 * WEIGHTS["motion"], abs_tol=1e-9)
print("PASS\n")

print("=== Case 2: contribution values are exactly weight*score (no suppression) ===")
for name in scores:
    assert math.isclose(contrib[name], WEIGHTS[name] * scores[name], abs_tol=1e-9)
print("PASS\n")

print("=== Case 3: seated hand activity suppression halves motion contribution only ===")
scores = {"motion": 0.9, "context": 0.3, "physiology": 0.1}
ctx = mock_context(last_state="seated hand activity", last_confidence=0.85)
mot = mock_motion(time_since_impact=None)
risk, contrib = fuse(scores, ctx, mot)
expected_motion_contrib = 0.9 * WEIGHTS["motion"] * 0.5
print(f"motion_contrib={contrib['motion']:.4f} expected={expected_motion_contrib:.4f}")
assert math.isclose(contrib["motion"], expected_motion_contrib, abs_tol=1e-9)
assert math.isclose(contrib["context"], 0.3 * WEIGHTS["context"], abs_tol=1e-9), "only motion should be halved"
assert math.isclose(contrib["physiology"], 0.1 * WEIGHTS["physiology"], abs_tol=1e-9)
print("PASS\n")

print("=== Case 4: suppression NOT applied when confidence <= 0.7 ===")
ctx_low_conf = mock_context(last_state="seated hand activity", last_confidence=0.7)
risk, contrib = fuse(scores, ctx_low_conf, mock_motion(time_since_impact=None))
assert math.isclose(contrib["motion"], 0.9 * WEIGHTS["motion"], abs_tol=1e-9), "confidence exactly 0.7 must NOT suppress (strictly greater than required)"
print("PASS\n")

print("=== Case 5: suppression blocked when an impact was just detected ===")
ctx_high_conf = mock_context(last_state="seated hand activity", last_confidence=0.95)
mot_impact = mock_motion(time_since_impact=0.3)  # impact 0.3s ago -- IS set
risk, contrib = fuse(scores, ctx_high_conf, mot_impact)
assert math.isclose(contrib["motion"], 0.9 * WEIGHTS["motion"], abs_tol=1e-9), "impact guard must block suppression even with high-confidence seated-activity context"
print("PASS\n")

print("=== Case 6: output always bounded in [0,1] ===")
risk, _ = fuse({"motion": 1.0, "context": 1.0, "physiology": 1.0}, mock_context(), mock_motion())
assert 0.0 <= risk <= 1.0
risk, _ = fuse({"motion": 0.0, "context": 0.0, "physiology": 0.0}, mock_context(), mock_motion())
assert 0.0 <= risk <= 1.0
print("PASS\n")

print("=== Case 7: missing/invalid scores handled safely ===")
risk, contrib = fuse({"motion": None, "context": 0.5, "physiology": float("nan")}, mock_context(), mock_motion())
print(f"risk={risk:.4f} contrib={contrib}")
assert "motion" not in contrib and "physiology" not in contrib
assert math.isclose(risk, 0.5, abs_tol=1e-9), "with only context valid, risk should equal context's score"
print("PASS\n")

print("=== Case 8: all scores missing -> safe default, empty contributions ===")
risk, contrib = fuse({"motion": None, "context": None, "physiology": None}, mock_context(), mock_motion())
assert risk == 0.0 and contrib == {}
print("PASS\n")

print("=== Case 9: fusion works with stub streams lacking last_state/time_since_impact ===")
class BareStub:
    pass

risk, contrib = fuse({"motion": 0.5, "context": 0.3, "physiology": 0.2}, BareStub(), BareStub())
print(f"risk={risk:.4f}")
assert 0.0 <= risk <= 1.0
print("PASS -- no crash on stub-stage stream objects\n")

print("ALL FUSION TESTS PASSED")