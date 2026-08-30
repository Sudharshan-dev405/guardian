# test_physiology.py -- focused tests for streams/physiology.py
# NOTE: context_state is passed via the window dict (window["context_state"]),
# NOT as a separate score() argument, to match the Stream contract's fixed
# score(window) -> float signature.
import math
import random
from streams.physiology import PhysiologyStream


def w(t, hr, context_state=None):
    d = {"t": t, "acc": None, "gyro": None, "hr": hr, "temp": None, "fs": 50}
    if context_state is not None:
        d["context_state"] = context_state
    return d


def run_windows(ps, values):
    """values: list of (t, hr, context_state). Returns list of scores."""
    scores = []
    for t, hr, ctx in values:
        scores.append(ps.score(w(t, hr, ctx)))
    return scores


print("=== Case 1: normal HR ~72 -> score stays ~0 ===")
ps = PhysiologyStream()
seq = [(i * 1.25, 72.0 + random.uniform(-1, 1), None) for i in range(10)]
scores = run_windows(ps, seq)
print("scores:", [round(s, 3) for s in scores])
assert all(s < 0.05 for s in scores), "normal HR should score near zero"
print("PASS\n")

print("=== Case 2: small HR deviation (102 bpm, single window) -> near zero ===")
ps2 = PhysiologyStream()
score = ps2.score(w(0.0, 102.0))
print(f"score={score:.4f}")
assert score < 0.05, "small single-window deviation should stay near zero"
print("PASS\n")

print("=== Case 3: single short spike (150 bpm) then recovery -> near zero, then zero ===")
ps3 = PhysiologyStream()
spike_score = ps3.score(w(0.0, 150.0))
recover_score = ps3.score(w(1.25, 72.0))
print(f"spike={spike_score:.4f} recover={recover_score:.4f}")
assert spike_score < 0.1, "single short spike should score near zero despite large deviation"
assert recover_score == 0.0, "back in normal range should score exactly zero"
print("PASS\n")

print("=== Case 4: sustained elevated HR (~105 bpm) for ~30s -> ramps toward full raw risk ===")
ps4 = PhysiologyStream()
seq = [(i * 1.25, 105.0, None) for i in range(25)]  # 25 * 1.25 = 31.25s
scores = run_windows(ps4, seq)
print("first 3:", [round(s, 3) for s in scores[:3]])
print("last 3: ", [round(s, 3) for s in scores[-3:]])
assert scores[0] < scores[-1], "score should ramp up over the sustained period"
assert scores[-1] > 0.3, "sustained ~30s at 105bpm should approach full raw-risk weight"
print("PASS\n")

print("=== Case 5: elevated HR then recovery -> score rises then drops immediately ===")
ps5 = PhysiologyStream()
seq = [(i * 1.25, 105.0, None) for i in range(20)] + [(20 * 1.25, 72.0, None)]
scores = run_windows(ps5, seq)
print(f"peak={max(scores[:-1]):.3f} after_recovery={scores[-1]:.3f}")
assert scores[-1] == 0.0, "recovery to normal range should drop score to zero immediately"
print("PASS\n")

print("=== Case 6: missing/invalid HR -> score 0, quality 0 ===")
ps6 = PhysiologyStream()
for bad_hr in [None, float("nan"), -5.0, "not_a_number"]:
    score = ps6.score(w(0.0, bad_hr))
    assert score == 0.0 and ps6.last_quality == 0.0, f"failed for hr={bad_hr!r}"
print("PASS (None, NaN, negative, non-numeric all handled)\n")

print("=== Case 7: ambulating context (via window dict) gates output, pauses dwell ===")
ps7 = PhysiologyStream()
seq = [(i * 1.25, 105.0, None) for i in range(12)]  # ~15s sustained
run_windows(ps7, seq)
mid_sustained = ps7._sustained_seconds

gated_scores = run_windows(ps7, [(15.0, 105.0, "ambulating"), (16.25, 105.0, "ambulating")])
print(f"gated scores: {gated_scores}, quality after gating: {ps7.last_quality}")
assert all(s == 0.0 for s in gated_scores)
assert ps7.last_quality == 0.0
assert ps7._sustained_seconds == mid_sustained, "dwell clock should pause, not reset, during ambulating"

resumed = ps7.score(w(17.5, 105.0))
print(f"resumed score={resumed:.3f}, sustained={ps7._sustained_seconds:.2f}")
assert ps7._sustained_seconds > mid_sustained
print("PASS\n")

print("=== Case 7b: generic uniform score(window) call -- confirms contract fidelity ===")
# This is the exact calling pattern used in the pipeline/dashboard:
#   scores = {name: s.score(window) for name, s in streams.items()}
# It must work identically for physiology as for every other stream.
ps7b = PhysiologyStream()
generic_window = w(0.0, 105.0, "ambulating")
generic_score = ps7b.score(generic_window)  # single positional arg, no kwarg
assert generic_score == 0.0 and ps7b.last_quality == 0.0
print("PASS -- gate works via plain score(window), no special calling convention needed\n")

print("=== Case 8: score/quality always in [0,1], no NaN, across random inputs ===")
ps8 = PhysiologyStream()
random.seed(7)
t = 0.0
for _ in range(200):
    hr = random.choice([None, random.uniform(30, 180)])
    ctx = random.choice([None, "ambulating", "stationary", "lying"])
    score = ps8.score(w(t, hr, ctx))
    assert 0.0 <= score <= 1.0, f"score out of range: {score}"
    assert 0.0 <= ps8.last_quality <= 1.0, f"quality out of range: {ps8.last_quality}"
    assert not math.isnan(score)
    t += 1.25
print("PASS -- 200 random windows, no violations\n")

print("ALL PHYSIOLOGY TESTS PASSED")