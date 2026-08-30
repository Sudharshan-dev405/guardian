# test_decision.py -- tests for core/decision.py
from core.decision import DecisionStateMachine, ENTRY_THRESHOLD, EXIT_THRESHOLD, DWELL_SECONDS


def run(machine, risk_sequence, dt=1.25, contributions=None):
    """risk_sequence: list of risk values. Returns list of resulting states."""
    states = []
    t = 0.0
    for risk in risk_sequence:
        states.append(machine.update(risk, t, contributions=contributions))
        t += dt
    return states


print("=== Case 1: normal state initially and under low risk ===")
m = DecisionStateMachine()
states = run(m, [0.1, 0.2, 0.15, 0.05], contributions={"motion": 0.1})
print(states)
assert all(s == "normal" for s in states)
print("PASS\n")

print("=== Case 2: crossing entry threshold leaves normal ===")
m = DecisionStateMachine()
states = run(m, [0.2, 0.6], contributions={"motion": 0.6})
print(states)
assert states[0] == "normal"
assert states[1] != "normal"
print("PASS\n")

print("=== Case 3: hysteresis -- dropping into the 0.40-0.55 band does not snap back to normal ===")
m = DecisionStateMachine()
states = run(m, [0.6, 0.45], contributions={"motion": 0.6})
print(states)
assert states[0] != "normal"
assert states[1] == states[0], "should hold the elevated state, not flicker to normal"
print("PASS\n")

print("=== Case 4: no premature escalation -- brief spike under 30s does not escalate ===")
m = DecisionStateMachine()
# 0.6 risk for 5 windows (~6.25s), then drop below exit
states = run(m, [0.6] * 5 + [0.1], contributions={"motion": 0.6})
print(states)
assert "escalate" not in states, "should not escalate before ~30s sustained"
print("PASS\n")

print("=== Case 5: escalation after ~30s sustained risk above entry ===")
m = DecisionStateMachine()
n_windows = int(DWELL_SECONDS / 1.25) + 2  # comfortably past 30s
states = run(m, [0.6] * n_windows, contributions={"motion": 0.6})
print("last state:", states[-1], " reached escalate at index:", states.index("escalate") if "escalate" in states else None)
assert states[-1] == "escalate"
assert states[0] != "escalate", "should not escalate on the very first elevated window"
print("PASS\n")

print("=== Case 6: recovery via exit threshold after escalate ===")
m = DecisionStateMachine()
n_windows = int(DWELL_SECONDS / 1.25) + 2
run(m, [0.6] * n_windows, contributions={"motion": 0.6})
assert m.state == "escalate"
recovered = m.update(0.2, n_windows * 1.25, contributions={"motion": 0.2})
print("after drop below exit:", recovered)
assert recovered == "normal"
print("PASS\n")

print("=== Case 7: manual SOS bypasses everything, immediately ===")
m = DecisionStateMachine()
state = m.update(0.0, 0.0, manual_sos=True)
print(state)
assert state == "escalate"
print("PASS\n")

print("=== Case 8: sensor uncertainty when contributions are empty ===")
m = DecisionStateMachine()
state = m.update(0.0, 0.0, contributions={})
print(state)
assert state == "sensor uncertainty"
print("PASS\n")

print("=== Case 9: dominant contributor selects motion event vs physiological confirmation ===")
m1 = DecisionStateMachine()
s1 = m1.update(0.6, 0.0, contributions={"motion": 0.5, "physiology": 0.1})
m2 = DecisionStateMachine()
s2 = m2.update(0.6, 0.0, contributions={"motion": 0.1, "physiology": 0.5})
print("motion-dominant:", s1, " physiology-dominant:", s2)
assert s1 == "motion event"
assert s2 == "physiological confirmation"
print("PASS\n")

print("ALL DECISION TESTS PASSED")