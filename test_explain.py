# test_explain.py -- tests for core/explain.py
from core.explain import explain


print("=== Case 1: contributions ranked descending in the output ===")
contributions = {"motion": 0.10, "context": 0.31, "physiology": 0.28}
text = explain(0.69, contributions)
print(text)
pos_context = text.index("context evidence")
pos_physiology = text.index("heart rate elevated")
pos_motion = text.index("motion evidence")
assert pos_context < pos_physiology < pos_motion, "should be ordered 0.31, 0.28, 0.10 descending"
print("PASS\n")

print("=== Case 2: displayed numbers match fusion contributions exactly (rounded) ===")
contributions = {"motion": 0.3123456, "context": 0.1, "physiology": 0.05}
text = explain(0.46, contributions)
print(text)
assert "0.31" in text  # 0.3123456 rounded to 2dp
assert "0.10" in text
assert "0.05" in text
assert "0.46" in text  # risk itself
print("PASS\n")

print("=== Case 3: no raw-score substitution -- function only ever sees contributions ===")
# explain() has no access to "scores" at all -- its signature is
# explain(risk, contributions). This test documents that guarantee:
# passing a contributions dict with values that differ from any
# hypothetical raw score must show exactly the contributions given.
contributions = {"motion": 0.05}  # e.g. a heavily-suppressed contribution
text = explain(0.05, contributions)
print(text)
assert "0.05" in text
assert "0.50" not in text  # a raw score of 0.5 must never leak through
print("PASS\n")

print("=== Case 4: empty contributions handled without crashing ===")
text = explain(0.0, {})
print(text)
assert "0.00" in text
print("PASS\n")

print("ALL EXPLAIN TESTS PASSED")