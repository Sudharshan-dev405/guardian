# test_stubs.py -- confirms all four stubs satisfy the Stream contract
from streams.context import ContextStream
from streams.motion import MotionStream
from streams.physiology import PhysiologyStream
from streams.quality import QualityStream

STREAMS = {
    "context": ContextStream(),
    "motion": MotionStream(),
    "physiology": PhysiologyStream(),
    "quality": QualityStream(),
}

dummy_window = {
    "t": 0.0,
    "acc": None,
    "gyro": None,
    "hr": None,
    "temp": None,
    "fs": 50,
}

all_ok = True
seen_scores = set()

for name, stream in STREAMS.items():
    score = stream.score(dummy_window)
    quality = stream.last_quality

    ok = True
    if not isinstance(score, float):
        print(f"FAIL [{name}]: score is not a float ({type(score)})")
        ok = False
    if not (0.0 <= score <= 1.0):
        print(f"FAIL [{name}]: score {score} out of [0,1]")
        ok = False
    if not isinstance(quality, float):
        print(f"FAIL [{name}]: last_quality is not a float ({type(quality)})")
        ok = False
    if not (0.0 <= quality <= 1.0):
        print(f"FAIL [{name}]: last_quality {quality} out of [0,1]")
        ok = False

    if ok:
        print(f"PASS [{name}]: score={score}, last_quality={quality}")
    else:
        all_ok = False

    seen_scores.add(score)

# Sanity check: stubs should have DIFFERENT constants, so we can
# visually distinguish them on the dashboard later.
if len(seen_scores) < len(STREAMS):
    print("WARN: two or more stubs share the same score constant -- "
          "you won't be able to tell them apart on the dashboard.")
else:
    print("PASS: all stub scores are distinct.")

print("\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED")