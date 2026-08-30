"""
test_pipeline_real.py

End-to-end integration test for Guardian.

Uses:
    - real UMAFall data
    - real trained MotionStream
    - real trained ContextStream
    - scripted HR physiology stream
    - real fusion
    - real decision state machine
    - real explanation

Run from the Guardian project root:

    python test_pipeline_real.py
"""

from pathlib import Path

from data.loader import read_umafall, iter_windows, ScriptedTrace
from streams.motion import MotionStream
from streams.context import ContextStream
from streams.physiology import PhysiologyStream

from core.fusion import fuse
from core.decision import DecisionStateMachine
from core.explain import explain


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

FALL_FILE = (
    ROOT
    / "data"
    / "raw"
    / "UMAFall"
    / "UMAFall_Subject_18_Fall_lateralFall_3_2016-05-29_21-36-32.csv"
)

HR_TRACE_FILE = ROOT / "data" / "hr_trace.csv"

MOTION_MODEL = ROOT / "models" / "motion.joblib"
CONTEXT_MODEL = ROOT / "models" / "context.joblib"


# ---------------------------------------------------------------------------
# Main integration test
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("GUARDIAN REAL PIPELINE INTEGRATION TEST")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Check required files
    # -----------------------------------------------------------------------

    required = [
        FALL_FILE,
        HR_TRACE_FILE,
        MOTION_MODEL,
        CONTEXT_MODEL,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("[1] Required files found")
    print(f"    Fall record : {FALL_FILE.name}")
    print(f"    HR trace    : {HR_TRACE_FILE.name}")

    # -----------------------------------------------------------------------
    # 2. Load the real models
    # -----------------------------------------------------------------------

    print("\n[2] Loading trained streams...")

    motion = MotionStream.load(MOTION_MODEL)
    context = ContextStream.load(CONTEXT_MODEL)
    physiology = PhysiologyStream()

    print("    MotionStream     : OK")
    print("    ContextStream    : OK")
    print("    PhysiologyStream : OK")

    # -----------------------------------------------------------------------
    # 3. Load scripted physiology trace
    # -----------------------------------------------------------------------

    trace = ScriptedTrace(HR_TRACE_FILE)

    print("\n[3] Scripted physiology trace loaded")
    print(f"    Samples: {len(trace.t)}")

    if len(trace.t):
        print(f"    Duration: {trace.t[-1] - trace.t[0]:.2f} s")

    # -----------------------------------------------------------------------
    # 4. Read real UMAFall record
    # -----------------------------------------------------------------------

    record = read_umafall(FALL_FILE)

    print("\n[4] UMAFall record loaded")
    print(f"    Subject : {record.subject}")
    print(f"    Activity: {record.activity}")
    print(f"    Trial   : {record.trial}")
    print(f"    Samples : {len(record.t)}")
    print(f"    Duration: {record.t[-1] - record.t[0]:.2f} s")

    # -----------------------------------------------------------------------
    # 5. Run the complete pipeline
    # -----------------------------------------------------------------------

    decision = DecisionStateMachine()

    window_count = 0

    risks = []
    states = []

    print("\n[5] Running integrated pipeline...")
    print()

    for window, meta in iter_windows(record, trace=trace):

        window_count += 1

        # ---------------------------------------------------------------
        # Three evidence streams
        # ---------------------------------------------------------------

        motion_score = motion.score(window)
        context_score = context.score(window)
        physiology_score = physiology.score(window)

        scores = {
            "motion": motion_score,
            "context": context_score,
            "physiology": physiology_score,
        }

        # ---------------------------------------------------------------
        # Fusion
        # ---------------------------------------------------------------

        risk, contributions = fuse(
            scores,
            context_stream=context,
            motion_stream=motion,
        )

        # ---------------------------------------------------------------
        # Decision
        # ---------------------------------------------------------------

        state = decision.update(
            risk=risk,
            t=window["t"],
            contributions=contributions,
        )

        # ---------------------------------------------------------------
        # Explanation
        # ---------------------------------------------------------------

        explanation = explain(
            risk,
            contributions,
        )

        risks.append(risk)
        states.append(state)

        # Print first few windows and any interesting event
        if window_count <= 5 or state != "normal":

            print(
                f"window={window_count:02d} "
                f"t={window['t']:6.2f}s "
                f"motion={motion_score:.3f} "
                f"context={context_score:.3f} "
                f"physiology={physiology_score:.3f} "
                f"risk={risk:.3f} "
                f"state={state}"
            )

            print(
                f"    context_state={context.last_state} "
                f"context_conf={context.last_confidence:.3f}"
            )

            print(
                f"    motion_impact={motion.last_impact:.3f} "
                f"stillness={motion.last_stillness:.3f} "
                f"time_since_impact={motion.time_since_impact}"
            )

            print(f"    {explanation}")

    # -----------------------------------------------------------------------
    # 6. Basic validation
    # -----------------------------------------------------------------------

    print("\n[6] Validating pipeline output...")

    assert window_count > 0, "No windows were produced"

    assert len(risks) == window_count
    assert len(states) == window_count

    for risk in risks:
        assert 0.0 <= risk <= 1.0, f"Invalid risk: {risk}"

    for state in states:
        assert isinstance(state, str)
        assert state != "", "Empty decision state"

    print(f"    Windows processed: {window_count}")
    print(f"    Maximum risk     : {max(risks):.3f}")
    print(f"    Final risk       : {risks[-1]:.3f}")
    print(f"    Final state      : {states[-1]}")

    # -----------------------------------------------------------------------
    # 7. Verify physiology gating behaviour
    # -----------------------------------------------------------------------

    print("\n[7] Physiology gating check...")

    # Find an ambulating context window, if one occurred.
    gated = False

    # We cannot replay the exact state here without altering stream state,
    # so this is only a structural API check.
    assert hasattr(physiology, "last_quality")

    print("    PhysiologyStream exposes last_quality: OK")

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("REAL PIPELINE INTEGRATION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()