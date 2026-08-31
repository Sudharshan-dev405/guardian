from pathlib import Path

from data.loader import ScriptedTrace
from streams.physiology import PhysiologyStream


ROOT = Path(__file__).resolve().parent
TRACE_FILE = ROOT / "data" / "hr_trace.csv"


def main():

    print("=" * 70)
    print("GUARDIAN PHYSIOLOGY TRACE TEST")
    print("=" * 70)

    print(f"\nTrace file: {TRACE_FILE}")

    if not TRACE_FILE.exists():
        raise FileNotFoundError(TRACE_FILE)

    trace = ScriptedTrace(TRACE_FILE)

    print(f"Samples: {len(trace.t)}")

    if len(trace.t):
        print(f"Start: {trace.t[0]:.2f} s")
        print(f"End:   {trace.t[-1]:.2f} s")

    physiology = PhysiologyStream()

    print("\nTime      HR        Physiology score")
    print("-" * 45)

    for t in [0, 10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100]:

        hr, temp = trace.at(t)

        window = {
            "t": float(t),
            "hr": hr,
            "temp": temp,
            "context_state": "seated hand activity",
        }

        score = physiology.score(window)

        print(
            f"{t:5.1f}s    "
            f"{str(hr):>6}    "
            f"{score:.3f}"
        )


if __name__ == "__main__":
    main()