"""
check_activity.py -- Terminal validation for Guardian Activity Module.

Runs ContextStream activity classification before and after MotionStream-detected
impacts across all UMAFall fall records and reports the final aggregate distribution.
"""

from __future__ import annotations

from pathlib import Path

from core.activity import (
    POST_SEC,
    PRE_SEC,
    STATES,
    aggregate_activity_dataset,
    analyze_fall_record,
    list_fall_files,
)
from streams.context import ContextStream
from streams.motion import MotionStream

# ======================================================================
# CONFIGURATION
# ======================================================================

ROOT = Path("data/raw/UMAFall")
CONTEXT_MODEL = Path("models/context.joblib")
MOTION_MODEL = Path("models/motion.joblib")

DEBUG_RECORDS = 5


# ======================================================================
# VALIDATION
# ======================================================================

if not ROOT.exists():
    raise FileNotFoundError(f"UMAFall directory not found: {ROOT}")

if not CONTEXT_MODEL.exists():
    raise FileNotFoundError(f"Context model not found: {CONTEXT_MODEL}")

if not MOTION_MODEL.exists():
    raise FileNotFoundError(f"Motion model not found: {MOTION_MODEL}")


# ======================================================================
# OPTIONAL DEBUG: TIMING CHECK FOR FIRST FEW IMPACTS
# ======================================================================

fall_files = list_fall_files(ROOT)
motion_stream = MotionStream.load(MOTION_MODEL)
context_stream = ContextStream.load(CONTEXT_MODEL)

if getattr(motion_stream, "model", None) is not None and hasattr(
    motion_stream.model, "n_jobs"
):
    motion_stream.model.n_jobs = 1

if getattr(context_stream, "model", None) is not None and hasattr(
    context_stream.model, "n_jobs"
):
    context_stream.model.n_jobs = 1

debug_count = 0
for path in fall_files:
    if debug_count >= DEBUG_RECORDS:
        break

    res = analyze_fall_record(path, motion_stream, context_stream)
    if res.status == "included" and res.impact_time is not None:
        print()
        print("-" * 70)
        print("IMPACT TIMING CHECK")
        print("-" * 70)
        print(f"File:                  {res.filename}")
        print(f"Reconstructed impact:  {res.impact_time:.3f} s")
        print(f"Before windows:        {len(res.before_states)}")
        print(f"After windows:         {len(res.after_states)}")
        debug_count += 1


# ======================================================================
# RUN DATASET AGGREGATION
# ======================================================================

result = aggregate_activity_dataset(
    dataset_root=ROOT,
    motion_model_path=MOTION_MODEL,
    context_model_path=CONTEXT_MODEL,
    fall_files=fall_files,
)


# ======================================================================
# PRINT DISTRIBUTION
# ======================================================================

def print_distribution(title: str, counter: dict[str, int]) -> None:
    total = sum(counter.values())

    print()
    print(title)
    print("-" * len(title))

    for state in STATES:
        count = counter.get(state, 0)
        percentage = 100.0 * count / total if total > 0 else 0.0
        print(f"{state:25s}{count:6d}{percentage:7.1f}%")

    print(f"{'TOTAL':25s}{total:6d}{100.0 if total > 0 else 0.0:7.1f}%")


# ======================================================================
# FINAL OUTPUT
# ======================================================================

print()
print("=" * 70)
print("FINAL AGGREGATE")
print("=" * 70)

print(f"Fall records scanned:   {result.scanned_records}")
print(f"Records with impact:    {result.records_with_impact}")
print(f"Records without impact: {result.records_without_impact}")
print(f"Records skipped:        {result.records_skipped}")

print_distribution(
    f"BEFORE IMPACT ({PRE_SEC:.1f} s)",
    result.before_counts,
)

print_distribution(
    f"AFTER IMPACT ({POST_SEC:.1f} s)",
    result.after_counts,
)

print()
print("=" * 70)