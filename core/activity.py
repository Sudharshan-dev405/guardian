"""
core/activity.py -- Guardian Activity Module core aggregation logic.

Aggregates ContextStream activity states before and after MotionStream-detected
impacts across UMAFall fall records.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from data.loader import _parse_umafall_name, iter_windows, map_activity, read_umafall
from streams.context import ContextStream
from streams.motion import MotionStream

STATES = (
    "ambulating",
    "stationary",
    "seated hand activity",
    "lying/immobile",
)

PRE_SEC = 2.5
POST_SEC = 2.5


@dataclass
class RecordAnalysisResult:
    """Analysis result for a single UMAFall record."""

    filename: str
    is_fall: bool
    status: str  # "included", "no impact", "skipped", "not fall"
    impact_time: float | None = None
    before_states: list[str] = field(default_factory=list)
    after_states: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class ActivityAggregateResult:
    """Aggregated activity analysis across all scanned fall records."""

    scanned_records: int
    records_with_impact: int
    records_without_impact: int
    records_skipped: int
    before_counts: Counter
    after_counts: Counter
    before_total: int
    after_total: int
    record_rows: pd.DataFrame


def list_fall_files(root: Path | str) -> list[Path]:
    """Find all UMAFall fall CSV recordings under root using activity mapping."""
    root = Path(root)
    if not root.exists():
        return []

    fall_paths = []
    for path in sorted(root.rglob("*.csv")):
        if path.name.lower() == "fall_timestamps.csv":
            continue

        _subject, activity, _trial = _parse_umafall_name(path.name)
        _state, is_fall = map_activity(activity)
        if is_fall:
            fall_paths.append(path)

    return fall_paths


def analyze_fall_record(
    record_path: Path | str,
    motion_stream: MotionStream,
    context_stream: ContextStream,
) -> RecordAnalysisResult:
    """Process a single fall record through MotionStream and ContextStream.

    1. Checks if the file is an actual fall record.
    2. Reads data via read_umafall, catching expected missing-wrist ValueError.
    3. Detects impact via MotionStream and reconstructs impact_time using
       internal_now = window['t'] + (len(acc) - 1) / fs.
    4. Classifies context windows in [impact_time - 2.5, impact_time) and
       [impact_time, impact_time + 2.5].
    """
    path = Path(record_path)
    _subject, activity, _trial = _parse_umafall_name(path.name)
    _state, is_fall = map_activity(activity)

    if not is_fall:
        return RecordAnalysisResult(
            filename=path.name,
            is_fall=False,
            status="not fall",
        )

    try:
        record = read_umafall(path)
    except ValueError as exc:
        # Expected loader error, e.g. missing SensorID=2 (wrist)
        return RecordAnalysisResult(
            filename=path.name,
            is_fall=True,
            status="skipped",
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return RecordAnalysisResult(
            filename=path.name,
            is_fall=True,
            status="skipped",
            error=str(exc),
        )

    windows = list(iter_windows(record))
    if not windows:
        return RecordAnalysisResult(
            filename=path.name,
            is_fall=True,
            status="no impact",
        )

    # MotionStream is stateful, so reset state before each recording.
    motion_stream.reset()

    impact_time = None

    # PASS 1 -- Find impact
    for window, _meta in windows:
        motion_stream.score(window)
        time_since_impact = getattr(motion_stream, "time_since_impact", None)

        if (
            impact_time is None
            and motion_stream.gate_open
            and time_since_impact is not None
        ):
            n_samples = len(window["acc"])
            fs = float(window.get("fs", 50))
            window_t = float(window["t"])

            # MotionStream internal 'now' corresponds to the last sample
            internal_now = window_t + (n_samples - 1) / fs
            impact_time = internal_now - float(time_since_impact)
            break

    if impact_time is None:
        return RecordAnalysisResult(
            filename=path.name,
            is_fall=True,
            status="no impact",
        )

    # PASS 2 -- Classify context windows around reconstructed impact time
    before_states: list[str] = []
    after_states: list[str] = []

    for window, _meta in windows:
        t = float(window["t"])

        if impact_time - PRE_SEC <= t < impact_time:
            context_stream.score(window)
            state = getattr(context_stream, "last_state", "unknown")
            if state in STATES:
                before_states.append(state)

        elif impact_time <= t <= impact_time + POST_SEC:
            context_stream.score(window)
            state = getattr(context_stream, "last_state", "unknown")
            if state in STATES:
                after_states.append(state)

    return RecordAnalysisResult(
        filename=path.name,
        is_fall=True,
        status="included",
        impact_time=float(impact_time),
        before_states=before_states,
        after_states=after_states,
    )


def aggregate_activity_dataset(
    dataset_root: Path | str,
    motion_model_path: Path | str,
    context_model_path: Path | str,
    fall_files: Sequence[Path | str] | None = None,
) -> ActivityAggregateResult:
    """Aggregate activity before and after impacts across all UMAFall fall records."""
    root = Path(dataset_root)

    if fall_files is None:
        fall_paths = list_fall_files(root)
    else:
        fall_paths = [Path(p) for p in fall_files]

    motion_stream = MotionStream.load(Path(motion_model_path))
    context_stream = ContextStream.load(Path(context_model_path))

    if getattr(motion_stream, "model", None) is not None and hasattr(
        motion_stream.model, "n_jobs"
    ):
        motion_stream.model.n_jobs = 1

    if getattr(context_stream, "model", None) is not None and hasattr(
        context_stream.model, "n_jobs"
    ):
        context_stream.model.n_jobs = 1

    before_counts = Counter()
    after_counts = Counter()

    records_with_impact = 0
    records_without_impact = 0
    records_skipped = 0
    rows: list[dict] = []

    for path in fall_paths:
        res = analyze_fall_record(path, motion_stream, context_stream)

        if res.status == "skipped":
            records_skipped += 1
            rows.append(
                {
                    "record": res.filename,
                    "impact_time": None,
                    "before_windows": 0,
                    "after_windows": 0,
                    "status": "skipped",
                    "error": res.error,
                }
            )
        elif res.status == "no impact":
            records_without_impact += 1
            rows.append(
                {
                    "record": res.filename,
                    "impact_time": None,
                    "before_windows": 0,
                    "after_windows": 0,
                    "status": "no impact",
                    "error": "",
                }
            )
        elif res.status == "included":
            records_with_impact += 1
            before_counts.update(res.before_states)
            after_counts.update(res.after_states)
            rows.append(
                {
                    "record": res.filename,
                    "impact_time": res.impact_time,
                    "before_windows": len(res.before_states),
                    "after_windows": len(res.after_states),
                    "status": "included",
                    "error": "",
                }
            )

    before_total = sum(before_counts.values())
    after_total = sum(after_counts.values())

    return ActivityAggregateResult(
        scanned_records=len(fall_paths),
        records_with_impact=records_with_impact,
        records_without_impact=records_without_impact,
        records_skipped=records_skipped,
        before_counts=before_counts,
        after_counts=after_counts,
        before_total=before_total,
        after_total=after_total,
        record_rows=pd.DataFrame(rows),
    )
