"""
streams/motion.py -- Guardian motion stream: impact + post-impact stillness.

Two stages, per the design:
  Stage 1  cheap gate: SVM peak > 2.5 g OR gyro resultant > 200 deg/s.
  Stage 2  RandomForest over a segment spanning 2.0 s before to 1.5 s after
           the gate trigger, temperature-scaled to a calibrated 0-1.
  Post-impact stillness: forearm tilt from a low-passed gravity estimate plus
           motion variance over the following 10 s.

  score = 0.6 * impact + 0.4 * stillness

The wrist is a poor site for impact-based fall detection (Kangas 2008,
Bagala 2012). This module is one weighted input among four, not a trigger.
Nothing here is named "lying posture" -- the tilt measured is FOREARM tilt,
which moves independently of the trunk.

Because the scorer needs 1.5 s of future relative to the trigger, and windows
arrive every 1.24 s, this stream keeps its own sample ring buffer and latches
an event once the post-trigger samples have arrived. score() is still pure
per-window from the caller's point of view.

Exposed for core/explain.py after every score():
    last_quality, last_impact, last_stillness, gate_open, time_since_impact

CLI:
    py -m streams.motion --selftest
    py -m streams.motion --train data\\raw\\UMAFall --out models\\motion.joblib
    py -m streams.motion --tune  data\\raw\\UMAFall
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import Stream  # noqa: E402

# --------------------------------------------------------------------------
# Tunables -- all in one place, all expected to move once real data lands
# --------------------------------------------------------------------------

GATE_SVM_G = 2.5             # g, wrist literature; retune with --tune
GATE_GYRO_DPS = 1e9          # DISABLED after --tune on UMAFall: at the wrist,
                             # ADL angular rates exceed fall rates (ADL p90 443
                             # deg/s vs fall median 282), so no threshold
                             # separates. Gyro still feeds Stage 2 as features.

PRE_SEC = 2.0                # segment start, before the trigger
POST_SEC = 1.5               # segment end, after the trigger
STILL_SEC = 10.0             # stillness observation span after impact
MIN_STILL_SEC = 2.0          # before this much has elapsed, stillness is untrusted

EVENT_HOLD_SEC = 60.0        # how long an impact keeps contributing
EVENT_DECAY_SEC = 30.0       # exponential decay applied after the hold
REFRACTORY_SEC = 3.0         # ignore re-triggers inside one impact

W_IMPACT, W_STILL = 0.6, 0.4

STILL_VAR_G2 = 0.01          # SVM variance at or below this reads as still
HORIZONTAL_DEG = 60.0        # forearm within this of horizontal counts as "at rest low"
CLIP_G = 15.9                # sensor full scale; fraction clipped drives quality
BUFFER_SEC = 30.0
GRAVITY_LP_HZ = 0.5

FEATURE_NAMES = (
    "svm_max", "svm_min", "svm_range", "freefall_dip",
    "dip_to_peak_s", "n_above_2g", "jerk_max", "jerk_rms",
    "gyro_res_max", "gyro_integrated", "gyro_res_pre", "gyro_res_post",
    "svm_std_pre", "svm_std_post", "energy_ratio",
    "tilt_change_deg", "tilt_post_mean", "has_gyro",
)


# --------------------------------------------------------------------------
# Segment features
# --------------------------------------------------------------------------

def segment_features(acc: np.ndarray, gyro, fs: float, trig_i: int) -> np.ndarray:
    """Features over one -PRE_SEC..+POST_SEC segment. trig_i is the index of
    the gate trigger within the segment."""
    acc = np.asarray(acc, dtype=float)
    n = acc.shape[0]
    svm = np.linalg.norm(acc, axis=1)

    svm_max = float(svm.max())
    svm_min = float(svm.min())
    svm_range = svm_max - svm_min

    # Free fall: how far below 1 g the signal dips in the second before impact.
    lo = max(0, trig_i - int(1.0 * fs))
    pre_dip = float(svm[lo:trig_i + 1].min()) if trig_i > lo else 1.0
    freefall_dip = max(0.0, 1.0 - pre_dip)

    i_peak = int(np.argmax(svm))
    i_dip = int(np.argmin(svm[:i_peak + 1])) if i_peak > 0 else 0
    dip_to_peak_s = float((i_peak - i_dip) / fs)

    n_above_2g = float(np.sum(svm > 2.0))

    jerk = np.diff(acc, axis=0) * fs
    jm = np.linalg.norm(jerk, axis=1) if len(jerk) else np.zeros(1)
    jerk_max = float(jm.max())
    jerk_rms = float(np.sqrt(np.mean(jm ** 2)))

    pre = slice(0, max(1, trig_i))
    post = slice(min(trig_i + 1, n - 1), n)
    svm_std_pre = float(svm[pre].std())
    svm_std_post = float(svm[post].std())
    e_pre = float(np.mean((svm[pre] - 1.0) ** 2)) + 1e-9
    e_post = float(np.mean((svm[post] - 1.0) ** 2))
    energy_ratio = float(np.clip(e_post / e_pre, 0.0, 1e3))

    if gyro is None:
        gyro_res_max = gyro_integrated = gyro_res_pre = gyro_res_post = 0.0
        has_gyro = 0.0
    else:
        g = np.asarray(gyro, dtype=float)
        res = np.linalg.norm(g, axis=1)
        gyro_res_max = float(res.max())
        gyro_integrated = float(res.sum() / fs)
        gyro_res_pre = float(res[pre].mean())
        gyro_res_post = float(res[post].mean())
        has_gyro = 1.0

    tilt = forearm_tilt(acc, fs)
    tilt_pre = float(np.mean(tilt[pre])) if trig_i > 0 else float(tilt[0])
    tilt_post_mean = float(np.mean(tilt[post]))
    tilt_change_deg = abs(tilt_post_mean - tilt_pre)

    f = np.array([
        svm_max, svm_min, svm_range, freefall_dip,
        dip_to_peak_s, n_above_2g, jerk_max, jerk_rms,
        gyro_res_max, gyro_integrated, gyro_res_pre, gyro_res_post,
        svm_std_pre, svm_std_post, energy_ratio,
        tilt_change_deg, tilt_post_mean, has_gyro,
    ], dtype=np.float64)
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def forearm_tilt(acc: np.ndarray, fs: float) -> np.ndarray:
    """Angle in degrees between the low-passed gravity estimate and the
    sensor Z axis. 0 = axis vertical, 90 = axis horizontal.

    This is the forearm, not the trunk. A wrist device cannot observe trunk
    posture: without a magnetometer heading is unobservable, and the forearm
    moves independently of the body. Do not rename this.
    """
    g = _lowpass(np.asarray(acc, dtype=float), fs, GRAVITY_LP_HZ)
    n = np.linalg.norm(g, axis=1)
    n[n < 1e-6] = 1e-6
    return np.degrees(np.arccos(np.clip(g[:, 2] / n, -1.0, 1.0)))


def _lowpass(x, fs, corner):
    wn = min(corner / (fs / 2.0), 0.99)
    b, a = signal.butter(2, wn, btype="low")
    if x.shape[0] <= 3 * max(len(a), len(b)):
        return np.repeat(x.mean(axis=0, keepdims=True), x.shape[0], axis=0)
    return signal.filtfilt(b, a, x, axis=0)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(probs, y_idx):
    logits = np.log(np.clip(probs, 1e-8, 1.0))

    def nll(logT):
        p = _softmax(logits / float(np.exp(logT)))
        return float(-np.mean(np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12)))

    r = minimize_scalar(nll, bounds=(0.0, np.log(20.0)), method="bounded")
    t = float(np.exp(r.x))
    # Lower-bounded at 1.0 on purpose. Temperature scaling exists to SOFTEN an
    # overconfident model; random forest vote fractions are never
    # underconfident. On a well-separated holdout the NLL minimum runs off
    # toward T -> 0, which would sharpen an already-perfect model into
    # nonsense. T = 1.0 means "no correction needed", which is a real answer.
    return max(t, 1.0)


# --------------------------------------------------------------------------
# The stream
# --------------------------------------------------------------------------

class MotionStream(Stream):
    """Impact + post-impact stillness. Conforms to contract.Stream."""

    def __init__(self, gate_svm=GATE_SVM_G, gate_gyro=GATE_GYRO_DPS):
        super().__init__()
        self.gate_svm = gate_svm
        self.gate_gyro = gate_gyro
        self.model: RandomForestClassifier | None = None
        self.classes_: list = []
        self.temperature: float = 1.0

        self._buf_acc = deque()
        self._buf_gyro = deque()
        self._buf_t = deque()
        self._last_t: float | None = None
        self._fs = 50.0
        self._has_gyro = True

        self._trig_t: float | None = None      # wall time of the current trigger
        self._impact: float = 0.0              # latched calibrated impact score
        self._impact_done = False

        self.last_quality = 0.0
        self.last_impact = 0.0
        self.last_stillness = 0.0
        self.gate_open = False
        self.time_since_impact: float | None = None

    def reset(self):
        self._buf_acc.clear(); self._buf_gyro.clear(); self._buf_t.clear()
        self._last_t = None
        self._trig_t = None
        self._impact = 0.0
        self._impact_done = False
        self.last_impact = self.last_stillness = 0.0
        self.time_since_impact = None
        self.gate_open = False

    # -- buffering --------------------------------------------------------

    def _ingest(self, window):
        acc = np.asarray(window["acc"], dtype=float)
        gyro = window.get("gyro")
        fs = float(window.get("fs", 50))
        self._fs = fs
        t0 = float(window.get("t", 0.0))
        n = acc.shape[0]

        if self._last_t is None or t0 <= self._last_t or t0 - self._last_t > n / fs:
            take = n                       # first window, or a replay discontinuity
            if self._last_t is not None and t0 - self._last_t > n / fs:
                self.reset()
                self._fs = fs
        else:
            take = int(round((t0 - self._last_t) * fs))
            take = int(np.clip(take, 1, n))

        self._has_gyro = gyro is not None
        gy = np.asarray(gyro, dtype=float) if gyro is not None else None
        for i in range(n - take, n):
            self._buf_acc.append(acc[i])
            self._buf_gyro.append(gy[i] if gy is not None else np.zeros(3))
            self._buf_t.append(t0 + i / fs)
        self._last_t = t0

        cap = int(BUFFER_SEC * fs)
        while len(self._buf_t) > cap:
            self._buf_acc.popleft(); self._buf_gyro.popleft(); self._buf_t.popleft()

    def _arrays(self):
        return (np.asarray(self._buf_acc), np.asarray(self._buf_gyro),
                np.asarray(self._buf_t))

    # -- stage 1 ----------------------------------------------------------

    def _check_gate(self, acc, gyro, t):
        """Return the buffer index of a new trigger, or None."""
        if self._trig_t is not None and (t[-1] - self._trig_t) < REFRACTORY_SEC:
            return None
        svm = np.linalg.norm(acc, axis=1)
        hit_a = svm > self.gate_svm
        if self._has_gyro:
            hit_g = np.linalg.norm(gyro, axis=1) > self.gate_gyro
        else:
            hit_g = np.zeros_like(hit_a)
        hits = np.flatnonzero(hit_a | hit_g)
        if len(hits) == 0:
            return None
        # Only consider triggers newer than the last one we handled.
        if self._trig_t is not None:
            hits = hits[t[hits] > self._trig_t + REFRACTORY_SEC]
            if len(hits) == 0:
                return None
        return int(hits[-1])

    # -- stage 2 ----------------------------------------------------------

    def _score_impact(self, acc, gyro, t, i_trig):
        pre_n = int(PRE_SEC * self._fs)
        post_n = int(POST_SEC * self._fs)
        lo = max(0, i_trig - pre_n)
        hi = i_trig + post_n + 1
        if hi > len(t):
            return None                     # post-trigger samples not here yet
        seg_a = acc[lo:hi]
        seg_g = gyro[lo:hi] if self._has_gyro else None
        f = segment_features(seg_a, seg_g, self._fs, i_trig - lo).reshape(1, -1)
        if self.model is None:
            # Untrained fallback so the pipeline still moves: a bounded,
            # monotone function of peak SVM. Not calibrated, and it says so
            # through last_quality.
            pk = float(np.linalg.norm(seg_a, axis=1).max())
            return float(np.clip((pk - self.gate_svm) / 4.0, 0.0, 1.0)), False
        p = self.model.predict_proba(f)
        p = _softmax(np.log(np.clip(p, 1e-8, 1.0)) / self.temperature)[0]
        fall_i = list(self.classes_).index(1) if 1 in list(self.classes_) else -1
        return float(p[fall_i]) if fall_i >= 0 else 0.0, True

    # -- stillness --------------------------------------------------------

    def _stillness(self, acc, t):
        if self._trig_t is None:
            return 0.0, 0.0
        m = t > self._trig_t
        if m.sum() < 4:
            return 0.0, 0.0
        elapsed = float(t[-1] - self._trig_t)
        seg = acc[m][:int(STILL_SEC * self._fs)]
        svm = np.linalg.norm(seg, axis=1)
        var_score = float(np.exp(-svm.var() / STILL_VAR_G2))     # 1 = perfectly still
        tilt = forearm_tilt(seg, self._fs)
        low_frac = float(np.mean(tilt > HORIZONTAL_DEG))         # forearm near-horizontal
        still = 0.6 * var_score + 0.4 * low_frac
        # Trust ramps with how much of the 10 s we have actually seen.
        conf = float(np.clip((elapsed - MIN_STILL_SEC) /
                             (STILL_SEC - MIN_STILL_SEC), 0.0, 1.0))
        return float(np.clip(still, 0.0, 1.0)), conf

    # -- contract ---------------------------------------------------------

    def score(self, window: dict) -> float:
        try:
            acc_in = window.get("acc")
            if acc_in is None or len(acc_in) == 0:
                self.last_quality = 0.0
                self.last_impact = self.last_stillness = 0.0
                self.gate_open = False
                return 0.0

            self._ingest(window)
            acc, gyro, t = self._arrays()
            now = float(t[-1])

            i_trig = self._check_gate(acc, gyro, t)
            if i_trig is not None:
                self._trig_t = float(t[i_trig])
                self._impact = 0.0
                self._impact_done = False
                self.gate_open = True

            calibrated = self.model is not None
            if self._trig_t is not None and not self._impact_done:
                i_now = int(np.searchsorted(t, self._trig_t))
                out = self._score_impact(acc, gyro, t, i_now)
                if out is not None:
                    self._impact, calibrated = out
                    self._impact_done = True

            if self._trig_t is None:
                self.last_impact = self.last_stillness = 0.0
                self.time_since_impact = None
                self.gate_open = False
                self.last_quality = self._quality(acc, calibrated, 1.0)
                return 0.0

            age = now - self._trig_t
            self.time_since_impact = age
            if age > EVENT_HOLD_SEC:
                decay = float(np.exp(-(age - EVENT_HOLD_SEC) / EVENT_DECAY_SEC))
                if decay < 0.02:
                    self.reset()
                    self.last_quality = self._quality(acc, calibrated, 1.0)
                    return 0.0
            else:
                decay = 1.0

            still, still_conf = self._stillness(acc, t)
            self.last_impact = float(self._impact)
            self.last_stillness = float(still)

            raw = W_IMPACT * self._impact + W_STILL * still
            out = float(np.clip(raw * decay, 0.0, 1.0))
            self.last_quality = self._quality(acc, calibrated,
                                              0.5 + 0.5 * still_conf)
            return out

        except Exception as e:                              # noqa: BLE001
            print(f"[motion] score failed: {e}", file=sys.stderr)
            self.last_quality = 0.0
            self.last_impact = self.last_stillness = 0.0
            return 0.0

    def _quality(self, acc, calibrated, completeness):
        """Quality falls for: an uncalibrated model, a missing gyroscope, a
        clipped accelerometer, or an event we have not fully observed."""
        q = 1.0
        if not calibrated:
            q *= 0.5
        if not self._has_gyro:
            q *= 0.7          # Stage 1 loses the rotational half of the gate
        svm = np.linalg.norm(acc[-int(2.5 * self._fs):], axis=1)
        clipped = float(np.mean(svm >= CLIP_G))
        q *= float(np.clip(1.0 - 3.0 * clipped, 0.0, 1.0))
        q *= float(np.clip(completeness, 0.0, 1.0))
        if len(self._buf_t) < int(PRE_SEC * self._fs):
            q *= 0.5          # buffer not yet primed at replay start
        return float(np.clip(q, 0.0, 1.0))

    # -- training ---------------------------------------------------------

    def fit_segments(self, X, y, groups=None, seed=0, n_estimators=400,
                     holdout_frac=0.25):
        X = np.asarray(X); y = np.asarray(y)
        g = np.asarray(groups) if groups is not None else np.arange(len(y))
        if len(np.unique(y)) < 2:
            raise ValueError("need both fall and ADL segments to train")
        tr, ho = next(GroupShuffleSplit(n_splits=1, test_size=holdout_frac,
                                        random_state=seed).split(X, y, groups=g))
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        self.model.fit(X[tr], y[tr])
        self.classes_ = list(self.model.classes_)
        p = self.model.predict_proba(X[ho])
        y_i = np.asarray([self.classes_.index(v) for v in y[ho]])
        self.temperature = fit_temperature(p, y_i)
        pc = _softmax(np.log(np.clip(p, 1e-8, 1.0)) / self.temperature)
        self._holdout = (y[ho], pc[:, self.classes_.index(1)])
        return self

    def save(self, path):
        import joblib
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "classes": self.classes_,
                     "temperature": self.temperature,
                     "gate_svm": self.gate_svm, "gate_gyro": self.gate_gyro}, path)
        return path

    @classmethod
    def load(cls, path):
        import joblib
        d = joblib.load(path)
        s = cls(gate_svm=d["gate_svm"], gate_gyro=d["gate_gyro"])
        s.model = d["model"]; s.classes_ = d["classes"]; s.temperature = d["temperature"]
        return s


# --------------------------------------------------------------------------
# Dataset -> training segments
# --------------------------------------------------------------------------

def segments_from_record(rec, gate_svm=GATE_SVM_G, gate_gyro=GATE_GYRO_DPS):
    """One segment per record, centred on its strongest gate trigger. If the
    gate never fires (common for quiet ADLs), centre on the peak SVM anyway
    and keep it as a hard negative -- otherwise the classifier only ever sees
    the ADLs that already look violent."""
    fs = 50.0
    acc = np.asarray(rec.acc, dtype=float)
    gyro = np.asarray(rec.gyro, dtype=float) if rec.gyro is not None else None
    n = len(acc)
    pre_n, post_n = int(PRE_SEC * fs), int(POST_SEC * fs)
    if n < pre_n + post_n + 2:
        return []
    svm = np.linalg.norm(acc, axis=1)
    hit = svm > gate_svm
    if gyro is not None:
        hit = hit | (np.linalg.norm(gyro, axis=1) > gate_gyro)
    # Use the FIRST gate crossing, not the global peak. The live stream has no
    # oracle: _check_gate fires on a threshold crossing and takes that instant
    # as the trigger. Training on a perfectly-centred impact and deploying on
    # an approximately-centred one is a train/deploy mismatch that flatters the
    # offline number. Ungated records fall back to their peak as hard negatives.
    idx = int(np.flatnonzero(hit)[0]) if hit.any() else int(np.argmax(svm))
    i = int(np.clip(idx, pre_n, n - post_n - 1))
    seg_a = acc[i - pre_n:i + post_n + 1]
    seg_g = gyro[i - pre_n:i + post_n + 1] if gyro is not None else None
    f = segment_features(seg_a, seg_g, fs, pre_n)
    return [(f, int(rec.is_fall), rec.subject, bool(hit.any()))]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _collect(root):
    from data.loader import scan                     # lazy: CLI only
    X, y, g, gated = [], [], [], []
    for rec in scan(root):
        for f, lab, subj, hit in segments_from_record(rec):
            X.append(f); y.append(lab); g.append(subj); gated.append(hit)
    return np.asarray(X), np.asarray(y), np.asarray(g), np.asarray(gated)


def _train(root, out):
    from sklearn.model_selection import LeaveOneGroupOut

    X, y, g, gated = _collect(root)
    if len(X) == 0:
        print("No segments. Run: py -m data.loader --scan <dir>")
        return
    subjects = sorted(set(g.tolist()))
    print(f"{len(X)} segments  falls={int(y.sum())}  ADLs={int((1-y).sum())}  "
          f"subjects={len(subjects)}")
    print(f"stage-1 gate fires on {gated.mean()*100:.1f}% of records "
          f"({gated[y==1].mean()*100:.1f}% of falls, {gated[y==0].mean()*100:.1f}% of ADLs)")

    # Leave-one-subject-out. A random or few-subject split on 617 stereotyped
    # staged records gives an AUC near 1.0 that no panel will believe. LOSO
    # over every subject is the number to report.
    oof = np.full(len(y), np.nan)
    fold_auc = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups=g):
        if len(np.unique(y[tr])) < 2:
            continue
        m = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                   class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        fi = list(m.classes_).index(1)
        p = m.predict_proba(X[te])[:, fi]
        oof[te] = p
        if len(np.unique(y[te])) == 2:
            fold_auc.append(roc_auc_score(y[te], p))

    print(f"\n--- leave-one-subject-out, {len(fold_auc)} evaluable folds ---")
    if fold_auc:
        a = np.asarray(fold_auc)
        print(f"per-subject AUC  mean {a.mean():.3f}  sd {a.std():.3f}  "
              f"min {a.min():.3f}  max {a.max():.3f}")

    def _roc_report(mask, label):
        m = mask & ~np.isnan(oof)
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            print(f"{label}: too few samples to score")
            return
        fpr, tpr, thr = roc_curve(y[m], oof[m])
        auc = roc_auc_score(y[m], oof[m])
        k95, k90 = fpr <= 0.05, fpr <= 0.10
        s95 = float(tpr[k95].max()) if k95.any() else 0.0
        t95 = float(thr[k95][int(np.argmax(tpr[k95]))]) if k95.any() else float("nan")
        s90 = float(tpr[k90].max()) if k90.any() else 0.0
        print(f"{label}  n={int(m.sum())} "
              f"(falls {int(y[m].sum())}, ADLs {int((1-y[m]).sum())})")
        print(f"   AUC {auc:.3f}   sens@95%spec {s95:.3f} (thr {t95:.2f})   "
              f"sens@90%spec {s90:.3f}")

    print()
    _roc_report(np.ones(len(y), bool), "ALL RECORDS      ")
    _roc_report(gated, "GATE-CONDITIONAL ")
    print("   ^ this is the number to report. Stage 2 only ever runs on segments")
    print("     Stage 1 gated, so the ungated ADLs are negatives the scorer never")
    print("     sees in deployment. Including them inflates the AUC.")

    # Final model on everything, calibrated on a subject-disjoint holdout.
    ms = MotionStream().fit_segments(X, y, g)
    imp = sorted(zip(FEATURE_NAMES, ms.model.feature_importances_),
                 key=lambda kv: -kv[1])[:6]
    print("\ntop features: " + ", ".join(f"{k} {v:.3f}" for k, v in imp))
    print(f"temperature T = {ms.temperature:.3f}")
    print(f"saved -> {ms.save(out)}")


def _tune(root):
    """The 2.5 g / 200 deg/s thresholds come from higher-rate wrist studies.
    Show what they actually do on this data before trusting them."""
    from data.loader import scan
    peaks_f, peaks_a, gyro_f, gyro_a = [], [], [], []
    for rec in scan(root):
        svm = np.linalg.norm(np.asarray(rec.acc, dtype=float), axis=1)
        gr = (np.linalg.norm(np.asarray(rec.gyro, dtype=float), axis=1).max()
              if rec.gyro is not None else 0.0)
        (peaks_f if rec.is_fall else peaks_a).append(float(svm.max()))
        (gyro_f if rec.is_fall else gyro_a).append(float(gr))
    if not peaks_f or not peaks_a:
        print("need both falls and ADLs"); return
    for name, f, a, thr, unit in (("SVM peak", peaks_f, peaks_a, GATE_SVM_G, "g"),
                                  ("gyro peak", gyro_f, gyro_a, GATE_GYRO_DPS, "deg/s")):
        f, a = np.asarray(f), np.asarray(a)
        print(f"\n{name}: falls median {np.median(f):.2f} {unit} "
              f"(p10 {np.percentile(f,10):.2f}), ADLs median {np.median(a):.2f} "
              f"(p90 {np.percentile(a,90):.2f})")
        print(f"  current threshold {thr} {unit}: catches {np.mean(f>thr)*100:.0f}% of falls, "
              f"fires on {np.mean(a>thr)*100:.0f}% of ADLs")
        for q in (0.05, 0.10, 0.20):
            t = float(np.quantile(f, q))
            print(f"  at {t:.2f} {unit} (p{int(q*100)} of falls): "
                  f"{np.mean(f>t)*100:.0f}% falls, {np.mean(a>t)*100:.0f}% ADLs")


def _selftest():
    """Synthetic fall and ADL fixtures pushed through as a window stream.
    Machinery test only; never used as project data."""
    fs, rng = 50, np.random.default_rng(0)

    def stream(kind, secs=25.0):
        n = int(secs * fs)
        t = np.arange(n) / fs
        acc = np.column_stack([rng.normal(0, .05, n), rng.normal(0, .05, n),
                               1 + rng.normal(0, .05, n)])
        gyro = rng.normal(0, 10, (n, 3))
        k = int(8.0 * fs)
        if kind == "fall":
            acc[k-25:k, 2] = 0.15                       # free fall
            acc[k:k+6, :] += np.array([4.0, 3.0, 5.0])  # impact
            gyro[k-10:k+10] += 400
            acc[k+10:, :] = np.array([0.98, 0.05, 0.08]) # forearm horizontal, still
            acc[k+10:, :] += rng.normal(0, .01, (n-k-10, 3))
            gyro[k+10:] = rng.normal(0, 1, (n-k-10, 3))
        elif kind == "hand":                            # hard desk slam, then activity
            acc[k:k+4, :] += np.array([3.0, 1.0, 2.5])
            gyro[k-5:k+5] += 250
            acc[k+10:, 0] += 0.5*np.sin(2*np.pi*0.9*t[k+10:])
            gyro[k+10:] += rng.normal(0, 80, (n-k-10, 3))
        for s in range(0, n - 125 + 1, 62):
            yield {"t": s/fs, "acc": acc[s:s+125].astype(np.float32),
                   "gyro": gyro[s:s+125].astype(np.float32),
                   "hr": None, "temp": None, "fs": fs}

    for kind in ("fall", "hand"):
        ms = MotionStream()
        peak, at_t, fired = 0.0, 0.0, False
        for w in stream(kind):
            s = ms.score(w)
            assert isinstance(s, float) and 0.0 <= s <= 1.0 and not np.isnan(s)
            assert 0.0 <= ms.last_quality <= 1.0
            if ms.gate_open:
                fired = True
            if s > peak:
                peak, at_t = s, w["t"]
        print(f"{kind:5s}: gate fired={fired}  peak score={peak:.2f} at t={at_t:.1f}s  "
              f"impact={ms.last_impact:.2f} stillness={ms.last_stillness:.2f} "
              f"q={ms.last_quality:.2f}")

    # A quiet stream must never trigger.
    ms = MotionStream()
    quiet = 0.0
    for w in stream("quiet"):
        quiet = max(quiet, ms.score(w))
    print(f"quiet: peak score={quiet:.2f} (must be 0.00)")
    assert quiet == 0.0

    # Contract edges.
    ms = MotionStream()
    assert ms.score({"acc": None, "fs": 50}) == 0.0 and ms.last_quality == 0.0
    assert ms.score({}) == 0.0
    ms2 = MotionStream()
    w = {"t": 0.0, "acc": np.ones((125, 3), np.float32)/np.sqrt(3),
         "gyro": None, "hr": None, "temp": None, "fs": 50}
    assert 0.0 <= ms2.score(w) <= 1.0
    print("no-gyro window handled, quality =", round(ms2.last_quality, 2))
    print("SELFTEST PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Guardian motion stream")
    ap.add_argument("--train", metavar="DIR")
    ap.add_argument("--out", default="models/motion.joblib")
    ap.add_argument("--tune", metavar="DIR")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.train:
        _train(a.train, a.out)
    elif a.tune:
        _tune(a.tune)
    elif a.selftest:
        _selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
