"""
streams/context.py -- Guardian four-state activity context stream.

States: stationary, ambulating, seated hand activity, lying/immobile,
plus "unknown" from the open-set reject.

Pipeline: 15 cheap per-window features -> RandomForestClassifier ->
temperature scaling on a held-out split -> open-set reject on calibrated
max probability.

What fusion consumes:
    score(window)        -> float in [0,1], context risk contribution
    self.last_quality    -> float in [0,1], calibrated confidence (0 if unknown)
    self.last_state      -> one of STATES or "unknown"
    self.last_confidence -> calibrated max class probability

The STATE LABEL matters more than the score: core/fusion.py uses
last_state == "seated hand activity" with last_confidence > 0.7 to halve
the motion contribution. Both attributes are set on every score() call.

CLI (run from the repo root):
    py -m streams.context --selftest
    py -m streams.context --train data\\raw\\UMAFall --out models\\context.joblib
    py -m streams.context --report models\\context.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.optimize import minimize_scalar
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract import Stream  # noqa: E402

# --------------------------------------------------------------------------
# Tunables -- all in one place
# --------------------------------------------------------------------------

STATES = ("stationary", "ambulating", "seated hand activity", "lying/immobile")
UNKNOWN = "unknown"

# score() output per state. Risk contribution, not a probability.
# lying/immobile high, stationary mid, ambulating low, hand activity lowest.
STATE_RISK = {
    "lying/immobile":       0.85,
    "stationary":           0.40,
    "ambulating":           0.15,
    "seated hand activity": 0.05,
    UNKNOWN:                0.00,
}

# Gait band. Young-adult default; the elderly widening is the second entry.
# Ustad/HAR70+ and the elderly-gait finding: widening 1.4-2.3 -> 0.8-2.8 Hz
# and dropping the amplitude floor 0.3 -> 0.1 g raised walking sensitivity
# 0.11 -> 0.73 at the cost of specificity 0.99 -> 0.75. Off by default.
GAIT_BAND = (1.4, 2.3)
GAIT_BAND_ELDERLY = (0.8, 2.8)

GRAVITY_LP_HZ = 0.5      # low-pass corner isolating the gravity vector
DEFAULT_REJECT_Q = 0.05      # reject threshold = this quantile of held-out
                             # calibrated confidence on correct predictions
REJECT_THRESHOLD_MAX = 0.85  # cap, so a degenerate fold cannot reject everything

FEATURE_NAMES = (
    "svm_mean", "svm_var", "svm_max", "svm_iqr",
    "dom_freq", "spec_entropy", "band_power_gait",
    "autocorr_peak", "cadence_hz",
    "gyro_orient_change", "gyro_res_mean",
    "jerk_rms", "tilt_mean", "tilt_std", "has_gyro",
)


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def extract_features(window: dict, gait_band=GAIT_BAND) -> np.ndarray:
    """15 features from one window dict. Cheap enough to run per window."""
    acc = np.asarray(window["acc"], dtype=float)
    gyro = window.get("gyro")
    fs = float(window.get("fs", 50))
    n = acc.shape[0]

    svm = np.linalg.norm(acc, axis=1)
    svm_ac = svm - svm.mean()

    svm_mean = float(svm.mean())
    svm_var = float(svm.var())
    svm_max = float(svm.max())
    svm_iqr = float(np.subtract(*np.percentile(svm, [75, 25])))

    # Spectrum of the AC part of SVM.
    freqs, psd = signal.welch(svm_ac, fs=fs, nperseg=min(n, 128),
                              noverlap=min(n, 128) // 2)
    if psd.sum() <= 0 or len(freqs) < 3:
        dom_freq = 0.0
        spec_entropy = 0.0
        band_power = 0.0
    else:
        keep = freqs > 0.3                      # drop DC and postural drift
        f_k, p_k = freqs[keep], psd[keep]
        if len(p_k) == 0 or p_k.sum() <= 0:
            dom_freq, spec_entropy, band_power = 0.0, 0.0, 0.0
        else:
            dom_freq = float(f_k[int(np.argmax(p_k))])
            p_norm = p_k / p_k.sum()
            spec_entropy = float(-np.sum(p_norm * np.log(p_norm + 1e-12))
                                 / np.log(len(p_norm)))
            in_band = (f_k >= gait_band[0]) & (f_k <= gait_band[1])
            band_power = float(p_k[in_band].sum() / p_k.sum())

    # Autocorrelation peak -> cadence. Search 0.5-3.0 Hz.
    autocorr_peak, cadence_hz = _autocorr_cadence(svm_ac, fs)

    # Jerk.
    jerk = np.diff(acc, axis=0) * fs
    jerk_rms = float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1)))) if len(jerk) else 0.0

    # Gravity-referenced tilt. Low-pass the accelerometer, take the angle
    # between the gravity estimate and the sensor's own vertical axis.
    # This is FOREARM tilt. It is not trunk posture and is never called that.
    grav = _lowpass(acc, fs, GRAVITY_LP_HZ)
    gn = np.linalg.norm(grav, axis=1)
    gn[gn < 1e-6] = 1e-6
    cos_t = np.clip(grav[:, 2] / gn, -1.0, 1.0)
    tilt = np.degrees(np.arccos(cos_t))
    tilt_mean = float(tilt.mean())
    tilt_std = float(tilt.std())

    if gyro is None:
        gyro_orient_change = 0.0
        gyro_res_mean = 0.0
        has_gyro = 0.0
    else:
        g = np.asarray(gyro, dtype=float)
        res = np.linalg.norm(g, axis=1)
        gyro_orient_change = float(np.sum(res) / fs)    # degrees swept
        gyro_res_mean = float(res.mean())
        has_gyro = 1.0

    feats = np.array([
        svm_mean, svm_var, svm_max, svm_iqr,
        dom_freq, spec_entropy, band_power,
        autocorr_peak, cadence_hz,
        gyro_orient_change, gyro_res_mean,
        jerk_rms, tilt_mean, tilt_std, has_gyro,
    ], dtype=np.float64)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def _autocorr_cadence(x: np.ndarray, fs: float, lo_hz=0.5, hi_hz=3.0):
    n = len(x)
    if n < 8 or x.std() < 1e-9:
        return 0.0, 0.0
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / (ac[0] + 1e-12)
    lag_lo = max(1, int(fs / hi_hz))
    lag_hi = min(n - 1, int(fs / lo_hz))
    if lag_hi <= lag_lo:
        return 0.0, 0.0
    seg = ac[lag_lo:lag_hi]
    i = int(np.argmax(seg))
    peak = float(seg[i])
    lag = lag_lo + i
    return max(peak, 0.0), float(fs / lag)


def _lowpass(x: np.ndarray, fs: float, corner: float):
    nyq = fs / 2.0
    wn = min(corner / nyq, 0.99)
    b, a = signal.butter(2, wn, btype="low")
    pad = 3 * max(len(a), len(b))
    if x.shape[0] <= pad:
        return np.repeat(x.mean(axis=0, keepdims=True), x.shape[0], axis=0)
    return signal.filtfilt(b, a, x, axis=0)


# --------------------------------------------------------------------------
# Temperature scaling (Guo et al., ICML 2017)
# --------------------------------------------------------------------------

def _probs_to_logits(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-8, 1.0))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y_idx: np.ndarray) -> float:
    """Single scalar T minimising held-out NLL. T > 1 softens (RF vote
    fractions are overconfident, so expect T > 1)."""
    logits = _probs_to_logits(probs)

    def nll(logT):
        T = float(np.exp(logT))
        p = _softmax(logits / T)
        return float(-np.mean(np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12)))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray, bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(conf)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# --------------------------------------------------------------------------
# The stream
# --------------------------------------------------------------------------

class ContextStream(Stream):
    """Four-state context classifier. Conforms to contract.Stream."""

    def __init__(self, gait_band=GAIT_BAND, reject_threshold: float | None = None,
                 elderly_gait: bool = False):
        super().__init__()
        self.gait_band = GAIT_BAND_ELDERLY if elderly_gait else gait_band
        self.model: RandomForestClassifier | None = None
        self.classes_: list[str] = []
        self.temperature: float = 1.0
        self.reject_threshold: float = 0.0 if reject_threshold is None else reject_threshold
        self.ece_before: float | None = None
        self.ece_after: float | None = None

        self.last_state: str = UNKNOWN
        self.last_confidence: float = 0.0
        self.last_quality: float = 0.0
        self.last_probs: dict = {}

    # -- training ---------------------------------------------------------

    def fit(self, df_normal):
        """Contract-compatible entry point. Accepts either a list of
        (window, meta) pairs or a DataFrame-like with 'window'/'state'."""
        pairs = list(df_normal)
        if pairs and isinstance(pairs[0], dict):
            raise ValueError("fit() needs (window, meta) pairs, not bare windows -- "
                             "labels live in meta")
        windows = [w for w, m in pairs]
        labels = [m["state"] for w, m in pairs]
        groups = [m.get("subject", "unknown") for w, m in pairs]
        return self.fit_labeled(windows, labels, groups)

    def fit_labeled(self, windows, labels, groups=None, seed=0,
                    n_estimators=300, holdout_frac=0.25):
        """Train the forest, then calibrate and tune the reject threshold on
        a subject-disjoint held-out split. Subject-disjoint matters: windows
        from one trial are near-duplicates, so a random split would leak."""
        keep = [i for i, l in enumerate(labels) if l in STATES]
        if len(keep) < 50:
            raise ValueError(f"only {len(keep)} labelled windows -- not enough to train")
        X = np.vstack([extract_features(windows[i], self.gait_band) for i in keep])
        y = np.asarray([labels[i] for i in keep])
        g = np.asarray([groups[i] for i in keep]) if groups is not None \
            else np.arange(len(keep))

        splitter = GroupShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
        tr, ho = next(splitter.split(X, y, groups=g))
        if len(np.unique(y[tr])) < 2:
            raise ValueError("training split has fewer than two states")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        self.model.fit(X[tr], y[tr])
        self.classes_ = list(self.model.classes_)

        p_ho = self.model.predict_proba(X[ho])
        y_ho = np.asarray([self.classes_.index(v) for v in y[ho]])
        self.ece_before = expected_calibration_error(p_ho, y_ho)
        self.temperature = fit_temperature(p_ho, y_ho)
        p_cal = _softmax(_probs_to_logits(p_ho) / self.temperature)
        self.ece_after = expected_calibration_error(p_cal, y_ho)

        # Reject threshold is tuned on the RAW forest vote fraction, not the
        # temperature-scaled probability. Temperature scaling exists to make
        # the number fusion multiplies by honest; if T < 1 it sharpens, which
        # would destroy the reject. The two jobs are kept separate.
        correct = p_ho.argmax(axis=1) == y_ho
        conf_ok = p_ho.max(axis=1)[correct]
        raw_thr = float(np.quantile(conf_ok, DEFAULT_REJECT_Q)) if len(conf_ok) else 0.0
        # Guard: on an easy held-out fold every correct prediction sits at
        # confidence 1.0 and the quantile comes back 1.0, which would reject
        # everything. Cap it.
        self.reject_threshold = float(np.clip(raw_thr, 0.0, REJECT_THRESHOLD_MAX))
        if raw_thr > REJECT_THRESHOLD_MAX:
            print(f"[context] held-out fold was degenerate (raw reject threshold "
                  f"{raw_thr:.3f}); capped at {REJECT_THRESHOLD_MAX}", file=sys.stderr)

        self._holdout = (X[ho], y[ho], p_cal)
        return self

    # -- inference --------------------------------------------------------

    def predict(self, window: dict):
        """Return (state, confidence, prob_dict). state may be 'unknown'."""
        if self.model is None:
            return UNKNOWN, 0.0, {}
        f = extract_features(window, self.gait_band).reshape(1, -1)
        p_raw = self.model.predict_proba(f)[0]
        p_cal = _softmax(_probs_to_logits(p_raw.reshape(1, -1)) / self.temperature)[0]
        i = int(np.argmax(p_cal))
        conf = float(p_cal[i])                       # what fusion weights by
        probs = {c: float(v) for c, v in zip(self.classes_, p_cal)}
        if float(p_raw.max()) < self.reject_threshold:   # open-set, on raw votes
            return UNKNOWN, conf, probs
        return self.classes_[i], conf, probs

    def score(self, window: dict) -> float:
        """Context risk contribution in [0,1]. Sets last_state,
        last_confidence, last_quality. Never returns None or NaN."""
        try:
            acc = window.get("acc")
            if self.model is None or acc is None or len(acc) == 0:
                self.last_state, self.last_confidence = UNKNOWN, 0.0
                self.last_quality = 0.0
                self.last_probs = {}
                return 0.0

            state, conf, probs = self.predict(window)
            self.last_state = state
            self.last_confidence = conf
            self.last_probs = probs

            if state == UNKNOWN:
                # Open-set reject: fusion should down-weight, not act on a
                # label we do not believe.
                self.last_quality = 0.0
                return 0.0

            self.last_quality = float(np.clip(conf, 0.0, 1.0))
            risk = STATE_RISK.get(state, 0.0)
            return float(np.clip(risk, 0.0, 1.0))

        except Exception as e:                          # noqa: BLE001
            print(f"[context] score failed: {e}", file=sys.stderr)
            self.last_state, self.last_confidence = UNKNOWN, 0.0
            self.last_quality = 0.0
            return 0.0

    # -- persistence ------------------------------------------------------

    def save(self, path):
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model, "classes": self.classes_,
            "temperature": self.temperature,
            "reject_threshold": self.reject_threshold,
            "gait_band": self.gait_band,
            "ece_before": self.ece_before, "ece_after": self.ece_after,
        }, path)
        return path

    @classmethod
    def load(cls, path):
        import joblib
        d = joblib.load(path)
        s = cls(gait_band=tuple(d["gait_band"]),
                reject_threshold=d["reject_threshold"])
        s.model = d["model"]
        s.classes_ = d["classes"]
        s.temperature = d["temperature"]
        s.ece_before = d.get("ece_before")
        s.ece_after = d.get("ece_after")
        return s


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _train(root, out, elderly=False):
    from data.loader import scan, iter_windows      # lazy: CLI only
    windows, labels, groups = [], [], []
    for rec in scan(root):
        if rec.is_fall or rec.state is None:
            continue
        for w, m in iter_windows(rec):
            windows.append(w)
            labels.append(m["state"])
            groups.append(m["subject"])
    if not windows:
        print("No usable ADL windows found. Run: py -m data.loader --scan <dir>")
        return
    print(f"{len(windows)} windows, {len(set(groups))} subjects")
    for s in STATES:
        print(f"  {s:24s} {labels.count(s)}")

    cs = ContextStream(elderly_gait=elderly).fit_labeled(windows, labels, groups)
    X, y, p = cs._holdout
    pred = [cs.classes_[i] for i in p.argmax(axis=1)]
    print("\n--- held-out subjects ---")
    print(classification_report(y, pred, zero_division=0))
    print("confusion matrix, rows true, order:", cs.classes_)
    print(confusion_matrix(y, pred, labels=cs.classes_))
    print(f"\ntemperature      T = {cs.temperature:.3f}")
    print(f"ECE before       {cs.ece_before:.4f}")
    print(f"ECE after        {cs.ece_after:.4f}")
    print(f"reject threshold {cs.reject_threshold:.3f}")
    print(f"\nsaved -> {cs.save(out)}")


def _selftest():
    """Synthetic four-state fixture. Proves the machinery trains, calibrates,
    rejects and conforms to the contract. Not project data, never reported."""
    rng = np.random.default_rng(0)
    fs, n = 50, 125

    def make(state, subj):
        t = np.arange(n) / fs
        if state == "ambulating":
            a = np.column_stack([0.45 * np.sin(2*np.pi*1.9*t), 0.35*np.cos(2*np.pi*1.9*t),
                                 1.0 + 0.5*np.sin(2*np.pi*3.8*t)])
            gy = rng.normal(0, 60, (n, 3))
        elif state == "stationary":
            a = np.column_stack([np.zeros(n), np.zeros(n), np.ones(n)])
            gy = rng.normal(0, 3, (n, 3))
        elif state == "seated hand activity":
            a = np.column_stack([0.6*np.sin(2*np.pi*0.9*t), 0.2*np.ones(n),
                                 1.0 + 0.2*np.sin(2*np.pi*2.7*t)])
            gy = rng.normal(0, 90, (n, 3))
        else:  # lying/immobile -- forearm near-horizontal, near-zero motion
            a = np.column_stack([np.ones(n), np.zeros(n), 0.05*np.ones(n)])
            gy = rng.normal(0, 1, (n, 3))
        a = a + rng.normal(0, 0.03, (n, 3))
        return {"t": 0.0, "acc": a.astype(np.float32), "gyro": gy.astype(np.float32),
                "hr": None, "temp": None, "fs": fs}

    windows, labels, groups = [], [], []
    for subj in range(8):
        for state in STATES:
            for _ in range(25):
                windows.append(make(state, subj))
                labels.append(state)
                groups.append(f"S{subj}")

    cs = ContextStream().fit_labeled(windows, labels, groups)
    print(f"T = {cs.temperature:.3f}   ECE {cs.ece_before:.4f} -> {cs.ece_after:.4f}"
          f"   reject < {cs.reject_threshold:.3f}")

    w = make("lying/immobile", 0)
    s = cs.score(w)
    assert isinstance(s, float) and 0.0 <= s <= 1.0 and not np.isnan(s)
    assert 0.0 <= cs.last_quality <= 1.0
    print(f"lying window -> state={cs.last_state} score={s:.2f} q={cs.last_quality:.2f}")

    w2 = make("seated hand activity", 0)
    cs.score(w2)
    print(f"hand window  -> state={cs.last_state} conf={cs.last_confidence:.2f}"
          f"  (fusion halves motion if conf > 0.7)")

    # Out-of-distribution: violent nonsense the forest has never seen.
    ood = {"t": 0.0, "acc": rng.normal(0, 6, (n, 3)).astype(np.float32),
           "gyro": rng.normal(0, 900, (n, 3)).astype(np.float32),
           "hr": None, "temp": None, "fs": fs}
    s_ood = cs.score(ood)
    print(f"OOD window   -> state={cs.last_state} score={s_ood:.2f} q={cs.last_quality:.2f}")

    # Contract: an untrained stream and a broken window must not raise.
    blank = ContextStream()
    assert blank.score(windows[0]) == 0.0 and blank.last_quality == 0.0
    assert cs.score({"acc": None, "fs": 50}) == 0.0 and cs.last_quality == 0.0
    assert cs.score({}) == 0.0
    print("SELFTEST PASS")


def _report(model_path):
    cs = ContextStream.load(model_path)
    print(f"classes          {cs.classes_}")
    print(f"gait band        {cs.gait_band[0]}-{cs.gait_band[1]} Hz")
    print(f"temperature      {cs.temperature:.3f}")
    print(f"reject threshold {cs.reject_threshold:.3f}")
    print(f"ECE before/after {cs.ece_before} / {cs.ece_after}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Guardian context stream")
    ap.add_argument("--train", metavar="DIR")
    ap.add_argument("--out", default="models/context.joblib")
    ap.add_argument("--elderly", action="store_true",
                    help="widen the gait band to 0.8-2.8 Hz")
    ap.add_argument("--report", metavar="MODEL")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.train:
        _train(a.train, a.out, a.elderly)
    elif a.report:
        _report(a.report)
    elif a.selftest:
        _selftest()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
