"""
core/decision.py

REAL decision state machine. Stateful, timestamp-driven (uses the
replay t supplied by the caller, never wall-clock time), no Streamlit
dependency.

STATES
------
normal, motion event, possible incident, post-event monitoring,
physiological confirmation, escalate, sensor uncertainty.

CORE MECHANISM (deterministic, documented here since this is what's
actually tested):

- entry threshold 0.55, exit threshold 0.40 (hysteresis): once risk
  rises to or above entry, the machine leaves "normal". It only
  returns to "normal" when risk drops BELOW exit (0.40) -- while risk
  sits in the 0.40-0.55 band, the machine holds its current elevated
  state rather than flickering back to normal.
- dwell: risk must stay at/above entry for ~30 continuous seconds
  (DWELL_SECONDS) before the state becomes "escalate". A brief spike
  that drops back below entry before 30s resets the dwell clock and
  does not escalate.
- manual SOS: update(..., manual_sos=True) forces "escalate"
  immediately, unconditionally, bypassing risk/dwell entirely.

STATE-LABEL MAPPING (this is a documented DESIGN CHOICE, not something
independently specified elsewhere, since update() only receives a
fused risk value + optional contributions -- not raw per-stream
signals like post-impact stillness):

- "sensor uncertainty": entered when `contributions` is passed and is
  empty (fusion had no usable evidence this window). Distinct from
  "normal", which means risk was confirmed low, not unknown.
- "motion event": the FIRST window of a new elevation episode
  (transitioning out of normal) where motion is the dominant
  contributor.
- "physiological confirmation": an elevation episode (first window or
  continuing) where physiology is the dominant contributor -- covers a
  standalone HR-driven rise with no impact.
- "possible incident": an elevation episode continuing on subsequent
  windows where motion/context (not physiology) dominates.
- "post-event monitoring": NOT reachable by this implementation.
  Reaching it correctly requires a post-impact stillness/settled
  signal from MotionStream that is not yet part of this review's
  interface (update() only gets risk + contributions). The state is
  defined and will be reachable once that signal exists; this module
  does not invent a fake trigger for it.
- "escalate": dwell condition satisfied, or manual SOS.

USAGE
-----
    machine = DecisionStateMachine()
    state = machine.update(risk, t, contributions=contributions)
    state = machine.update(risk, t, manual_sos=True)   # forces escalate
"""

ENTRY_THRESHOLD = 0.55
EXIT_THRESHOLD = 0.40
DWELL_SECONDS = 30.0
DEFAULT_DT = 1.25   # assumed spacing on the very first update() call
MAX_DT = 5.0        # sanity cap against replay time-gaps


def _dominant_contributor(contributions):
    if not contributions:
        return None
    return max(contributions, key=contributions.get)


class DecisionStateMachine:
    def __init__(self) -> None:
        self.state = "normal"
        self._above_entry_seconds = 0.0
        self._episode_started = False
        self._last_t = None

    def update(self, risk: float, t: float, contributions: dict = None, manual_sos: bool = False) -> str:
        if manual_sos:
            self.state = "escalate"
            self._last_t = t
            return self.state

        # Sensor uncertainty: fusion had nothing usable this window.
        if contributions is not None and len(contributions) == 0:
            self.state = "sensor uncertainty"
            self._last_t = t
            return self.state

        risk = max(0.0, min(1.0, float(risk)))

        if t is not None and self._last_t is not None:
            dt = t - self._last_t
        else:
            dt = DEFAULT_DT
        dt = max(0.0, min(dt, MAX_DT))
        self._last_t = t

        if risk >= ENTRY_THRESHOLD:
            self._above_entry_seconds += dt
        elif risk < EXIT_THRESHOLD:
            self._above_entry_seconds = 0.0
            self._episode_started = False
            self.state = "normal"
            return self.state
        # else: hysteresis band (exit <= risk < entry) -- dwell timer
        # frozen, fall through without resetting.

        if self._above_entry_seconds >= DWELL_SECONDS:
            self.state = "escalate"
        elif risk >= ENTRY_THRESHOLD:
            dominant = _dominant_contributor(contributions)
            if dominant == "physiology":
                self.state = "physiological confirmation"
            elif not self._episode_started:
                self.state = "motion event"
            else:
                self.state = "possible incident"
            self._episode_started = True
        # else: in hysteresis band -- keep whatever state we already had
        # (self.state is left unchanged, which is the whole point of
        # hysteresis: no flicker).

        return self.state