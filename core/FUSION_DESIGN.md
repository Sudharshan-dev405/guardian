# Fusion design (Stage 5.11, on paper)

Status: **design only**. Nothing here is built or tested yet; that is Stage 7.
Numbers marked *starting value* are first guesses to be tuned on the simulated
scenarios and later on real band data. None of them is a clinical threshold.

## 1. What fusion reads, every window (about every 1.24 s)

| Stream | Values fusion uses | Status |
|---|---|---|
| motion | `score` (impact, 0 to 1), `quality`, `stillness`, `still_s` (seconds without real movement), `moved_after_impact_s`, `recovered`, `time_since_impact`, `gate_open` | built |
| activity | `state` (stationary, ambulating, seated hand activity, lying/immobile, unknown), `confidence`, `quality` | being replaced (Sudharshan) |
| physiological | `score`, `quality`, plus heart-rate fields still to be agreed | placeholder (Mithuna) |
| context | time of day, what the person was doing in the last few minutes | derived inside fusion |
| SOS button | pressed / not pressed | event, from the band |

All of these are already stored per window in `stream_outputs` (extras), so
fusion can run live or replay a stored run.

## 2. Rules that hold for every emergency type

1. No emergency depends on the motion gate alone. Faints, unresponsiveness and
   heart-rate problems often have no hard impact.
2. Every piece of evidence is weighted by its stream's `quality`. A missing
   stream adds nothing and lowers confidence; it is never assumed normal.
3. Activity is smoothed before use (majority over the last 10 s), because the
   current module flips state every 1 to 3 s.
4. Stillness alone is never an emergency. People sleep and sit still. It needs
   a preceding impact, an abnormal heart rate, or no answer to a check-in.
5. Every alert stores its evidence: which streams said what, and when.
6. No SpO2, no personal baseline, no claim that the score is a validated risk.

## 3. Emergency types

| Emergency | Evidence that raises it | Evidence that lowers it | Timing (starting values) |
|---|---|---|---|
| Hard fall | motion score above 0.81 | moved again within 30 s (`recovered`), activity back to ambulating | flag at impact + 3 s; wearer cancel window 30 s |
| Long lie | an impact (score above 0.5), then `still_s` keeps growing, `recovered` stays false | any real movement | caregiver alert at 60 s still; urgent at 5 min |
| Faint / soft collapse | upright activity then lying/immobile within a few seconds, heart rate drop or jump, stillness after; impact optional | moved again, normal heart rate | 30 s after the change |
| Unresponsive, no fall | very long stillness at an unusual time, abnormal heart rate, no answer to a check-in | night-time, lying down gradually (looks like sleep), normal heart rate | check-in after 30 min still in the daytime (starting value) |
| Heart-rate emergency | physiological score high with good quality for a sustained period | activity explains it (walking, stairs), poor signal quality | sustained 60 s; thresholds from Mithuna, population-level, with sources |
| Seizure (convulsive) | rhythmic shaking for tens of seconds | not covered | needs its own detector; future work |
| Overheating / cold | skin temperature trend only | weak evidence | information only, never an alert on its own |
| SOS button | button pressed | none | immediate |

The clinical "long lie" is an hour or more on the floor (Fleming and Brayne
2008: 30 per cent of fallers over 90 lay over an hour). The 60 s alert is meant
to stop a long lie from happening, not to detect one.

## 4. Emergency score and alert levels

For each emergency type: score = weighted evidence (each piece times its
stream's quality), minus weighted counter-evidence, clipped to 0 to 1.
The emergency score for the window is the highest type score, and the alert
names that type.

| Emergency score | What happens |
|---|---|
| below 0.4 | nothing; logged |
| 0.4 to 0.7 | check-in on the band ("Are you OK?"); wearer can dismiss |
| 0.7 and above | wearer cancel window (30 s), then caregiver alert |
| long lie past 5 min, or no answer to a check-in | urgent caregiver alert |

Weights are *starting values*, set so the six scenarios below come out as
expected, then frozen before any new scenario is tried.

## 5. Alert lifecycle (the `alerts` table)

open (wearer can cancel) -> sent to caregivers -> acknowledged (by a linked
caregiver or the admin) -> resolved or false alarm. `evidence` holds the
streams and values behind it. Who acknowledged and when is stored; every view
goes to `audit_log`.

## 6. Expected results on the simulated scenarios

| Scenario | Expected |
|---|---|
| normal_day | no alert |
| hard_fall_long_lie | hard fall flagged a few seconds after impact, caregiver alert at 60 s still |
| fall_then_get_up | hard fall flagged, then cleared once the person moves again; no caregiver alert |
| faint_collapse | alert (impact path today; heart-rate path once Mithuna's module is in) |
| false_alarm | no alert (gate opens, score stays low) |
| hr_spike_no_fall | no alert today (no physiological module); heart-rate alert after Stage 6 |
| new: night_sleep | long stillness at night, normal heart rate: no alert |

## 7. Open questions for the team

- Heart-rate fields and thresholds (Mithuna), with sources.
- Final activity states (Sudharshan's module).
- How the band shows a check-in and the cancel button (Sudharshan, Stage 8).
