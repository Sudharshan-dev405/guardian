# test_pipeline_stub.py -- confirms the full stub chain runs end to end
from data.loader import replay_stub
from streams.context import ContextStream
from streams.motion import MotionStream
from streams.physiology import PhysiologyStream
from streams.quality import QualityStream
from core.fusion import fuse_stub
from core.decision import decide_stub
from core.explain import explain_stub

streams = {
    "context": ContextStream(),
    "motion": MotionStream(),
    "physiology": PhysiologyStream(),
    "quality": QualityStream(),
}

for window in replay_stub(n_windows=5):
    scores = {name: s.score(window) for name, s in streams.items()}
    risk = fuse_stub(scores)
    state = decide_stub(risk)
    explanation = explain_stub(risk, scores)

    print(f"t={window['t']:.2f}  risk={risk:.3f}  state={state}")
    print(f"  {explanation}")