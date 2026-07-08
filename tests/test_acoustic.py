import os
import numpy as np
import pytest
from eou_detector.eou.acoustic import AcousticEOU, StubAcousticEOU


def test_stub_returns_constant_in_range():
    a = StubAcousticEOU(0.7)
    p = a.predict(np.zeros(16000, dtype=np.int16))
    assert isinstance(a, AcousticEOU)
    assert 0.0 <= p <= 1.0 and p == 0.7


@pytest.mark.integration
def test_smart_turn_runs_if_model_present():
    from eou_detector.eou.acoustic import SmartTurnV3, find_smart_turn_model
    path = os.environ.get("SMART_TURN_ONNX") or find_smart_turn_model()
    if not path or not os.path.exists(path):
        pytest.skip("Smart Turn v3 ONNX not downloaded (python scripts/fetch_models.py)")
    a = SmartTurnV3(path)
    # Whisper log-mel preprocessing must yield the model's (1, 80, 800) input
    # and a probability in range, on a non-trivial waveform.
    rng = np.random.default_rng(0)
    speechy = (rng.standard_normal(16000 * 3) * 3000).astype(np.int16)
    p = a.predict(speechy)
    assert 0.0 <= p <= 1.0
