import numpy as np
import pytest
from eou_detector.vad.base import VAD


class _ProbVAD(VAD):
    """Test double over the same interface (no model download)."""
    def __init__(self, probs, threshold=0.5):
        self._probs, self._i, self._t = probs, 0, threshold
    def process(self, frame: bytes) -> bool:
        p = self._probs[min(self._i, len(self._probs) - 1)]
        self._i += 1
        return p >= self._t
    def reset(self) -> None:
        self._i = 0


def test_interface_threshold_behaviour():
    v = _ProbVAD([0.1, 0.9, 0.4], threshold=0.5)
    frame = np.zeros(512, dtype=np.int16).tobytes()
    assert v.process(frame) is False
    assert v.process(frame) is True
    assert v.process(frame) is False


@pytest.mark.integration
def test_silero_loads_and_scores_silence():
    pytest.importorskip("silero_vad")
    from eou_detector.vad.silero_vad import SileroVAD
    v = SileroVAD(sample_rate=16000, threshold=0.5)
    silence = np.zeros(512, dtype=np.int16).tobytes()
    assert v.process(silence) is False
