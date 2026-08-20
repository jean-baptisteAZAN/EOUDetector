import numpy as np
import torch
from silero_vad import load_silero_vad

from .base import VAD


class SileroVAD(VAD):
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self._model = load_silero_vad(onnx=False)
        self._sr = sample_rate
        self._threshold = threshold

    def process(self, frame: bytes) -> bool:
        pcm = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        t = torch.from_numpy(pcm)
        with torch.no_grad():
            prob = float(self._model(t, self._sr).item())
        return prob >= self._threshold

    def reset(self) -> None:
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()
