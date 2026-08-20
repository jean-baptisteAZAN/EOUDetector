import abc
import glob
import os
from typing import Optional

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

# Preferred Smart Turn v3 ONNX variants, best accuracy first (see the model
# card benchmarks: v3.2 > v3.1 > v3.0; *-cpu is the CPU-optimised export).
_SMART_TURN_PREFERENCE = (
    "smart-turn-v3.2-cpu.onnx",
    "smart-turn-v3.1-cpu.onnx",
    "smart-turn-v3.0.onnx",
)


def find_smart_turn_model(models_dir: str = "models") -> Optional[str]:
    """Locate a downloaded Smart Turn v3 ONNX, preferring the best variant.

    Returns the path, or None if no .onnx is present (callers fall back to the
    stub). This makes the exact downloaded filename (e.g. ``smart-turn-v3.0``
    vs ``smart-turn-v3.2-cpu``) a non-issue for the demo/harness."""
    for name in _SMART_TURN_PREFERENCE:
        p = os.path.join(models_dir, name)
        if os.path.exists(p):
            return p
    others = sorted(glob.glob(os.path.join(models_dir, "smart-turn*.onnx")))
    return others[0] if others else None


class AcousticEOU(abc.ABC):
    @abc.abstractmethod
    def predict(self, pcm: np.ndarray) -> float:
        """int16 PCM window -> probability the speaker finished (p_ac)."""


class StubAcousticEOU(AcousticEOU):
    def __init__(self, value: float = 0.5):
        self._value = float(value)

    def predict(self, pcm: np.ndarray) -> float:
        return self._value


class SmartTurnV3(AcousticEOU):
    """Smart Turn v3 (Whisper-Tiny encoder + linear head), ONNX, local CPU.

    Preprocessing matches the official pipecat-ai/smart-turn inference exactly:
    take the trailing ``window_s`` seconds of audio, build Whisper log-mel
    features (80 mels x 800 frames) via WhisperFeatureExtractor with
    do_normalize=True, right-pad to the 8 s window. The ONNX graph applies the
    sigmoid internally, so the output is already a completion probability in
    [0, 1] -- do NOT squash it again."""

    def __init__(self, model_path: str, sample_rate: int = 16000,
                 window_s: float = 8.0):
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self._in_name = self._sess.get_inputs()[0].name
        self._out_name = self._sess.get_outputs()[0].name
        self._sr = sample_rate
        self._win = int(window_s * sample_rate)
        # chunk_length=8 -> 8 s -> 800 frames; defaults give 80 mels, n_fft=400,
        # hop=160 (Whisper-Tiny front end).
        self._fe = WhisperFeatureExtractor(chunk_length=8)

    def _prep(self, pcm: np.ndarray) -> np.ndarray:
        x = np.asarray(pcm, dtype=np.float32) / 32768.0
        x = x[-self._win:]  # truncate to the last 8 s (trailing portion)
        inputs = self._fe(
            x, sampling_rate=self._sr, return_tensors="np",
            padding="max_length", max_length=self._win, truncation=True,
            do_normalize=True)
        feats = inputs.input_features.squeeze(0).astype(np.float32)
        return feats[np.newaxis, ...]  # (1, 80, 800)

    def predict(self, pcm: np.ndarray) -> float:
        feats = self._prep(pcm)
        out = self._sess.run([self._out_name], {self._in_name: feats})[0]
        prob = float(np.asarray(out).reshape(-1)[0])  # already a probability
        return min(1.0, max(0.0, prob))
