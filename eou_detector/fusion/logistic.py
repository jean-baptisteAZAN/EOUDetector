import math
from .base import Fusion, required_silence
from eou_detector.config import Settings
from eou_detector.types import FusionInput, FusionResult


class LogisticFusion(Fusion):
    """Drop-in replacement for RuleFusion once weights are learned. Same
    interface, same silence policy; only the p_eou model differs."""

    def __init__(self, settings: Settings, weights: dict):
        self._s = settings
        self._w = weights

    def fuse(self, x: FusionInput) -> FusionResult:
        s, w = self._s, self._w
        sil_norm = min(x.silence_ms / s.max_silence_ms, 1.0)
        z = (w.get("bias", 0.0)
             + w.get("p_ac", 0.0) * (x.p_ac if x.p_ac_available else 0.0)
             + w.get("p_lex", 0.0) * (x.p_lex if x.p_lex_available else 0.0)
             + w.get("inter", 0.0) * x.p_ac * x.p_lex
             + w.get("sil", 0.0) * sil_norm)
        p_eou = 1.0 / (1.0 + math.exp(-z))
        req = required_silence(p_eou, s)
        decision = "ENDPOINT" if (p_eou >= s.mid and x.silence_ms >= req) else "WAIT"
        return FusionResult(p_eou, decision, req, "logistic")
