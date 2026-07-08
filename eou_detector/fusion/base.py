import abc
from eou_detector.config import Settings
from eou_detector.types import FusionInput, FusionResult


class Fusion(abc.ABC):
    @abc.abstractmethod
    def fuse(self, x: FusionInput) -> FusionResult: ...


def required_silence(p_eou: float, s: Settings) -> float:
    """Confidence -> how much trailing silence we demand before endpointing."""
    if p_eou >= s.high:
        return s.short_ms
    if p_eou >= s.mid:
        return s.med_ms
    return s.long_ms
