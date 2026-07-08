from eou_detector.config import Settings
from eou_detector.types import FusionResult


class EndpointController:
    """Applies the temporal silence policy on top of fusion's instantaneous
    decision. Confidence sets how long of a pause is required; a hard
    max-silence catch-all guarantees the turn eventually ends."""

    def __init__(self, settings: Settings):
        self._s = settings

    def evaluate(self, fusion: FusionResult, silence_ms: float) -> str:
        if silence_ms >= self._s.max_silence_ms:
            return "ENDPOINT"  # catch-all overrides any veto
        if fusion.decision == "ENDPOINT" and silence_ms >= fusion.required_silence_ms:
            return "ENDPOINT"
        return "WAIT"

    def reset(self) -> None:
        return None
