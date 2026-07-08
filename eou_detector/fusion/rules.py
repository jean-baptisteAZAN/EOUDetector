from .base import Fusion, required_silence
from eou_detector.config import Settings
from eou_detector.types import FusionInput, FusionResult


class RuleFusion(Fusion):
    def __init__(self, settings: Settings):
        self._s = settings

    def fuse(self, x: FusionInput) -> FusionResult:
        s = self._s
        # 1) Bilateral veto on available signals.
        if x.p_lex_available and x.p_lex < s.veto_lex:
            p = min(x.p_lex, x.p_ac if x.p_ac_available else x.p_lex)
            return FusionResult(p, "WAIT", s.max_silence_ms, "lexical_veto")
        if x.p_ac_available and x.p_ac < s.veto_ac:
            p = min(x.p_ac, x.p_lex if x.p_lex_available else x.p_ac)
            return FusionResult(p, "WAIT", s.max_silence_ms, "acoustic_veto")

        # 2) Combine available signals into p_eou.
        if x.p_ac_available and x.p_lex_available:
            p_eou = 0.5 * x.p_ac + 0.5 * x.p_lex
        elif x.p_ac_available:
            p_eou = x.p_ac
        elif x.p_lex_available:
            p_eou = x.p_lex
        else:
            p_eou = 0.0

        # 3) Both strongly confident -> endpoint immediately (short silence).
        if (x.p_ac_available and x.p_lex_available
                and x.p_ac >= s.high and x.p_lex >= s.high):
            return FusionResult(p_eou, "ENDPOINT", s.short_ms, "both_confident")

        req = required_silence(p_eou, s)
        decision = "ENDPOINT" if (p_eou >= s.mid and x.silence_ms >= req) else "WAIT"
        reason = "score_endpoint" if decision == "ENDPOINT" else "score_wait"
        return FusionResult(p_eou, decision, req, reason)
