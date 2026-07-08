from eou_detector.config import Settings
from eou_detector.types import FusionResult
from eou_detector.endpoint.controller import EndpointController

S = Settings()


def fr(decision, req, reason="x", p=0.9):
    return FusionResult(p_eou=p, decision=decision, required_silence_ms=req,
                        reason=reason)


def test_endpoints_when_silence_meets_requirement():
    c = EndpointController(S)
    assert c.evaluate(fr("ENDPOINT", S.short_ms), silence_ms=S.short_ms) == "ENDPOINT"


def test_waits_when_silence_below_requirement():
    c = EndpointController(S)
    assert c.evaluate(fr("ENDPOINT", S.med_ms), silence_ms=100.0) == "WAIT"


def test_veto_holds_until_max_silence_catch_all():
    c = EndpointController(S)
    veto = fr("WAIT", S.max_silence_ms, reason="lexical_veto", p=0.05)
    assert c.evaluate(veto, silence_ms=800.0) == "WAIT"
    assert c.evaluate(veto, silence_ms=S.max_silence_ms) == "ENDPOINT"  # catch-all


def test_catch_all_forces_endpoint_on_low_confidence():
    c = EndpointController(S)
    low = fr("WAIT", S.long_ms, reason="score_wait", p=0.2)
    assert c.evaluate(low, silence_ms=S.max_silence_ms + 1) == "ENDPOINT"
