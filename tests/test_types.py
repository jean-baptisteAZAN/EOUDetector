from eou_detector.types import (
    Partial, LexResult, FusionInput, FusionResult, Decision,
)


def test_partial_defaults():
    p = Partial(text="bonjour")
    assert p.text == "bonjour"
    assert p.is_final is False
    assert p.ts_ms == 0.0


def test_fusion_result_fields():
    r = FusionResult(p_eou=0.9, decision="ENDPOINT",
                     required_silence_ms=200.0, reason="both_confident")
    assert r.decision == "ENDPOINT"
    assert r.required_silence_ms == 200.0


def test_decision_carries_signals():
    d = Decision(ts_ms=1.0, p_ac=0.8, p_lex=0.7, p_eou=0.78,
                 decision="WAIT", required_silence_ms=600.0, silence_ms=120.0,
                 latency_ms=30.0, reason="mid", partial_text="le vingt")
    assert d.p_ac == 0.8 and d.decision == "WAIT"
