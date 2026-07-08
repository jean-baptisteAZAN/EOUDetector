from eou_detector.config import Settings
from eou_detector.types import FusionInput
from eou_detector.fusion.rules import RuleFusion
from eou_detector.fusion.logistic import LogisticFusion

S = Settings()


def fi(p_ac, p_lex, sil=300.0, ac=True, lex=True):
    return FusionInput(p_ac=p_ac, p_ac_available=ac, p_lex=p_lex,
                       p_lex_available=lex, silence_ms=sil)


def test_lexical_veto_forces_wait():
    r = RuleFusion(S).fuse(fi(p_ac=0.9, p_lex=0.05))
    assert r.decision == "WAIT" and r.reason == "lexical_veto"


def test_acoustic_veto_forces_wait():
    r = RuleFusion(S).fuse(fi(p_ac=0.05, p_lex=0.9))
    assert r.decision == "WAIT" and r.reason == "acoustic_veto"


def test_both_confident_endpoints_with_short_silence():
    r = RuleFusion(S).fuse(fi(p_ac=0.95, p_lex=0.95))
    assert r.decision == "ENDPOINT"
    assert r.reason == "both_confident"
    assert r.required_silence_ms == S.short_ms


def test_midband_returns_probability_and_longer_silence():
    r = RuleFusion(S).fuse(fi(p_ac=0.6, p_lex=0.6))
    assert 0.0 <= r.p_eou <= 1.0
    assert r.required_silence_ms >= S.med_ms


def test_missing_lexical_uses_acoustic_only():
    r = RuleFusion(S).fuse(fi(p_ac=0.95, p_lex=0.0, lex=False))
    assert r.decision in ("ENDPOINT", "WAIT")
    assert r.reason != "lexical_veto"   # cannot veto on unavailable signal


def test_logistic_same_interface():
    w = {"bias": -4.0, "p_ac": 4.0, "p_lex": 4.0, "inter": 0.0, "sil": 0.0}
    r = LogisticFusion(S, w).fuse(fi(p_ac=0.95, p_lex=0.95))
    assert 0.0 <= r.p_eou <= 1.0
    assert r.decision in ("ENDPOINT", "WAIT")
