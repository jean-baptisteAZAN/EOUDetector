from eou_detector.types import Decision
from demo import format_decision


def test_format_decision_contains_fields():
    d = Decision(ts_ms=1000.0, p_ac=0.91, p_lex=0.80, p_eou=0.85,
                 decision="ENDPOINT", required_silence_ms=200.0, silence_ms=210.0,
                 latency_ms=27.3, reason="both_confident", partial_text="oui")
    line = format_decision(d)
    for token in ["p_ac=0.91", "p_lex=0.80", "dec=ENDPOINT", "lat=27", "reason=both_confident"]:
        assert token in line
