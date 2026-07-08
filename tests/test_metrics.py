from eou_detector.config import Settings
from eou_detector.types import Decision
from eval.metrics import evaluate_clip, summarize, ClipResult

S = Settings()


def d(decision, silence_ms, reason="score_endpoint"):
    return Decision(ts_ms=0, p_ac=0.9, p_lex=0.9, p_eou=0.9, decision=decision,
                    required_silence_ms=200.0, silence_ms=silence_ms,
                    latency_ms=20.0, reason=reason, partial_text="x")


def test_evaluate_clip_semantic_endpoint():
    fin, lat = evaluate_clip([d("WAIT", 100), d("ENDPOINT", 250)], S)
    assert fin is True and lat == 250


def test_evaluate_clip_only_catch_all_is_not_finished():
    fin, lat = evaluate_clip([d("WAIT", 800),
                              d("ENDPOINT", S.max_silence_ms, reason="catch_all")], S)
    assert fin is False and lat is None


def test_summarize_counts_fp_fn_and_latency():
    results = [
        ClipResult("fini", True, 220.0),
        ClipResult("fini", False, None),       # false negative
        ClipResult("pas_fini", True, 300.0),   # false positive
        ClipResult("pas_fini", False, None),   # correct
    ]
    s = summarize(results)
    assert s["n"] == 4
    assert s["false_negatives"] == 1
    assert s["false_positives"] == 1
    assert s["accuracy"] == 0.5
    assert s["median_latency_ms"] == 260.0
