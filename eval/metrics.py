from dataclasses import dataclass
from statistics import median
from typing import List, Optional, Tuple

from eou_detector.config import Settings
from eou_detector.types import Decision


@dataclass
class ClipResult:
    label: str                       # "fini" | "pas_fini"
    predicted_finished: bool
    latency_ms: Optional[float]


def evaluate_clip(decisions: List[Decision], s: Settings) -> Tuple[bool, Optional[float]]:
    for dcn in decisions:
        if dcn.decision == "ENDPOINT":
            # A genuine semantic endpoint fires before the max-silence catch-all.
            if dcn.silence_ms < s.max_silence_ms:
                return True, dcn.silence_ms
            return False, None
    return False, None


def _pct(values, q):
    if not values:
        return None
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


def summarize(results: List[ClipResult]) -> dict:
    n = len(results)
    fp = sum(1 for r in results
             if r.label == "pas_fini" and r.predicted_finished)
    fn = sum(1 for r in results
             if r.label == "fini" and not r.predicted_finished)
    correct = n - fp - fn
    lats = [r.latency_ms for r in results if r.latency_ms is not None]
    return {
        "n": n,
        "accuracy": (correct / n) if n else 0.0,
        "false_positives": fp,
        "false_negatives": fn,
        "median_latency_ms": median(lats) if lats else None,
        "p90_latency_ms": _pct(lats, 0.9),
    }
