import asyncio
import numpy as np
import pytest

from eou_detector.config import Settings
from eou_detector.audio.source import AudioSource
from eou_detector.vad.base import VAD
from eou_detector.asr.scripted_asr import ScriptedASR
from eou_detector.eou.acoustic import StubAcousticEOU
from eou_detector.eou.lexical import FrenchSemanticEOU
from eou_detector.fusion.rules import RuleFusion
from eou_detector.endpoint.controller import EndpointController
from eou_detector.orchestrator import Orchestrator

FRAME = np.zeros(512, dtype=np.int16).tobytes()


class ListSource(AudioSource):
    def __init__(self, n): self._n = n
    async def frames(self):
        for _ in range(self._n):
            yield FRAME


class PatternVAD(VAD):
    """speech for the first `speech_frames`, then silence."""
    def __init__(self, speech_frames): self._sf, self._i = speech_frames, 0
    def process(self, frame):
        self._i += 1
        return self._i <= self._sf
    def reset(self): self._i = 0


def _fake_time():
    t = {"ms": 0.0}
    def now():
        t["ms"] += 32.0   # one 32 ms frame per call
        return t["ms"]
    return now


def _run(asr, settings=None):
    s = settings or Settings()
    decisions = []
    orch = Orchestrator(
        settings=s, source=ListSource(60), vad=PatternVAD(speech_frames=10),
        asr=asr, acoustic=StubAcousticEOU(0.95),
        lexical=FrenchSemanticEOU(),
        fusion=RuleFusion(s), endpoint=EndpointController(s),
        on_decision=decisions.append, time_fn=_fake_time())
    asyncio.run(orch.run())
    return decisions


def test_complete_partial_reaches_endpoint():
    # "oui c'est exact" -> no heuristic veto, stub p_ac high -> ENDPOINT.
    asr = ScriptedASR(script=[(0, ""), (5, "oui c'est exact")])
    decisions = _run(asr)
    assert any(d.decision == "ENDPOINT" for d in decisions)


def test_spelling_partial_is_held_by_veto():
    # trailing "m a r" -> lexical veto -> only the max-silence catch-all may end it.
    asr = ScriptedASR(script=[(0, ""), (5, "mon nom c'est m a r")])
    decisions = _run(asr)
    waits = [d for d in decisions if d.decision == "WAIT"]
    assert len(waits) >= 1
    assert any(d.reason == "lexical_veto" for d in decisions)
