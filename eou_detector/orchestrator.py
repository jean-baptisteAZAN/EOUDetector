"""Standalone driver around an EouSession.

Owns an AudioSource and an ASR (e.g. for the demo / eval harness): it pulls
frames, pushes them to the ASR, forwards the latest partial into the session,
and relays the session's decisions to a callback. The WebSocket service drives
an EouSession directly instead of going through this class.
"""
import asyncio
from typing import Callable

from eou_detector.config import Settings
from eou_detector.audio.source import AudioSource
from eou_detector.vad.base import VAD
from eou_detector.asr.base import ASR
from eou_detector.eou.acoustic import AcousticEOU
from eou_detector.eou.lexical import LexicalEOU
from eou_detector.fusion.base import Fusion
from eou_detector.endpoint.controller import EndpointController
from eou_detector.session import EouSession, monotonic_ms
from eou_detector.types import Decision


class Orchestrator:
    def __init__(self, settings: Settings, source: AudioSource, vad: VAD,
                 asr: ASR, acoustic: AcousticEOU, lexical: LexicalEOU,
                 fusion: Fusion, endpoint: EndpointController,
                 on_decision: Callable[[Decision], None],
                 time_fn: Callable[[], float] = monotonic_ms):
        self._source = source
        self._asr = asr
        self._on_decision = on_decision
        self._session = EouSession(settings, vad, acoustic, lexical, fusion,
                                   endpoint, time_fn=time_fn)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        await self._asr.start()
        try:
            async for frame in self._source.frames():
                self._asr.send_audio(frame)
                self._session.set_partial(self._asr.latest_partial().text)
                for dec in await self._session.process(frame, loop):
                    self._on_decision(dec)
        finally:
            await self._asr.stop()
