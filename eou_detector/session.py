"""Per-call end-of-utterance session.

The decision core, isolated from any transport. One instance per call/callSid.
It is fed raw 16 kHz mono int16 PCM (the SAME audio the VAD service receives)
and the latest ASR partial (injected from outside via set_partial). On each
speech->silence micro-pause it fuses the acoustic and lexical signals and
returns a Decision. Both the standalone demo orchestrator and the WebSocket
service drive the exact same EouSession, so there is one decision code path.
"""
import asyncio
import time
from typing import Callable, List, Optional

import numpy as np

from eou_detector.config import Settings
from eou_detector.audio.ring_buffer import RingBuffer
from eou_detector.vad.base import VAD
from eou_detector.eou.acoustic import AcousticEOU
from eou_detector.eou.lexical import LexicalEOU
from eou_detector.fusion.base import Fusion
from eou_detector.endpoint.controller import EndpointController
from eou_detector.types import FusionInput, Decision


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0


class EouSession:
    """Stateful per-call endpoint detector. Not thread-safe: drive one session
    from a single asyncio task. Heavy models (acoustic, lexical) may be shared
    across sessions; the VAD is stateful and must be per-session."""

    def __init__(self, settings: Settings, vad: VAD, acoustic: AcousticEOU,
                 lexical: LexicalEOU, fusion: Fusion,
                 endpoint: EndpointController,
                 time_fn: Callable[[], float] = monotonic_ms):
        self._s = settings
        self._vad = vad
        self._acoustic = acoustic
        self._lexical = lexical
        self._fusion = fusion
        self._endpoint = endpoint
        self._now = time_fn  # wall-clock, used ONLY for compute latency
        self._ring = RingBuffer(int(settings.ring_seconds * settings.sample_rate))
        self._win = int(settings.smart_turn_window_s * settings.sample_rate)
        self._fs = settings.frame_samples
        self._frame_ms = settings.frame_samples / settings.sample_rate * 1000.0
        self._tail = b""        # leftover bytes awaiting a full frame
        self._partial = ""      # latest ASR partial (injected)
        self._audio_ms = 0.0    # audio-sample clock (silence timing)
        self._in_speech = False
        self._silence_start: Optional[float] = None
        self._latched = False   # endpoint already fired for this turn

    def set_partial(self, text: str) -> None:
        """Inject the latest ASR partial. Non-blocking; read at evaluation."""
        self._partial = text or ""

    async def process(self, pcm_bytes: bytes,
                      loop: Optional[asyncio.AbstractEventLoop] = None
                      ) -> List[Decision]:
        """Feed an arbitrary-length chunk of int16 PCM (e.g. the 20 ms Twilio
        frames the VAD gets). It is re-framed into fixed 512-sample frames and
        returns the Decision(s) produced (usually 0 or 1 per call)."""
        loop = loop or asyncio.get_running_loop()
        self._tail += pcm_bytes
        frame_bytes = self._fs * 2  # int16 -> 2 bytes/sample
        out: List[Decision] = []
        while len(self._tail) >= frame_bytes:
            frame = self._tail[:frame_bytes]
            self._tail = self._tail[frame_bytes:]
            dec = await self._process_frame(frame, loop)
            if dec is not None:
                out.append(dec)
        return out

    async def _process_frame(self, frame: bytes,
                             loop: asyncio.AbstractEventLoop
                             ) -> Optional[Decision]:
        self._audio_ms += self._frame_ms
        now = self._audio_ms
        self._ring.extend(np.frombuffer(frame, dtype=np.int16))
        speech = self._vad.process(frame)

        if speech:
            self._in_speech = True
            self._silence_start = None
            self._latched = False
            self._endpoint.reset()
            return None
        if not self._in_speech:
            return None  # leading silence, nothing to end

        if self._silence_start is None:
            self._silence_start = now
        silence_ms = now - self._silence_start
        if silence_ms < self._s.min_silence_ms or self._latched:
            return None

        # --- evaluation: acoustic + lexical concurrently, off the event loop ---
        t0 = self._now()
        window = self._ring.snapshot(self._win)
        partial = self._partial                       # snapshot, never blocks
        ac_fut = loop.run_in_executor(None, self._acoustic.predict, window)
        lex_fut = loop.run_in_executor(None, self._lexical.predict, partial)
        p_ac = await ac_fut
        lex = await lex_fut
        fr = self._fusion.fuse(FusionInput(
            p_ac=p_ac, p_ac_available=True,
            p_lex=lex.p_lex, p_lex_available=bool(partial),
            silence_ms=silence_ms))
        final = self._endpoint.evaluate(fr, silence_ms)
        latency = self._now() - t0

        if final == "ENDPOINT":
            self._latched = True
            self._in_speech = False  # turn done; await next speech onset
        return Decision(
            ts_ms=now, p_ac=p_ac, p_lex=lex.p_lex, p_eou=fr.p_eou,
            decision=final, required_silence_ms=fr.required_silence_ms,
            silence_ms=silence_ms, latency_ms=latency, reason=fr.reason,
            partial_text=partial, lex_reason=lex.reason)
