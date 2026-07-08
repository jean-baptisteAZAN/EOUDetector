import threading
from typing import List, Tuple
from .base import ASR
from eou_detector.types import Partial


class ScriptedASR(ASR):
    """Deterministic ASR for tests / offline runs.

    script: list of (frame_index, text) or (frame_index, text, is_final),
    sorted by frame_index. latest_partial() returns the newest entry whose
    frame_index <= frames received so far.
    """

    def __init__(self, script: List[Tuple]):
        self._script = sorted(script, key=lambda e: e[0])
        self._count = 0
        self._lock = threading.Lock()

    async def start(self) -> None:
        with self._lock:
            self._count = 0

    def send_audio(self, frame: bytes) -> None:
        with self._lock:
            self._count += 1

    def latest_partial(self) -> Partial:
        with self._lock:
            count = self._count
        text, is_final = "", False
        for entry in self._script:
            if entry[0] <= count:
                text = entry[1]
                is_final = entry[2] if len(entry) > 2 else False
            else:
                break
        return Partial(text=text, is_final=is_final, ts_ms=float(count))

    async def stop(self) -> None:
        return None
