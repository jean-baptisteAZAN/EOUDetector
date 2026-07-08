import abc
import asyncio
from typing import AsyncIterator, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_wav_mono_16k(path: str, target_sr: int = 16000) -> np.ndarray:
    data, sr = sf.read(path, dtype="int16", always_2d=True)
    mono = data.mean(axis=1).astype(np.int16)
    if sr != target_sr:
        from math import gcd
        g = gcd(sr, target_sr)
        res = resample_poly(mono.astype(np.float32), target_sr // g, sr // g)
        mono = np.clip(np.round(res), -32768, 32767).astype(np.int16)
    return mono


class AudioSource(abc.ABC):
    @abc.abstractmethod
    def frames(self) -> AsyncIterator[bytes]:
        ...


class WavStreamSource(AudioSource):
    def __init__(self, path: str, sample_rate: int = 16000,
                 frame_samples: int = 512, realtime: bool = True,
                 pad_silence_ms: float = 0.0):
        pcm = load_wav_mono_16k(path, sample_rate)
        if pad_silence_ms > 0:
            # Append trailing silence so a finished utterance presents the pause
            # the endpoint logic evaluates (real recordings have one; clipped
            # fixtures often don't).
            n = int(pad_silence_ms / 1000.0 * sample_rate)
            pcm = np.concatenate([pcm, np.zeros(n, dtype=np.int16)])
        self._pcm = pcm
        self._sr = sample_rate
        self._fs = frame_samples
        self._realtime = realtime

    async def frames(self) -> AsyncIterator[bytes]:
        period = self._fs / self._sr
        for start in range(0, len(self._pcm), self._fs):
            chunk = self._pcm[start:start + self._fs]
            if chunk.size < self._fs:
                chunk = np.concatenate(
                    [chunk, np.zeros(self._fs - chunk.size, dtype=np.int16)])
            if self._realtime:
                await asyncio.sleep(period)
            yield chunk.tobytes()


class MicSource(AudioSource):
    def __init__(self, sample_rate: int = 16000, frame_samples: int = 512,
                 device: Optional[int] = None):
        self._sr = sample_rate
        self._fs = frame_samples
        self._device = device
        self._queue: asyncio.Queue = asyncio.Queue()

    async def frames(self) -> AsyncIterator[bytes]:
        import sounddevice as sd
        loop = asyncio.get_running_loop()

        def cb(indata, frames_count, time_info, status):
            pcm = (indata[:, 0] * 32767.0).astype(np.int16).tobytes()
            loop.call_soon_threadsafe(self._queue.put_nowait, pcm)

        with sd.InputStream(samplerate=self._sr, channels=1, dtype="float32",
                            blocksize=self._fs, device=self._device,
                            callback=cb):
            while True:
                yield await self._queue.get()
