# Semantic EOU Detector Implementation Plan

> Test-driven, task-by-task plan. Each step lists a failing test first, then the
> implementation that makes it pass. Checkbox (`- [ ]`) syntax tracks progress.

**Goal:** Build a Python POC that decides in real time whether a French phone
caller has finished speaking, fusing an acoustic (Smart Turn v3) and a lexical
(turn-detector + FR heuristic) signal on top of a Silero VAD gate, behind clean
swappable interfaces.

**Architecture:** An asyncio orchestrator gates audio through Silero VAD; on each
micro-pause it scores the recent audio buffer (acoustic ONNX) and the latest ASR
partial (lexical), fuses them with bilateral-veto rules into a probability +
decision, and applies a confidence-driven dynamic silence timeout. The lexical
read never blocks the decision. Azure Speech (mirroring the prod NestJS service,
plus a `recognizing` partials handler) is the ASR, behind a swappable interface.

**Tech Stack:** Python 3.11+, asyncio, numpy, soundfile + scipy (wav/resample),
sounddevice (mic), onnxruntime (Smart Turn v3 + turn-detector), silero-vad
(torch), azure-cognitiveservices-speech, huggingface_hub, python-dotenv, pytest +
pytest-asyncio.

## Global Constraints

- Audio everywhere: **16 kHz, mono, 16-bit signed PCM** (`int16`). Frame size =
  **512 samples (32 ms)**.
- Recognition language: **`fr-FR`**.
- ASR credentials come **only** from env vars `AZURE_STT_API_KEY` and
  `AZURE_STT_REGION` (reuse prod values; never hard-code).
- The decision path **must never `await` the ASR**. Lexical input is always a
  snapshot of the latest partial, even if slightly stale.
- Every module sits behind an interface (ABC) and is independently swappable:
  `AudioSource`, `VAD`, `ASR`, `AcousticEOU`, `LexicalEOU`, `Fusion`.
- `Fusion` returns a probability `p_eou` plus a decision, so a `LogisticFusion`
  can replace `RuleFusion` with zero caller change.
- Decisions enum (strings): `"ENDPOINT"` | `"WAIT"`.
- Off-the-shelf models only; no fine-tuning; no telephony/SIP; no prod deploy.
- TDD throughout. Commit after every green task.

---

## File Structure

```
eou_detector/
  __init__.py
  config.py            Settings dataclass + load_settings() from env / .env
  types.py             Partial, LexResult, FusionInput, FusionResult, Decision
  audio/
    __init__.py
    ring_buffer.py     RingBuffer (thread-safe rolling int16 PCM)
    source.py          AudioSource (ABC); WavStreamSource; MicSource
  vad/
    __init__.py
    base.py            VAD (ABC)
    silero_vad.py      SileroVAD
  asr/
    __init__.py
    base.py            ASR (ABC)
    azure_asr.py       AzureSpeechASR (recognizing + recognized)
    scripted_asr.py    ScriptedASR (deterministic test/offline fixture)
  eou/
    __init__.py
    acoustic.py        AcousticEOU (ABC); SmartTurnV3; StubAcousticEOU
    lexical.py         LexicalEOU (ABC); FRHeuristicVeto; TurnDetectorEOU;
                       CompositeLexicalEOU
  fusion/
    __init__.py
    base.py            Fusion (ABC)
    rules.py           RuleFusion
    logistic.py        LogisticFusion
  endpoint/
    __init__.py
    controller.py      EndpointController
  orchestrator.py      Orchestrator (wires everything; asyncio loop)
demo.py
eval/
  __init__.py
  metrics.py           pure metric computations
  harness.py           labelled-clips runner → metrics
  clips/
    fini/              labelled wav (finished)
    pas_fini/          labelled wav (not finished)
tests/
  ...                  one test module per source module
README.md
requirements.txt
.env.example
pyproject.toml
```

---

### Task 1: Project scaffold, config, core types

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore`
- Create: `eou_detector/__init__.py` (empty)
- Create: `eou_detector/config.py`
- Create: `eou_detector/types.py`
- Test: `tests/test_config.py`, `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Settings` dataclass with fields (all with defaults except creds):
    `azure_key: str|None`, `azure_region: str|None`, `lang: str="fr-FR"`,
    `sample_rate: int=16000`, `frame_samples: int=512`, `ring_seconds: float=8.0`,
    `smart_turn_window_s: float=8.0`, `min_silence_ms: float=200.0`,
    `veto_lex: float=0.25`, `veto_ac: float=0.25`, `high: float=0.85`,
    `mid: float=0.5`, `short_ms: float=200.0`, `med_ms: float=600.0`,
    `long_ms: float=1200.0`, `max_silence_ms: float=2000.0`,
    `vad_threshold: float=0.5`.
  - `load_settings(env: Mapping[str,str]|None=None) -> Settings` (reads
    `AZURE_STT_API_KEY`, `AZURE_STT_REGION`, optional `EOU_LANG`; falls back to
    `os.environ`; loads `.env` via python-dotenv if present).
  - `Partial(text: str, is_final: bool=False, ts_ms: float=0.0)` (dataclass).
  - `LexResult(p_lex: float, veto: bool, reason: str)` (dataclass).
  - `FusionInput(p_ac: float, p_ac_available: bool, p_lex: float,
    p_lex_available: bool, silence_ms: float)` (dataclass).
  - `FusionResult(p_eou: float, decision: str, required_silence_ms: float,
    reason: str)` (dataclass).
  - `Decision(ts_ms: float, p_ac: float, p_lex: float, p_eou: float,
    decision: str, required_silence_ms: float, silence_ms: float,
    latency_ms: float, reason: str, partial_text: str)` (dataclass).

- [ ] **Step 1: Create `requirements.txt`**

```
numpy>=1.26
soundfile>=0.12
scipy>=1.11
sounddevice>=0.4
onnxruntime>=1.17
silero-vad>=5.1
torch>=2.2
azure-cognitiveservices-speech>=1.42
huggingface_hub>=0.23
transformers>=4.40
python-dotenv>=1.0
pytest>=8.0
pytest-asyncio>=0.23
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "eou-detector"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: Create `.env.example` and `.gitignore`**

`.env.example`:
```
# Reuse the production CareCallHouseMade Azure Speech credentials.
AZURE_STT_API_KEY=
AZURE_STT_REGION=
# Optional overrides
EOU_LANG=fr-FR
```

`.gitignore`:
```
__pycache__/
*.pyc
.env
.venv/
models/
*.onnx
```

- [ ] **Step 4: Write the failing tests**

`tests/test_types.py`:
```python
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
```

`tests/test_config.py`:
```python
from eou_detector.config import load_settings, Settings


def test_defaults_present():
    s = load_settings(env={})
    assert isinstance(s, Settings)
    assert s.lang == "fr-FR"
    assert s.sample_rate == 16000
    assert s.frame_samples == 512
    assert s.azure_key is None


def test_reads_azure_env():
    s = load_settings(env={"AZURE_STT_API_KEY": "k", "AZURE_STT_REGION": "westeurope"})
    assert s.azure_key == "k"
    assert s.azure_region == "westeurope"


def test_lang_override():
    s = load_settings(env={"EOU_LANG": "fr-CA"})
    assert s.lang == "fr-CA"
```

- [ ] **Step 5: Run tests, verify they fail**

Run: `pytest tests/test_types.py tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: eou_detector.types`.

- [ ] **Step 6: Implement `eou_detector/types.py`**

```python
from dataclasses import dataclass


@dataclass
class Partial:
    text: str
    is_final: bool = False
    ts_ms: float = 0.0


@dataclass
class LexResult:
    p_lex: float
    veto: bool
    reason: str


@dataclass
class FusionInput:
    p_ac: float
    p_ac_available: bool
    p_lex: float
    p_lex_available: bool
    silence_ms: float


@dataclass
class FusionResult:
    p_eou: float
    decision: str  # "ENDPOINT" | "WAIT"
    required_silence_ms: float
    reason: str


@dataclass
class Decision:
    ts_ms: float
    p_ac: float
    p_lex: float
    p_eou: float
    decision: str
    required_silence_ms: float
    silence_ms: float
    latency_ms: float
    reason: str
    partial_text: str
```

- [ ] **Step 7: Implement `eou_detector/config.py`**

```python
import os
from dataclasses import dataclass
from typing import Mapping, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # load .env into os.environ if present
except Exception:
    pass


@dataclass
class Settings:
    azure_key: Optional[str] = None
    azure_region: Optional[str] = None
    lang: str = "fr-FR"
    sample_rate: int = 16000
    frame_samples: int = 512
    ring_seconds: float = 8.0
    smart_turn_window_s: float = 8.0
    min_silence_ms: float = 200.0
    veto_lex: float = 0.25
    veto_ac: float = 0.25
    high: float = 0.85
    mid: float = 0.5
    short_ms: float = 200.0
    med_ms: float = 600.0
    long_ms: float = 1200.0
    max_silence_ms: float = 2000.0
    vad_threshold: float = 0.5


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    e = env if env is not None else os.environ
    return Settings(
        azure_key=e.get("AZURE_STT_API_KEY"),
        azure_region=e.get("AZURE_STT_REGION"),
        lang=e.get("EOU_LANG", "fr-FR"),
    )
```

- [ ] **Step 8: Run tests, verify they pass**

Run: `pytest tests/test_types.py tests/test_config.py -v`
Expected: PASS (6 tests).

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml requirements.txt .env.example .gitignore eou_detector/ tests/test_config.py tests/test_types.py
git commit -m "feat: project scaffold, config, core types"
```

---

### Task 2: RingBuffer (thread-safe rolling PCM)

**Files:**
- Create: `eou_detector/audio/__init__.py` (empty), `eou_detector/audio/ring_buffer.py`
- Test: `tests/test_ring_buffer.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RingBuffer(capacity_samples: int)` with:
  - `extend(samples: np.ndarray) -> None` (accepts `int16` 1-D array)
  - `snapshot(n_samples: int) -> np.ndarray` (last `n` samples, `int16`; if fewer
    available, left-pads with zeros to length `n`)
  - `__len__() -> int` (samples currently stored, capped at capacity)

- [ ] **Step 1: Write the failing test**

`tests/test_ring_buffer.py`:
```python
import numpy as np
from eou_detector.audio.ring_buffer import RingBuffer


def test_extend_and_snapshot_recent():
    rb = RingBuffer(capacity_samples=10)
    rb.extend(np.arange(1, 6, dtype=np.int16))   # 1..5
    snap = rb.snapshot(3)
    assert snap.tolist() == [3, 4, 5]


def test_snapshot_left_pads_when_short():
    rb = RingBuffer(capacity_samples=10)
    rb.extend(np.array([7, 8], dtype=np.int16))
    snap = rb.snapshot(4)
    assert snap.tolist() == [0, 0, 7, 8]


def test_overwrites_when_over_capacity():
    rb = RingBuffer(capacity_samples=4)
    rb.extend(np.arange(1, 7, dtype=np.int16))   # 1..6, only last 4 kept
    assert rb.snapshot(4).tolist() == [3, 4, 5, 6]
    assert len(rb) == 4
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_ring_buffer.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `eou_detector/audio/ring_buffer.py`**

```python
import threading
import numpy as np


class RingBuffer:
    """Fixed-capacity rolling buffer of int16 PCM, safe across threads."""

    def __init__(self, capacity_samples: int):
        self._cap = int(capacity_samples)
        self._buf = np.zeros(self._cap, dtype=np.int16)
        self._size = 0
        self._lock = threading.Lock()

    def extend(self, samples: np.ndarray) -> None:
        s = np.asarray(samples, dtype=np.int16).reshape(-1)
        with self._lock:
            n = s.size
            if n >= self._cap:
                self._buf[:] = s[-self._cap:]
                self._size = self._cap
                return
            self._buf = np.roll(self._buf, -n)
            self._buf[-n:] = s
            self._size = min(self._cap, self._size + n)

    def snapshot(self, n_samples: int) -> np.ndarray:
        n = int(n_samples)
        with self._lock:
            avail = min(self._size, self._cap)
            data = self._buf[-avail:] if avail else np.zeros(0, dtype=np.int16)
        if data.size >= n:
            return data[-n:].copy()
        out = np.zeros(n, dtype=np.int16)
        if data.size:
            out[-data.size:] = data
        return out

    def __len__(self) -> int:
        with self._lock:
            return self._size
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_ring_buffer.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add eou_detector/audio/ tests/test_ring_buffer.py
git commit -m "feat: thread-safe rolling PCM ring buffer"
```

---

### Task 3: AudioSource — WavStreamSource + MicSource

**Files:**
- Create: `eou_detector/audio/source.py`
- Test: `tests/test_wav_source.py`

**Interfaces:**
- Consumes: `Settings` (for `sample_rate`, `frame_samples`).
- Produces:
  - `AudioSource` ABC with `async def frames(self) -> AsyncIterator[bytes]` (each
    yielded item is `frame_samples*2` bytes of int16 PCM).
  - `WavStreamSource(path: str, sample_rate=16000, frame_samples=512,
    realtime: bool=True)` — reads a wav, downmixes to mono, resamples to
    `sample_rate`, yields fixed-size frames; when `realtime=True`, paces with
    `asyncio.sleep(frame_samples/sample_rate)`; pads the final short frame.
  - `MicSource(sample_rate=16000, frame_samples=512, device=None)` — sounddevice
    input stream feeding an `asyncio.Queue` from its callback thread.
  - Helper `load_wav_mono_16k(path, target_sr) -> np.ndarray[int16]` (exported for
    reuse by the eval harness).

- [ ] **Step 1: Write the failing test**

`tests/test_wav_source.py`:
```python
import numpy as np
import soundfile as sf
import pytest
from eou_detector.audio.source import WavStreamSource, load_wav_mono_16k


def _write_wav(path, data, sr):
    sf.write(path, data.astype(np.int16), sr, subtype="PCM_16")


def test_load_downmix_and_resample(tmp_path):
    p = tmp_path / "stereo8k.wav"
    n = 8000  # 1 s @ 8k
    stereo = np.stack([np.full(n, 100), np.full(n, 300)], axis=1).astype(np.int16)
    _write_wav(p, stereo, 8000)
    mono16 = load_wav_mono_16k(str(p), 16000)
    assert mono16.dtype == np.int16
    assert abs(mono16.size - 16000) <= 2          # ~1 s @ 16k
    assert 150 <= int(np.median(mono16)) <= 250    # mean of 100 & 300


@pytest.mark.asyncio
async def test_frames_fixed_size_and_count(tmp_path):
    p = tmp_path / "mono16k.wav"
    _write_wav(p, np.zeros(512 * 3 + 100, dtype=np.int16), 16000)
    src = WavStreamSource(str(p), sample_rate=16000, frame_samples=512,
                          realtime=False)
    frames = [f async for f in src.frames()]
    assert all(len(f) == 512 * 2 for f in frames)   # int16 → 2 bytes/sample
    assert len(frames) == 4                          # 3 full + 1 padded
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_wav_source.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `eou_detector/audio/source.py`**

```python
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
                 frame_samples: int = 512, realtime: bool = True):
        self._pcm = load_wav_mono_16k(path, sample_rate)
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
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_wav_source.py -v`
Expected: PASS (2 tests). (MicSource is not unit-tested — it needs hardware.)

- [ ] **Step 5: Commit**

```bash
git add eou_detector/audio/source.py tests/test_wav_source.py
git commit -m "feat: wav-stream and mic audio sources"
```

---

### Task 4: VAD — Silero behind interface

**Files:**
- Create: `eou_detector/vad/__init__.py` (empty), `eou_detector/vad/base.py`,
  `eou_detector/vad/silero_vad.py`
- Test: `tests/test_vad.py`

**Interfaces:**
- Consumes: `Settings` (`sample_rate`, `vad_threshold`).
- Produces:
  - `VAD` ABC with `process(self, frame: bytes) -> bool` (True = speech) and
    `reset(self) -> None`.
  - `SileroVAD(sample_rate=16000, threshold=0.5)` implementing `VAD` via the
    `silero-vad` package model.

- [ ] **Step 1: Write the failing test**

`tests/test_vad.py`:
```python
import numpy as np
import pytest
from eou_detector.vad.base import VAD


class _ProbVAD(VAD):
    """Test double over the same interface (no model download)."""
    def __init__(self, probs, threshold=0.5):
        self._probs, self._i, self._t = probs, 0, threshold
    def process(self, frame: bytes) -> bool:
        p = self._probs[min(self._i, len(self._probs) - 1)]
        self._i += 1
        return p >= self._t
    def reset(self) -> None:
        self._i = 0


def test_interface_threshold_behaviour():
    v = _ProbVAD([0.1, 0.9, 0.4], threshold=0.5)
    frame = np.zeros(512, dtype=np.int16).tobytes()
    assert v.process(frame) is False
    assert v.process(frame) is True
    assert v.process(frame) is False


@pytest.mark.integration
def test_silero_loads_and_scores_silence():
    pytest.importorskip("silero_vad")
    from eou_detector.vad.silero_vad import SileroVAD
    v = SileroVAD(sample_rate=16000, threshold=0.5)
    silence = np.zeros(512, dtype=np.int16).tobytes()
    assert v.process(silence) is False
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_vad.py::test_interface_threshold_behaviour -v`
Expected: FAIL — `eou_detector.vad.base` missing.

- [ ] **Step 3: Implement `eou_detector/vad/base.py`**

```python
import abc


class VAD(abc.ABC):
    @abc.abstractmethod
    def process(self, frame: bytes) -> bool:
        """Return True if the frame contains speech."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state between utterances/streams."""
```

- [ ] **Step 4: Implement `eou_detector/vad/silero_vad.py`**

```python
import numpy as np
from .base import VAD


class SileroVAD(VAD):
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        import torch
        from silero_vad import load_silero_vad
        self._torch = torch
        self._model = load_silero_vad(onnx=False)
        self._sr = sample_rate
        self._threshold = threshold

    def process(self, frame: bytes) -> bool:
        pcm = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        t = self._torch.from_numpy(pcm)
        with self._torch.no_grad():
            prob = float(self._model(t, self._sr).item())
        return prob >= self._threshold

    def reset(self) -> None:
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()
```

- [ ] **Step 5: Run unit test, verify it passes**

Run: `pytest tests/test_vad.py::test_interface_threshold_behaviour -v`
Expected: PASS.

- [ ] **Step 6: Run the integration test (requires torch + silero-vad installed)**

Run: `pytest tests/test_vad.py -v -m integration`
Expected: PASS (silence → False). If deps absent, it is skipped, not failed.

- [ ] **Step 7: Commit**

```bash
git add eou_detector/vad/ tests/test_vad.py
git commit -m "feat: Silero VAD behind swappable interface"
```

---

### Task 5: ASR — interface, Azure (recognizing + recognized), scripted fixture

**Files:**
- Create: `eou_detector/asr/__init__.py` (empty), `eou_detector/asr/base.py`,
  `eou_detector/asr/azure_asr.py`, `eou_detector/asr/scripted_asr.py`
- Test: `tests/test_scripted_asr.py`, `tests/test_azure_asr.py`

**Interfaces:**
- Consumes: `Settings` (`azure_key`, `azure_region`, `lang`, `sample_rate`),
  `Partial` type.
- Produces:
  - `ASR` ABC:
    - `async def start(self) -> None`
    - `def send_audio(self, frame: bytes) -> None` (non-blocking push)
    - `def latest_partial(self) -> Partial` (snapshot; never blocks)
    - `async def stop(self) -> None`
  - `AzureSpeechASR(settings: Settings, phrase_list: list[str]|None=None,
    segmentation_silence_ms: int=500)` — mirrors prod `azure-stt.service.ts`
    (PCM 16k/16/1 push stream, `fr-FR`, Detailed output, segmentation timeout),
    **plus** a `recognizing` handler updating `latest_partial`.
  - `ScriptedASR(script: list[tuple[int, str]])` — deterministic fixture: each
    `send_audio` advances a frame counter; `latest_partial` returns the text of
    the last script entry whose frame index `<=` counter. `final_at`/`is_final`
    handled by an optional 3rd tuple element.

- [ ] **Step 1: Write the failing test for the scripted fixture**

`tests/test_scripted_asr.py`:
```python
import asyncio
import numpy as np
from eou_detector.asr.scripted_asr import ScriptedASR

FRAME = np.zeros(512, dtype=np.int16).tobytes()


def test_partial_advances_with_frames():
    asr = ScriptedASR(script=[(0, ""), (2, "je m'appelle"), (4, "je m'appelle martin")])
    asyncio.run(asr.start())
    assert asr.latest_partial().text == ""
    asr.send_audio(FRAME); asr.send_audio(FRAME)      # counter = 2
    assert asr.latest_partial().text == "je m'appelle"
    asr.send_audio(FRAME); asr.send_audio(FRAME)      # counter = 4
    assert asr.latest_partial().text == "je m'appelle martin"


def test_latest_partial_never_blocks_before_start():
    asr = ScriptedASR(script=[(0, "")])
    assert asr.latest_partial().text == ""
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_scripted_asr.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `eou_detector/asr/base.py`**

```python
import abc
from eou_detector.types import Partial


class ASR(abc.ABC):
    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    def send_audio(self, frame: bytes) -> None: ...

    @abc.abstractmethod
    def latest_partial(self) -> Partial: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...
```

- [ ] **Step 4: Implement `eou_detector/asr/scripted_asr.py`**

```python
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
```

- [ ] **Step 5: Run test, verify it passes**

Run: `pytest tests/test_scripted_asr.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Write the Azure handler-logic test (no network — test the partial holder directly)**

`tests/test_azure_asr.py`:
```python
import pytest
from eou_detector.config import Settings
from eou_detector.asr.azure_asr import AzureSpeechASR


def test_recognizing_updates_latest_partial():
    asr = AzureSpeechASR(Settings(azure_key="x", azure_region="r"))
    # Simulate the SDK 'recognizing' callback payload.
    asr._on_recognizing_text("le vingt", is_final=False)
    assert asr.latest_partial().text == "le vingt"
    assert asr.latest_partial().is_final is False
    asr._on_recognizing_text("le vingt juin", is_final=True)
    p = asr.latest_partial()
    assert p.text == "le vingt juin" and p.is_final is True


def test_start_without_credentials_raises():
    asr = AzureSpeechASR(Settings(azure_key=None, azure_region=None))
    with pytest.raises(RuntimeError):
        import asyncio
        asyncio.run(asr.start())
```

- [ ] **Step 7: Run test, verify it fails**

Run: `pytest tests/test_azure_asr.py -v`
Expected: FAIL — module missing.

- [ ] **Step 8: Implement `eou_detector/asr/azure_asr.py`**

```python
import threading
from typing import List, Optional
from .base import ASR
from eou_detector.config import Settings
from eou_detector.types import Partial


class AzureSpeechASR(ASR):
    """Azure Speech STT mirroring the prod CareCallHouseMade config, with an
    added `recognizing` handler to surface real-time partials for the lexical
    branch. Credentials come from Settings (AZURE_STT_API_KEY / _REGION)."""

    def __init__(self, settings: Settings,
                 phrase_list: Optional[List[str]] = None,
                 segmentation_silence_ms: int = 500):
        self._s = settings
        self._phrase_list = phrase_list or []
        self._seg_ms = segmentation_silence_ms
        self._partial = Partial(text="")
        self._lock = threading.Lock()
        self._recognizer = None
        self._push_stream = None

    # --- handler logic, unit-testable without the SDK ---
    def _on_recognizing_text(self, text: str, is_final: bool) -> None:
        with self._lock:
            self._partial = Partial(text=text, is_final=is_final)

    def latest_partial(self) -> Partial:
        with self._lock:
            return self._partial

    async def start(self) -> None:
        if not self._s.azure_key or not self._s.azure_region:
            raise RuntimeError("AZURE_STT_API_KEY / AZURE_STT_REGION not set")
        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(
            subscription=self._s.azure_key, region=self._s.azure_region)
        speech_config.speech_recognition_language = self._s.lang
        speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
            str(self._seg_ms))
        speech_config.output_format = speechsdk.OutputFormat.Detailed

        fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self._s.sample_rate, bits_per_sample=16, channels=1)
        self._push_stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config)

        if self._phrase_list:
            pl = speechsdk.PhraseListGrammar.from_recognizer(self._recognizer)
            for ph in self._phrase_list:
                pl.addPhrase(ph)

        self._recognizer.recognizing.connect(
            lambda evt: self._on_recognizing_text(evt.result.text, is_final=False))
        self._recognizer.recognized.connect(
            lambda evt: self._on_recognizing_text(evt.result.text, is_final=True))
        self._recognizer.start_continuous_recognition_async().get()

    def send_audio(self, frame: bytes) -> None:
        if self._push_stream is not None:
            self._push_stream.write(frame)

    async def stop(self) -> None:
        if self._recognizer is not None:
            self._recognizer.stop_continuous_recognition_async().get()
        if self._push_stream is not None:
            self._push_stream.close()
```

- [ ] **Step 9: Run tests, verify they pass**

Run: `pytest tests/test_azure_asr.py -v`
Expected: PASS (2 tests).

- [ ] **Step 10: Commit**

```bash
git add eou_detector/asr/ tests/test_scripted_asr.py tests/test_azure_asr.py
git commit -m "feat: ASR interface, Azure (recognizing+recognized), scripted fixture"
```

---

### Task 6: Acoustic EOU — Smart Turn v3 (ONNX) + stub

**Files:**
- Create: `eou_detector/eou/__init__.py` (empty), `eou_detector/eou/acoustic.py`
- Create: `scripts/fetch_models.py`
- Test: `tests/test_acoustic.py`

**Interfaces:**
- Consumes: `Settings` (`sample_rate`, `smart_turn_window_s`).
- Produces:
  - `AcousticEOU` ABC: `predict(self, pcm: np.ndarray) -> float` (int16 1-D in,
    `p_ac` in [0,1] out).
  - `StubAcousticEOU(value: float=0.5)` returning a constant (for tests/fallback).
  - `SmartTurnV3(model_path: str, sample_rate=16000, window_s=8.0)` — onnxruntime
    wrapper. Preprocess = mono float32, right-trim/left-pad to `window_s`. Input/
    output tensor names are discovered from the session (do not hard-code).

- [ ] **Step 1: Write `scripts/fetch_models.py` (model download helper)**

```python
"""Download off-the-shelf ONNX models into ./models.

Usage: python scripts/fetch_models.py
Verifies file names by listing the repo; prints the resolved local paths.
"""
import os
from huggingface_hub import hf_hub_download, list_repo_files

MODELS = {
    "smart_turn_v3": ("pipecat-ai/smart-turn-v3", None),   # filename auto-detected
}


def main():
    os.makedirs("models", exist_ok=True)
    for key, (repo, fname) in MODELS.items():
        files = [f for f in list_repo_files(repo) if f.endswith(".onnx")]
        target = fname or (files[0] if files else None)
        if not target:
            print(f"[{key}] no .onnx found in {repo}; files={files}")
            continue
        path = hf_hub_download(repo, target, local_dir="models")
        print(f"[{key}] {repo}/{target} -> {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing test**

`tests/test_acoustic.py`:
```python
import os
import numpy as np
import pytest
from eou_detector.eou.acoustic import AcousticEOU, StubAcousticEOU


def test_stub_returns_constant_in_range():
    a = StubAcousticEOU(0.7)
    p = a.predict(np.zeros(16000, dtype=np.int16))
    assert isinstance(a, AcousticEOU)
    assert 0.0 <= p <= 1.0 and p == 0.7


@pytest.mark.integration
def test_smart_turn_runs_if_model_present():
    path = os.environ.get("SMART_TURN_ONNX", "models/smart-turn-v3.onnx")
    if not os.path.exists(path):
        pytest.skip("Smart Turn v3 ONNX not downloaded")
    from eou_detector.eou.acoustic import SmartTurnV3
    a = SmartTurnV3(path)
    p = a.predict(np.zeros(16000 * 2, dtype=np.int16))
    assert 0.0 <= p <= 1.0
```

- [ ] **Step 3: Run test, verify it fails**

Run: `pytest tests/test_acoustic.py::test_stub_returns_constant_in_range -v`
Expected: FAIL — module missing.

- [ ] **Step 4: Implement `eou_detector/eou/acoustic.py`**

```python
import abc
import numpy as np


class AcousticEOU(abc.ABC):
    @abc.abstractmethod
    def predict(self, pcm: np.ndarray) -> float:
        """int16 PCM window -> probability the speaker finished (p_ac)."""


class StubAcousticEOU(AcousticEOU):
    def __init__(self, value: float = 0.5):
        self._value = float(value)

    def predict(self, pcm: np.ndarray) -> float:
        return self._value


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class SmartTurnV3(AcousticEOU):
    def __init__(self, model_path: str, sample_rate: int = 16000,
                 window_s: float = 8.0):
        import onnxruntime as ort
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self._in_name = self._sess.get_inputs()[0].name
        self._out_name = self._sess.get_outputs()[0].name
        self._sr = sample_rate
        self._win = int(window_s * sample_rate)

    def _prep(self, pcm: np.ndarray) -> np.ndarray:
        x = np.asarray(pcm, dtype=np.float32) / 32768.0
        if x.size >= self._win:
            x = x[-self._win:]
        else:
            x = np.concatenate([np.zeros(self._win - x.size, dtype=np.float32), x])
        return x[np.newaxis, :]  # (1, win)

    def predict(self, pcm: np.ndarray) -> float:
        feats = self._prep(pcm)
        out = self._sess.run([self._out_name], {self._in_name: feats})[0]
        val = float(np.asarray(out).reshape(-1)[0])
        # Model card: output is a completion probability in [0,1]; if a raw
        # logit is emitted instead, squash it.
        return val if 0.0 <= val <= 1.0 else float(_sigmoid(val))
```

> **Implementer note:** after `python scripts/fetch_models.py`, run
> `python -c "import onnxruntime as ort; s=ort.InferenceSession('models/smart-turn-v3.onnx'); print([i.name for i in s.get_inputs()], [i.shape for i in s.get_inputs()], [o.name for o in s.get_outputs()])"`
> and confirm the input is a single waveform tensor `(1, N)`. If the model card
> specifies log-mel features or a fixed N, adjust `_prep` to match and update the
> integration test's input length. Keep `predict`'s [0,1] contract unchanged.

- [ ] **Step 5: Run unit test, verify it passes**

Run: `pytest tests/test_acoustic.py::test_stub_returns_constant_in_range -v`
Expected: PASS.

- [ ] **Step 6: Fetch the model and run the integration test**

Run: `python scripts/fetch_models.py && pytest tests/test_acoustic.py -v -m integration`
Expected: PASS (or SKIP if download unavailable).

- [ ] **Step 7: Commit**

```bash
git add eou_detector/eou/__init__.py eou_detector/eou/acoustic.py scripts/fetch_models.py tests/test_acoustic.py
git commit -m "feat: acoustic EOU (Smart Turn v3 ONNX) + stub"
```

---

### Task 7: Lexical EOU — FR heuristic veto + turn-detector + composite

**Files:**
- Create: `eou_detector/eou/lexical.py`
- Test: `tests/test_lexical.py`

**Interfaces:**
- Consumes: `LexResult` type.
- Produces:
  - `LexicalEOU` ABC: `predict(self, text: str) -> LexResult`.
  - `FRHeuristicVeto()` with `check(self, text: str) -> tuple[bool, str]` — returns
    `(veto, reason)`; `veto=True` means "clearly mid-utterance" (spelling in
    progress, trailing connector/hesitation, open number/date). Also usable as a
    standalone `LexicalEOU` (`predict` maps veto→low `p_lex`).
  - `TurnDetectorEOU(model_path, tokenizer_name)` — ONNX turn-detector → `p_lex`.
  - `CompositeLexicalEOU(model: LexicalEOU|None, heuristic: FRHeuristicVeto)` —
    `p_lex` from the model (or 0.5 if no model); if heuristic vetoes, force
    `p_lex = min(p_lex, 0.1)` and `veto=True`.

- [ ] **Step 1: Write the failing test (heuristic is the TDD core)**

`tests/test_lexical.py`:
```python
from eou_detector.eou.lexical import (
    FRHeuristicVeto, CompositeLexicalEOU, LexicalEOU,
)


def test_spelling_in_progress_vetoes():
    h = FRHeuristicVeto()
    assert h.check("mon nom c'est m a r")[0] is True
    assert h.check("M. A. R")[0] is True


def test_trailing_hesitation_vetoes():
    h = FRHeuristicVeto()
    assert h.check("alors je voudrais euh")[0] is True
    assert h.check("c'est le numero zero six")[0] is True   # open number run


def test_complete_sentence_not_vetoed():
    h = FRHeuristicVeto()
    assert h.check("oui c'est exact")[0] is False
    assert h.check("je voudrais prendre rendez-vous demain")[0] is False


def test_composite_without_model_uses_heuristic():
    lex = CompositeLexicalEOU(model=None, heuristic=FRHeuristicVeto())
    assert isinstance(lex, LexicalEOU)
    r_open = lex.predict("m a r")
    assert r_open.veto is True and r_open.p_lex <= 0.1
    r_done = lex.predict("oui c'est exact")
    assert r_done.veto is False and r_done.p_lex == 0.5


def test_composite_model_score_passthrough_when_no_veto():
    class _M(LexicalEOU):
        def predict(self, text): 
            from eou_detector.types import LexResult
            return LexResult(p_lex=0.8, veto=False, reason="model")
    lex = CompositeLexicalEOU(model=_M(), heuristic=FRHeuristicVeto())
    assert lex.predict("oui c'est exact").p_lex == 0.8
    assert lex.predict("m a r").p_lex <= 0.1   # heuristic still vetoes
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_lexical.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `eou_detector/eou/lexical.py`**

```python
import abc
import re
from typing import Optional, Tuple
from eou_detector.types import LexResult

# French hesitation / connector words that signal an unfinished turn.
_TRAILING_INCOMPLETE = {
    "euh", "heu", "hum", "et", "ou", "donc", "alors", "mais", "car", "puis",
    "que", "qui", "de", "du", "le", "la", "les", "un", "une", "mon", "ma",
    "mes", "je", "j", "c'est", "a", "à", "au", "aux", "pour", "avec", "dans",
}
# Number words: a run ending here is likely still being dictated.
_NUMBER_WORDS = {
    "zero", "zéro", "un", "une", "deux", "trois", "quatre", "cinq", "six",
    "sept", "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze",
    "quinze", "seize", "vingt", "trente", "quarante", "cinquante", "soixante",
    "cent", "cents", "mille",
}


def _tokens(text: str):
    return re.findall(r"[a-zàâäéèêëïîôöùûüç']+|\d+", text.lower().strip())


class LexicalEOU(abc.ABC):
    @abc.abstractmethod
    def predict(self, text: str) -> LexResult: ...


class FRHeuristicVeto(LexicalEOU):
    """Cheap French rules that veto endpointing while the caller is clearly
    mid-utterance: spelling a name, dictating a number/date, or trailing on a
    connector/hesitation."""

    def check(self, text: str) -> Tuple[bool, str]:
        raw = text.strip()
        if not raw:
            return False, "empty"
        toks = _tokens(raw)
        if not toks:
            return False, "no_tokens"

        # Spelling in progress: >=2 trailing single-letter tokens (m a r / M. A. R).
        singles = 0
        for t in reversed(toks):
            if len(t) == 1 and t.isalpha():
                singles += 1
            else:
                break
        if singles >= 2:
            return True, "spelling_in_progress"

        last = toks[-1]
        if last in _TRAILING_INCOMPLETE:
            return True, "trailing_connector"
        # Open number run: ends on a number word and no terminal punctuation.
        if (last in _NUMBER_WORDS or last.isdigit()) and raw[-1] not in ".?!":
            return True, "open_number_run"
        return False, "complete"

    def predict(self, text: str) -> LexResult:
        veto, reason = self.check(text)
        return LexResult(p_lex=(0.05 if veto else 0.5), veto=veto, reason=reason)


class TurnDetectorEOU(LexicalEOU):
    """ONNX multilingual turn-detector → P(end-of-turn) for the partial text."""

    def __init__(self, model_path: str, tokenizer_name: str):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])
        self._tok = AutoTokenizer.from_pretrained(tokenizer_name)

    def predict(self, text: str) -> LexResult:
        import numpy as np
        enc = self._tok(text, return_tensors="np", truncation=True, max_length=128)
        feeds = {i.name: enc[i.name] for i in self._sess.get_inputs()
                 if i.name in enc}
        out = self._sess.run(None, feeds)[0]
        arr = np.asarray(out).reshape(-1)
        if arr.size == 1:
            p = float(1.0 / (1.0 + np.exp(-arr[0])))
        else:
            e = np.exp(arr - arr.max())
            p = float((e / e.sum())[-1])
        return LexResult(p_lex=p, veto=False, reason="turn_detector")


class CompositeLexicalEOU(LexicalEOU):
    def __init__(self, model: Optional[LexicalEOU], heuristic: FRHeuristicVeto):
        self._model = model
        self._heuristic = heuristic

    def predict(self, text: str) -> LexResult:
        veto, vreason = self._heuristic.check(text)
        if self._model is not None:
            base = self._model.predict(text)
            p_lex, reason = base.p_lex, base.reason
        else:
            p_lex, reason = 0.5, "no_model"
        if veto:
            return LexResult(p_lex=min(p_lex, 0.1), veto=True, reason=vreason)
        return LexResult(p_lex=p_lex, veto=False, reason=reason)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_lexical.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add eou_detector/eou/lexical.py tests/test_lexical.py
git commit -m "feat: lexical EOU (FR heuristic veto + turn-detector + composite)"
```

---

### Task 8: Fusion — RuleFusion (bilateral veto) + LogisticFusion swap

**Files:**
- Create: `eou_detector/fusion/__init__.py` (empty), `eou_detector/fusion/base.py`,
  `eou_detector/fusion/rules.py`, `eou_detector/fusion/logistic.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: `Settings`, `FusionInput`, `FusionResult`.
- Produces:
  - `Fusion` ABC: `fuse(self, x: FusionInput) -> FusionResult`.
  - `RuleFusion(settings: Settings)` — bilateral veto; always returns `p_eou` +
    decision + `required_silence_ms` + reason.
  - `LogisticFusion(settings, weights: dict)` — same interface, `p_eou` from a
    logistic on `[p_ac, p_lex, p_ac*p_lex, silence_ms_norm]`; reuses the same
    silence-policy mapping as `RuleFusion` (shared helper `required_silence`).

- [ ] **Step 1: Write the failing test (truth table)**

`tests/test_fusion.py`:
```python
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
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_fusion.py -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Implement `eou_detector/fusion/base.py`**

```python
import abc
from eou_detector.config import Settings
from eou_detector.types import FusionInput, FusionResult


class Fusion(abc.ABC):
    @abc.abstractmethod
    def fuse(self, x: FusionInput) -> FusionResult: ...


def required_silence(p_eou: float, s: Settings) -> float:
    """Confidence -> how much trailing silence we demand before endpointing."""
    if p_eou >= s.high:
        return s.short_ms
    if p_eou >= s.mid:
        return s.med_ms
    return s.long_ms
```

- [ ] **Step 4: Implement `eou_detector/fusion/rules.py`**

```python
from .base import Fusion, required_silence
from eou_detector.config import Settings
from eou_detector.types import FusionInput, FusionResult


class RuleFusion(Fusion):
    def __init__(self, settings: Settings):
        self._s = settings

    def fuse(self, x: FusionInput) -> FusionResult:
        s = self._s
        # 1) Bilateral veto on available signals.
        if x.p_lex_available and x.p_lex < s.veto_lex:
            p = min(x.p_lex, x.p_ac if x.p_ac_available else x.p_lex)
            return FusionResult(p, "WAIT", s.max_silence_ms, "lexical_veto")
        if x.p_ac_available and x.p_ac < s.veto_ac:
            p = min(x.p_ac, x.p_lex if x.p_lex_available else x.p_ac)
            return FusionResult(p, "WAIT", s.max_silence_ms, "acoustic_veto")

        # 2) Combine available signals into p_eou.
        if x.p_ac_available and x.p_lex_available:
            p_eou = 0.5 * x.p_ac + 0.5 * x.p_lex
        elif x.p_ac_available:
            p_eou = x.p_ac
        elif x.p_lex_available:
            p_eou = x.p_lex
        else:
            p_eou = 0.0

        # 3) Both strongly confident -> endpoint immediately (short silence).
        if (x.p_ac_available and x.p_lex_available
                and x.p_ac >= s.high and x.p_lex >= s.high):
            return FusionResult(p_eou, "ENDPOINT", s.short_ms, "both_confident")

        req = required_silence(p_eou, s)
        decision = "ENDPOINT" if (p_eou >= s.mid and x.silence_ms >= req) else "WAIT"
        reason = "score_endpoint" if decision == "ENDPOINT" else "score_wait"
        return FusionResult(p_eou, decision, req, reason)
```

- [ ] **Step 5: Implement `eou_detector/fusion/logistic.py`**

```python
import math
from .base import Fusion, required_silence
from eou_detector.config import Settings
from eou_detector.types import FusionInput, FusionResult


class LogisticFusion(Fusion):
    """Drop-in replacement for RuleFusion once weights are learned. Same
    interface, same silence policy; only the p_eou model differs."""

    def __init__(self, settings: Settings, weights: dict):
        self._s = settings
        self._w = weights

    def fuse(self, x: FusionInput) -> FusionResult:
        s, w = self._s, self._w
        sil_norm = min(x.silence_ms / s.max_silence_ms, 1.0)
        z = (w.get("bias", 0.0)
             + w.get("p_ac", 0.0) * (x.p_ac if x.p_ac_available else 0.0)
             + w.get("p_lex", 0.0) * (x.p_lex if x.p_lex_available else 0.0)
             + w.get("inter", 0.0) * x.p_ac * x.p_lex
             + w.get("sil", 0.0) * sil_norm)
        p_eou = 1.0 / (1.0 + math.exp(-z))
        req = required_silence(p_eou, s)
        decision = "ENDPOINT" if (p_eou >= s.mid and x.silence_ms >= req) else "WAIT"
        return FusionResult(p_eou, decision, req, "logistic")
```

- [ ] **Step 6: Run test, verify it passes**

Run: `pytest tests/test_fusion.py -v`
Expected: PASS (6 tests).

- [ ] **Step 7: Commit**

```bash
git add eou_detector/fusion/ tests/test_fusion.py
git commit -m "feat: rule-based bilateral-veto fusion + logistic swap"
```

---

### Task 9: EndpointController — dynamic silence policy over time

**Files:**
- Create: `eou_detector/endpoint/__init__.py` (empty),
  `eou_detector/endpoint/controller.py`
- Test: `tests/test_endpoint.py`

**Interfaces:**
- Consumes: `Settings`, `FusionResult`.
- Produces:
  - `EndpointController(settings)` with:
    - `evaluate(self, fusion: FusionResult, silence_ms: float) -> str` returning
      `"ENDPOINT"` or `"WAIT"`. Fires `ENDPOINT` when `silence_ms >=
      fusion.required_silence_ms` **and** `fusion.decision == "ENDPOINT"`, OR when
      `silence_ms >= max_silence_ms` (catch-all), unless the fusion reason is a
      veto and the catch-all has not been reached.
    - `reset(self) -> None`.

- [ ] **Step 1: Write the failing test**

`tests/test_endpoint.py`:
```python
from eou_detector.config import Settings
from eou_detector.types import FusionResult
from eou_detector.endpoint.controller import EndpointController

S = Settings()


def fr(decision, req, reason="x", p=0.9):
    return FusionResult(p_eou=p, decision=decision, required_silence_ms=req,
                        reason=reason)


def test_endpoints_when_silence_meets_requirement():
    c = EndpointController(S)
    assert c.evaluate(fr("ENDPOINT", S.short_ms), silence_ms=S.short_ms) == "ENDPOINT"


def test_waits_when_silence_below_requirement():
    c = EndpointController(S)
    assert c.evaluate(fr("ENDPOINT", S.med_ms), silence_ms=100.0) == "WAIT"


def test_veto_holds_until_max_silence_catch_all():
    c = EndpointController(S)
    veto = fr("WAIT", S.max_silence_ms, reason="lexical_veto", p=0.05)
    assert c.evaluate(veto, silence_ms=800.0) == "WAIT"
    assert c.evaluate(veto, silence_ms=S.max_silence_ms) == "ENDPOINT"  # catch-all


def test_catch_all_forces_endpoint_on_low_confidence():
    c = EndpointController(S)
    low = fr("WAIT", S.long_ms, reason="score_wait", p=0.2)
    assert c.evaluate(low, silence_ms=S.max_silence_ms + 1) == "ENDPOINT"
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_endpoint.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `eou_detector/endpoint/controller.py`**

```python
from eou_detector.config import Settings
from eou_detector.types import FusionResult


class EndpointController:
    """Applies the temporal silence policy on top of fusion's instantaneous
    decision. Confidence sets how long of a pause is required; a hard
    max-silence catch-all guarantees the turn eventually ends."""

    def __init__(self, settings: Settings):
        self._s = settings

    def evaluate(self, fusion: FusionResult, silence_ms: float) -> str:
        if silence_ms >= self._s.max_silence_ms:
            return "ENDPOINT"  # catch-all overrides any veto
        if fusion.decision == "ENDPOINT" and silence_ms >= fusion.required_silence_ms:
            return "ENDPOINT"
        return "WAIT"

    def reset(self) -> None:
        return None
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_endpoint.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add eou_detector/endpoint/ tests/test_endpoint.py
git commit -m "feat: dynamic-silence endpoint controller with max catch-all"
```

---

### Task 10: Orchestrator — asyncio pipeline wiring it all

**Files:**
- Create: `eou_detector/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `Settings`, `AudioSource`, `VAD`, `ASR`, `AcousticEOU`, `LexicalEOU`,
  `Fusion`, `EndpointController`, `RingBuffer`, `Decision`.
- Produces:
  - `Orchestrator(settings, source, vad, asr, acoustic, lexical, fusion,
    endpoint, on_decision: Callable[[Decision], None])`.
  - `async def run(self) -> None` — consumes frames; gates via VAD; on a
    speech→silence pause sustained `min_silence_ms`, evaluates acoustic+lexical,
    fuses, applies the endpoint policy, and calls `on_decision` for every
    evaluation; emits a final `ENDPOINT` decision then resets per turn.
  - Acoustic inference runs via `loop.run_in_executor`; lexical is a plain
    snapshot read (never awaited on the ASR).
  - `monotonic_ms()` injected as `time_fn` for deterministic tests.

- [ ] **Step 1: Write the failing integration test**

`tests/test_orchestrator.py`:
```python
import asyncio
import numpy as np
import pytest

from eou_detector.config import Settings
from eou_detector.audio.source import AudioSource
from eou_detector.vad.base import VAD
from eou_detector.asr.scripted_asr import ScriptedASR
from eou_detector.eou.acoustic import StubAcousticEOU
from eou_detector.eou.lexical import CompositeLexicalEOU, FRHeuristicVeto
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
        lexical=CompositeLexicalEOU(None, FRHeuristicVeto()),
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
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `eou_detector/orchestrator.py`**

```python
import asyncio
import time
from typing import Callable, Optional

import numpy as np

from eou_detector.config import Settings
from eou_detector.audio.source import AudioSource
from eou_detector.audio.ring_buffer import RingBuffer
from eou_detector.vad.base import VAD
from eou_detector.asr.base import ASR
from eou_detector.eou.acoustic import AcousticEOU
from eou_detector.eou.lexical import LexicalEOU
from eou_detector.fusion.base import Fusion
from eou_detector.endpoint.controller import EndpointController
from eou_detector.types import FusionInput, Decision


def monotonic_ms() -> float:
    return time.monotonic() * 1000.0


class Orchestrator:
    def __init__(self, settings: Settings, source: AudioSource, vad: VAD,
                 asr: ASR, acoustic: AcousticEOU, lexical: LexicalEOU,
                 fusion: Fusion, endpoint: EndpointController,
                 on_decision: Callable[[Decision], None],
                 time_fn: Callable[[], float] = monotonic_ms):
        self._s = settings
        self._source = source
        self._vad = vad
        self._asr = asr
        self._acoustic = acoustic
        self._lexical = lexical
        self._fusion = fusion
        self._endpoint = endpoint
        self._on_decision = on_decision
        self._now = time_fn
        self._ring = RingBuffer(int(settings.ring_seconds * settings.sample_rate))
        self._win = int(settings.smart_turn_window_s * settings.sample_rate)

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        await self._asr.start()
        try:
            in_speech = False
            silence_start: Optional[float] = None
            latched = False  # endpoint already fired for this turn

            async for frame in self._source.frames():
                now = self._now()
                pcm = np.frombuffer(frame, dtype=np.int16)
                self._ring.extend(pcm)
                self._asr.send_audio(frame)
                speech = self._vad.process(frame)

                if speech:
                    in_speech = True
                    silence_start = None
                    latched = False
                    self._endpoint.reset()
                    continue

                if not in_speech:
                    continue  # leading silence, nothing to end

                if silence_start is None:
                    silence_start = now
                silence_ms = now - silence_start
                if silence_ms < self._s.min_silence_ms or latched:
                    continue

                # --- evaluation (micro-pause reached) ---
                t0 = self._now()
                window = self._ring.snapshot(self._win)
                p_ac = await loop.run_in_executor(
                    None, self._acoustic.predict, window)
                partial = self._asr.latest_partial()              # snapshot, no await
                lex = self._lexical.predict(partial.text)
                fr = self._fusion.fuse(FusionInput(
                    p_ac=p_ac, p_ac_available=True,
                    p_lex=lex.p_lex, p_lex_available=bool(partial.text),
                    silence_ms=silence_ms))
                final = self._endpoint.evaluate(fr, silence_ms)
                latency = self._now() - t0

                self._on_decision(Decision(
                    ts_ms=now, p_ac=p_ac, p_lex=lex.p_lex, p_eou=fr.p_eou,
                    decision=final, required_silence_ms=fr.required_silence_ms,
                    silence_ms=silence_ms, latency_ms=latency,
                    reason=(lex.reason if lex.veto else fr.reason),
                    partial_text=partial.text))

                if final == "ENDPOINT":
                    latched = True
                    in_speech = False  # turn done; await next speech onset
        finally:
            await self._asr.stop()
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full suite**

Run: `pytest -v`
Expected: PASS (all unit tests; integration tests skip without models/hardware).

- [ ] **Step 6: Commit**

```bash
git add eou_detector/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: asyncio orchestrator wiring VAD/ASR/EOU/fusion/endpoint"
```

---

### Task 11: demo.py — real-time mic/wav runner with live logging

**Files:**
- Create: `demo.py`
- Test: `tests/test_demo_smoke.py`

**Interfaces:**
- Consumes: everything above; builds real components (Silero, Azure or scripted,
  Smart Turn or stub).
- Produces:
  - CLI: `python demo.py --wav FILE [--no-realtime]` or `python demo.py --mic`.
  - Flags: `--asr {azure,scripted}` (default azure), `--acoustic {smart_turn,stub}`
    (default smart_turn, auto-falls back to stub if model missing),
    `--smart-turn-onnx PATH` (default `models/smart-turn-v3.onnx`).
  - `build_components(args, settings) -> tuple` factory (unit-testable) returning
    `(source, vad, asr, acoustic, lexical, fusion, endpoint)`.
  - `format_decision(d: Decision) -> str` log line:
    `t=..ms p_ac=.. p_lex=.. p_eou=.. dec=.. req=..ms sil=..ms lat=..ms reason=..`.

- [ ] **Step 1: Write the failing smoke test**

`tests/test_demo_smoke.py`:
```python
from eou_detector.types import Decision
from demo import format_decision


def test_format_decision_contains_fields():
    d = Decision(ts_ms=1000.0, p_ac=0.91, p_lex=0.80, p_eou=0.85,
                 decision="ENDPOINT", required_silence_ms=200.0, silence_ms=210.0,
                 latency_ms=27.3, reason="both_confident", partial_text="oui")
    line = format_decision(d)
    for token in ["p_ac=0.91", "p_lex=0.80", "dec=ENDPOINT", "lat=27", "reason=both_confident"]:
        assert token in line
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_demo_smoke.py -v`
Expected: FAIL — `demo` module / `format_decision` missing.

- [ ] **Step 3: Implement `demo.py`**

```python
import argparse
import asyncio
import os

from eou_detector.config import load_settings
from eou_detector.types import Decision
from eou_detector.audio.source import WavStreamSource, MicSource
from eou_detector.vad.silero_vad import SileroVAD
from eou_detector.asr.azure_asr import AzureSpeechASR
from eou_detector.asr.scripted_asr import ScriptedASR
from eou_detector.eou.acoustic import SmartTurnV3, StubAcousticEOU
from eou_detector.eou.lexical import CompositeLexicalEOU, FRHeuristicVeto
from eou_detector.fusion.rules import RuleFusion
from eou_detector.endpoint.controller import EndpointController
from eou_detector.orchestrator import Orchestrator


def format_decision(d: Decision) -> str:
    return (f"t={d.ts_ms:7.0f}ms p_ac={d.p_ac:.2f} p_lex={d.p_lex:.2f} "
            f"p_eou={d.p_eou:.2f} dec={d.decision} "
            f"req={d.required_silence_ms:.0f}ms sil={d.silence_ms:.0f}ms "
            f"lat={d.latency_ms:.0f}ms reason={d.reason} | '{d.partial_text}'")


def build_components(args, settings):
    if args.mic:
        source = MicSource(settings.sample_rate, settings.frame_samples)
    else:
        source = WavStreamSource(args.wav, settings.sample_rate,
                                 settings.frame_samples,
                                 realtime=not args.no_realtime)
    vad = SileroVAD(settings.sample_rate, settings.vad_threshold)

    if args.asr == "scripted":
        asr = ScriptedASR(script=[(0, ""), (10, "oui c'est exact")])
    else:
        asr = AzureSpeechASR(settings)

    if args.acoustic == "smart_turn" and os.path.exists(args.smart_turn_onnx):
        acoustic = SmartTurnV3(args.smart_turn_onnx, settings.sample_rate,
                               settings.smart_turn_window_s)
    else:
        if args.acoustic == "smart_turn":
            print(f"[warn] {args.smart_turn_onnx} missing; using stub acoustic")
        acoustic = StubAcousticEOU(0.5)

    lexical = CompositeLexicalEOU(model=None, heuristic=FRHeuristicVeto())
    fusion = RuleFusion(settings)
    endpoint = EndpointController(settings)
    return source, vad, asr, acoustic, lexical, fusion, endpoint


def main():
    ap = argparse.ArgumentParser(description="Real-time semantic EOU demo")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--wav", help="path to a wav replayed as a stream")
    g.add_argument("--mic", action="store_true", help="use the live microphone")
    ap.add_argument("--no-realtime", action="store_true",
                    help="replay wav as fast as possible (no pacing)")
    ap.add_argument("--asr", choices=["azure", "scripted"], default="azure")
    ap.add_argument("--acoustic", choices=["smart_turn", "stub"], default="smart_turn")
    ap.add_argument("--smart-turn-onnx", default="models/smart-turn-v3.onnx")
    args = ap.parse_args()

    settings = load_settings()
    comps = build_components(args, settings)
    orch = Orchestrator(settings, *comps,
                        on_decision=lambda d: print(format_decision(d)))
    asyncio.run(orch.run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_demo_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Manual smoke (offline, no creds/models needed)**

Run: `python demo.py --wav eval/clips/fini/<any>.wav --no-realtime --asr scripted --acoustic stub`
Expected: prints decision log lines ending in an `ENDPOINT`.

- [ ] **Step 6: Commit**

```bash
git add demo.py tests/test_demo_smoke.py
git commit -m "feat: real-time mic/wav demo with live decision logging"
```

---

### Task 12: Eval harness — labelled clips → accuracy / FP / FN / latency

**Files:**
- Create: `eval/__init__.py` (empty), `eval/metrics.py`, `eval/harness.py`
- Create: `eval/clips/fini/.gitkeep`, `eval/clips/pas_fini/.gitkeep`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: orchestrator stack, `Decision`.
- Produces:
  - `eval/metrics.py`:
    - `ClipResult(label: str, predicted_finished: bool, latency_ms: float|None)`
      dataclass.
    - `evaluate_clip(decisions: list[Decision], settings) -> tuple[bool, float|None]`
      — `predicted_finished` = an `ENDPOINT` fired **before** the max-silence
      catch-all (i.e. a genuine semantic decision); `latency_ms` = time from
      silence onset to that ENDPOINT (the decision's `silence_ms`). Returns
      `(False, None)` if only the catch-all fired.
    - `summarize(results: list[ClipResult]) -> dict` — `accuracy`,
      `false_positives` (label `pas_fini` predicted finished),
      `false_negatives` (label `fini` not predicted finished),
      `median_latency_ms`, `p90_latency_ms`, `n`.
  - `eval/harness.py`: `python -m eval.harness [--asr scripted|azure]
    [--acoustic stub|smart_turn]` — walks `clips/{fini,pas_fini}`, replays each
    wav (non-realtime) through the orchestrator, prints the summary table.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
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
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `eval/metrics.py`**

```python
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
```

- [ ] **Step 4: Run test, verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Implement `eval/harness.py`**

```python
import argparse
import asyncio
import os

from eou_detector.config import load_settings
from eou_detector.audio.source import WavStreamSource
from eou_detector.vad.silero_vad import SileroVAD
from eou_detector.asr.scripted_asr import ScriptedASR
from eou_detector.asr.azure_asr import AzureSpeechASR
from eou_detector.eou.acoustic import SmartTurnV3, StubAcousticEOU
from eou_detector.eou.lexical import CompositeLexicalEOU, FRHeuristicVeto
from eou_detector.fusion.rules import RuleFusion
from eou_detector.endpoint.controller import EndpointController
from eou_detector.orchestrator import Orchestrator
from eval.metrics import ClipResult, evaluate_clip, summarize

CLIPS = os.path.join(os.path.dirname(__file__), "clips")


def _build(wav, args, settings):
    source = WavStreamSource(wav, settings.sample_rate, settings.frame_samples,
                             realtime=False)
    vad = SileroVAD(settings.sample_rate, settings.vad_threshold)
    asr = (ScriptedASR(script=[(0, "")]) if args.asr == "scripted"
           else AzureSpeechASR(settings))
    if args.acoustic == "smart_turn" and os.path.exists(args.smart_turn_onnx):
        acoustic = SmartTurnV3(args.smart_turn_onnx, settings.sample_rate,
                               settings.smart_turn_window_s)
    else:
        acoustic = StubAcousticEOU(0.5)
    lexical = CompositeLexicalEOU(None, FRHeuristicVeto())
    return (source, vad, asr, acoustic, lexical,
            RuleFusion(settings), EndpointController(settings))


def _run_clip(wav, args, settings):
    decisions = []
    comps = _build(wav, args, settings)
    orch = Orchestrator(settings, *comps, on_decision=decisions.append)
    asyncio.run(orch.run())
    return decisions


def main():
    ap = argparse.ArgumentParser(description="EOU eval harness")
    ap.add_argument("--asr", choices=["scripted", "azure"], default="azure")
    ap.add_argument("--acoustic", choices=["stub", "smart_turn"], default="smart_turn")
    ap.add_argument("--smart-turn-onnx", default="models/smart-turn-v3.onnx")
    args = ap.parse_args()
    settings = load_settings()

    results = []
    for label in ("fini", "pas_fini"):
        folder = os.path.join(CLIPS, label)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(".wav"):
                continue
            decisions = _run_clip(os.path.join(folder, fn), args, settings)
            fin, lat = evaluate_clip(decisions, settings)
            results.append(ClipResult(label, fin, lat))
            print(f"{label:9} {fn:30} finished={fin} latency={lat}")

    s = summarize(results)
    print("\n=== SUMMARY ===")
    for k, v in s.items():
        print(f"{k:18}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run test + harness smoke**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS.
Run (after adding at least one wav to each folder): `python -m eval.harness --asr scripted --acoustic stub`
Expected: prints per-clip lines + a SUMMARY block.

- [ ] **Step 7: Commit**

```bash
git add eval/ tests/test_metrics.py
git commit -m "feat: eval harness — accuracy, FP/FN, latency over labelled clips"
```

---

### Task 13: README + final full-suite verification

**Files:**
- Create: `README.md`
- Test: run the full suite.

**Interfaces:**
- Consumes: everything.
- Produces: `README.md` documenting quickstart, Azure config (reuse prod creds),
  model download, demo + eval usage, and the new-ASR extension point.

- [ ] **Step 1: Write `README.md`**

````markdown
# Semantic EOU Detector (POC)

Real-time end-of-utterance detection for a French medical voice agent. Fuses an
acoustic signal (Smart Turn v3, ONNX) and a lexical signal (turn-detector + FR
heuristic veto) on top of a Silero VAD gate, behind swappable interfaces. Built
to integrate with the CareCallHouseMade NestJS stack (reuses its Azure Speech
credentials and audio format).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_models.py        # downloads Smart Turn v3 ONNX into ./models
```

## Azure config (reuse prod credentials)

The ASR uses the **same env vars as prod** (`recall/CareCallHouseMade`):

```bash
export AZURE_STT_API_KEY=...      # = prod AZURE_STT_API_KEY
export AZURE_STT_REGION=...       # = prod AZURE_STT_REGION
# or: cp .env.example .env && fill it (auto-loaded via python-dotenv)
# or: source the prod .env directly
```

Language is `fr-FR`, audio is 16 kHz / mono / 16-bit PCM — matching prod. The
POC's Azure ASR mirrors `azure-stt.service.ts` and additionally subscribes to the
`recognizing` event to obtain real-time partials for the lexical branch (prod
stays final-only and is untouched).

## Run the demo

```bash
# Replay a wav as a real-time stream (Azure ASR + Smart Turn):
python demo.py --wav path/to/clip.wav

# Live microphone:
python demo.py --mic

# Fully offline smoke (no creds, no models):
python demo.py --wav path/to/clip.wav --no-realtime --asr scripted --acoustic stub
```

Each line logs: `p_ac`, `p_lex`, `p_eou`, decision, required/observed silence,
and **decision latency**.

## Eval harness

Drop labelled clips into `eval/clips/fini/` and `eval/clips/pas_fini/`, then:

```bash
python -m eval.harness                       # Azure + Smart Turn
python -m eval.harness --asr scripted --acoustic stub   # offline
```

Outputs accuracy, false positives / false negatives, and median / p90 latency.

## Architecture

```
AudioSource → Silero VAD (gate) ─┬─ RingBuffer ─→ Smart Turn v3 → p_ac ┐
                                 ├─ Azure push-stream → recognizing ── partial → Lexical → p_lex
                                 └─ on micro-pause: Fusion(p_ac,p_lex) → EndpointController → decision
```

Modules behind interfaces (`eou_detector/`): `audio`, `vad`, `asr`, `eou`
(acoustic + lexical), `fusion`, `endpoint`, `orchestrator`. Each is swappable.

The decision **never blocks on the ASR**: lexical is a snapshot of the latest
partial; only the acoustic ONNX runs on the decision path.

### Fusion: rules now, logistic later

`RuleFusion` applies a bilateral veto (lexical or acoustic strongly mid-turn →
WAIT) and returns a probability `p_eou`. To switch to a learned model, drop in
`LogisticFusion` (same `Fusion` interface) — no caller changes.

### Extension point: a new ASR

Implement `eou_detector/asr/base.py:ASR` (`start`, `send_audio`,
`latest_partial`, `stop`) and pass it to the `Orchestrator`. Example: a
self-hosted STT client. `AzureSpeechASR` and `ScriptedASR` are reference
implementations. VAD swap: implement `eou_detector/vad/base.py:VAD` (e.g. a
WebSocket client to the prod VAD microservice) in place of `SileroVAD`.

## Tunables

All thresholds live in `eou_detector/config.py:Settings` (veto bands, confidence
high/mid, short/med/long/max silence). Defaults are POC starting points.
````

- [ ] **Step 2: Run the full test suite**

Run: `pytest -v`
Expected: PASS — all unit tests green; integration tests (Silero, Smart Turn,
Azure) skip cleanly when models/creds/hardware are absent.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README — quickstart, Azure config, extension points"
```

---

## Self-Review Notes (author)

- **Spec coverage:** VAD gate (T4), acoustic branch (T6), lexical branch (T7),
  bilateral-veto fusion + logistic-ready (T8), dynamic endpoint (T9), Azure ASR
  with `recognizing` partials reusing prod creds (T5), mic + wav stream input
  (T3), never-block-on-lexical (T10 snapshot read), demo (T11), eval harness
  (T12), README (T13). All §-sections of the design map to a task.
- **Latency constraint:** enforced in T10 — only `acoustic.predict` is awaited
  (in an executor); lexical is a plain `latest_partial()` read.
- **Decoupling:** every module is an ABC with at least one alternate impl
  (Stub/Scripted/Logistic) proving swappability.
- **Type consistency:** `Decision`, `FusionResult`, `LexResult`, `FusionInput`,
  `Partial` field names are identical across T1 definitions and all consumers.
- **Known follow-ups for the implementer:** confirm Smart Turn v3 ONNX I/O names
  + preprocessing against the model card (T6 note); optionally wire a real
  turn-detector model into `TurnDetectorEOU` (T7) — heuristic-only is the
  default and passes all tests.
```
