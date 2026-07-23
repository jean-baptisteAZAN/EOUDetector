# Semantic End-of-Utterance (EOU) Detector — POC Design

**Date:** 2026-06-26
**Status:** Approved (design), pre-implementation
**Author:** POC team

## 1. Problem & Goal

Build a Python POC for **real-time semantic endpointing** for a French medical
voice agent (patients on the phone). Decide **fast** whether the caller has
finished speaking by fusing an **acoustic** signal (waveform) and a **lexical**
signal (partial transcript), layered on top of an existing VAD.

A Silero VAD already separates speech/silence but ignores semantics, producing
**false positives** (cutting the caller off) on:
- name spelling (`M-A-R-T-I-N`),
- dates and phone/ID numbers given in chunks,
- thinking pauses.

The POC must eliminate these cases without adding decision latency.

## 2. Integration Context (existing prod system)

The POC targets clean integration with the production NestJS system at
`recall/CareCallHouseMade/src`. Facts mirrored from prod:

| Aspect | Prod value | POC stance |
|---|---|---|
| ASR | Azure Speech SDK, `fr-FR`, PCM **16kHz/16-bit/mono** push-stream | Same SDK/format/lang, reuse creds |
| ASR creds (env) | `AZURE_STT_API_KEY`, `AZURE_STT_REGION` | **Reuse same env vars** (`source` prod `.env`) |
| ASR events | **`recognized` (final) only** — no interim today | POC **adds `recognizing`** handler for real-time partials (non-invasive; prod unchanged) |
| Endpoint today | Azure `Speech_SegmentationSilenceTimeoutMs` (500ms default, 1700ms for `+33939240743`) + orchestrator frame-count (`SPEECH_FRAMES_THRESHOLD=5`) | Semantic layer augments this; dynamic timeout mirrors `updateSilenceTimeoutMs()` pattern |
| VAD | Separate **WebSocket microservice** (`ws://HOST:PORT/vad` → `{callSid,isSpeech}`) | POC runs **Silero in-process** behind a `VAD` interface; WS-adapter is the documented swap |
| STT interface | `ISttService` (`createSttConnection / sendAudio / stopSttConnection / updateSilenceTimeoutMs`) | Python `ASR` interface mirrors this conceptually |

**Decision:** prod's final-only Azure config is kept; the POC's Azure ASR simply
subscribes to **both** `recognizing` (→ partials → `p_lex`) and `recognized`.

## 3. Locked Decisions

- **Lexical branch (`p_lex`)** = ML **turn-detector** model (multilingual incl.
  French, ONNX/CPU) on the latest partial hypothesis, **plus a thin FR
  heuristic veto** for spelling-in-progress / partial dates / partial numbers
  that a generic model may miss. Both behind one `LexicalEOU` interface.
  Primary model candidate: LiveKit multilingual turn-detector (ONNX). Fallback:
  heuristic-only if model unavailable.
- **Acoustic branch (`p_ac`)** = **Smart Turn v3** (ONNX, local) on the recent
  audio buffer. Candidate: `pipecat-ai/smart-turn-v3`.
- **ASR** = **Azure only** (mirrors prod), with `recognizing` partials added.
  Eval hits the network per run (acceptable for POC); optional `--record` cache.
- **VAD** = **Silero in-process** (`silero-vad`), self-contained.
- **Fusion** = **rule-based bilateral veto** first, structured so a **logistic
  regression** drops in behind the same interface with zero caller change.
- **Concurrency** = **asyncio core + ThreadPoolExecutor** for ONNX inference;
  Azure SDK and `sounddevice` run their own callback threads, bridged via a
  thread-safe latest-partial holder.
- **Models** = downloaded at build time (Smart Turn v3 + turn-detector).

## 4. Architecture

### 4.1 Module map (each swappable behind an interface)

```
eou_detector/
  config.py            env-based config; reuses AZURE_STT_* ; thresholds
  audio/
    source.py          AudioSource (iface); MicSource ; WavStreamSource (real-time replay)
    ring_buffer.py     thread-safe rolling PCM buffer (~8 s)
  vad/
    base.py            VAD (iface): speech/silence per frame
    silero_vad.py      SileroVAD (in-process)        [swap: WSVadClient → prod microservice]
  asr/
    base.py            ASR (iface): start / send_audio / latest_partial / stop ; mirrors ISttService
    azure_asr.py       AzureSpeechASR (recognizing + recognized), reuses AZURE_STT_* env
  eou/
    acoustic.py        AcousticEOU (iface); SmartTurnV3 (ONNX → p_ac)
    lexical.py         LexicalEOU (iface); TurnDetector (ONNX → p_lex) + FRHeuristicVeto
  fusion/
    base.py            Fusion (iface); FusionInput ; FusionResult
    rules.py           RuleFusion (bilateral veto)
    logistic.py        LogisticFusion (swap-in, same iface)
  endpoint/
    controller.py      EndpointController (dynamic silence policy)
  orchestrator.py      wires VAD gate → trigger → acoustic+lexical → fusion → endpoint
demo.py                --mic | --wav FILE ; logs realtime p_ac p_lex p_eou decision latency
eval/
  harness.py           labelled clips/{fini,pas_fini}/*.wav → accuracy, FP/FN, median latency
  clips/               labelled wav fixtures
README.md
requirements.txt
.env.example           references AZURE_STT_REGION, AZURE_STT_API_KEY
```

### 4.2 Interfaces (conceptual signatures)

```python
class AudioSource:        # async generator of int16 PCM frames @16k mono
    async def frames(self) -> AsyncIterator[bytes]: ...

class VAD:
    def process(self, frame: bytes) -> bool: ...        # True = speech

class ASR:
    async def start(self) -> None: ...
    def send_audio(self, frame: bytes) -> None: ...      # push-stream, non-blocking
    def latest_partial(self) -> Partial: ...             # snapshot, may be slightly stale
    async def stop(self) -> None: ...

class AcousticEOU:
    def predict(self, pcm_window: np.ndarray) -> float:  # p_ac in [0,1]

class LexicalEOU:
    def predict(self, partial_text: str) -> LexResult:   # p_lex + veto flag + reason

@dataclass
class FusionInput:
    p_ac: float; p_ac_available: bool
    p_lex: float; p_lex_available: bool
    silence_ms: float

@dataclass
class FusionResult:
    p_eou: float
    decision: Literal["ENDPOINT", "WAIT"]
    required_silence_ms: float
    reason: str

class Fusion:
    def fuse(self, x: FusionInput) -> FusionResult: ...
```

### 4.3 Real-time pipeline

1. `AudioSource` emits **32 ms frames (512 samples @16k)**. Each frame fans out
   to: `RingBuffer`, Azure push-stream (`send_audio`), Silero VAD.
2. Silero per-frame → speech/silence. On a **speech→silence edge** sustained for
   `MIN_SILENCE_MS` (~200 ms) → fire **evaluation** (the "micro-pause" trigger).
3. **Acoustic**: `SmartTurnV3.predict(recent_window)` → `p_ac` (in executor,
   ~25–80 ms).
4. **Lexical**: read `asr.latest_partial()` (atomic snapshot, **never awaits the
   ASR**) → `LexicalEOU.predict(text)` → `p_lex` (+ veto flag).
5. **Fusion**: `RuleFusion.fuse(FusionInput{...})` → `FusionResult`.
6. **Dynamic endpoint**: `EndpointController` maps confidence → required silence,
   emits `ENDPOINT` when met, `MAX_SILENCE_MS` (~2000 ms) catch-all force-fire.

### 4.4 Fusion logic (bilateral veto → logistic-ready)

```
VETO_LEX, VETO_AC      = strong "not done" thresholds (e.g. 0.25)
HIGH                   = strong "done" threshold (e.g. 0.85)

RuleFusion.fuse(x):
  if x.p_lex_available and x.p_lex < VETO_LEX:   # mid-spelling / partial number
      return WAIT, p_eou=min(...), reason="lexical_veto"
  if x.p_ac_available and x.p_ac < VETO_AC:      # acoustic says clearly mid-turn
      return WAIT, p_eou=..., reason="acoustic_veto"
  if both high:
      return ENDPOINT now, reason="both_confident"
  p_eou = weighted_combine(p_ac, p_lex, availability)
  decision = ENDPOINT if p_eou >= thr else WAIT
```

`p_eou` is always returned, so swapping in `LogisticFusion` (learned weights on
`[p_ac, p_lex, p_ac*p_lex, silence_ms]`) requires **no change** in the
`EndpointController` or orchestrator.

### 4.5 Dynamic endpoint policy

```
p_eou >= HIGH        → required_silence = SHORT_MS   (~200 ms)   # fast cut
MID  <= p_eou < HIGH → required_silence = MED_MS     (~600 ms)
p_eou <  MID         → required_silence = LONG_MS    (~1200 ms)  # extend / give time
any                  → MAX_SILENCE_MS  (~2000 ms) force ENDPOINT (catch-all)
```

Mirrors prod's `updateSilenceTimeoutMs()` idea: the timeout is data-driven, not
fixed.

## 5. Concurrency & Buffers

- **asyncio** orchestrator owns the decision loop. `asyncio.Queue` carries VAD
  edge events from the audio thread to the loop.
- **ThreadPoolExecutor** runs both ONNX models (acoustic always; lexical if the
  model variant is heavy) via `loop.run_in_executor`.
- Azure SDK `recognizing` callback (its own thread) writes `latest_partial`
  under a lock; fusion reads a snapshot. **No await on ASR in the decision path.**
- `sounddevice` mic callback (its own thread) pushes frames into the asyncio
  loop via `call_soon_threadsafe`.
- `WavStreamSource` paces frames at **wall-clock real time** so file replay
  reproduces live timing (reproducible tests, real latency numbers).
- `RingBuffer`: fixed ~8 s of int16 PCM, lock-guarded, gives Smart Turn its
  recent window without reallocating.

## 6. Latency Guarantee

The only compute on the decision path is the **acoustic ONNX** call (~25–80 ms).
Lexical is a pointer read of the newest partial. If acoustic overruns its
budget, fuse **lexical-only** (or last `p_ac`) and proceed. **The decision never
blocks on ASR lag** — it always uses the freshest partial available, even if
slightly stale. This satisfies the hard real-time constraint.

## 7. Deliverables

1. **`demo.py`** — takes a stream (`--mic` or `--wav FILE`) and logs in real time:
   `t  p_ac  p_lex  p_eou  decision  required_silence_ms  latency_ms  reason`.
2. **`eval/harness.py`** — points at `clips/{fini,pas_fini}/*.wav`, replays each
   as a stream, outputs **accuracy, false-positive / false-negative counts,
   median (and p90) decision latency**. Optional `--record` to cache Azure
   partials for offline re-runs.
3. **`README.md`** — quickstart, Azure config (reuse prod creds via env / source
   prod `.env`), and the **extension point for a new ASR** (implement `ASR`).

## 8. Configuration (env)

| Var | Source | Use |
|---|---|---|
| `AZURE_STT_API_KEY` | prod `.env` | Azure Speech key (reused) |
| `AZURE_STT_REGION` | prod `.env` | Azure region (reused) |
| `EOU_LANG` | default `fr-FR` | recognition language |
| `EOU_*_THRESHOLD` / `EOU_*_MS` | config defaults | tunable fusion/endpoint knobs |

## 9. Testing Strategy

- **Unit**: `RuleFusion` truth table (veto cases, both-confident, mid-band);
  `FRHeuristicVeto` on spelling / partial-date / partial-number strings;
  `WavStreamSource` pacing; `RingBuffer` thread-safety.
- **Integration**: `WavStreamSource` → full orchestrator with a **fake/recorded
  ASR partial track** so the fusion + endpoint path is testable without network.
- **Eval**: labelled clip harness (§7.2) is the acceptance metric — target:
  eliminate the three FP classes (spelling / chunked numbers / thinking pauses)
  without raising false-negative cut latency on genuinely-finished turns.

## 10. Out of Scope (POC)

No production deploy, no telephony/SIP, no fine-tuning, off-the-shelf models
only, no wiring of the prod VAD WS microservice (Silero in-process instead;
WS-adapter documented as the swap).

## 11. Open Risks

- Turn-detector model French quality on **isolated partial utterances** (vs
  multi-turn chat context it was trained on) — mitigated by the FR heuristic veto.
- Smart Turn v3 window length / preprocessing must match the model card — verify
  on download.
- Azure partial (`recognizing`) cadence/stability for `fr-FR` — measure during
  build; `latest_partial` staleness is by-design tolerated.
