# Semantic EOU Detector (POC)

Real-time end-of-utterance detection for a French medical voice agent. Fuses an
acoustic signal (Smart Turn v3, ONNX) and a lexical signal (turn-detector + FR
heuristic veto) on top of a Silero VAD gate, behind swappable interfaces. Built
to integrate with the CareCallHouseMade NestJS stack (reuses its Azure Speech
credentials and audio format).

**Abbreviations.** EOU = End-of-Utterance · VAD = Voice Activity Detection ·
ASR = Automatic Speech Recognition · STT = Speech-to-Text · PCM = Pulse-Code Modulation ·
ONNX = Open Neural Network Exchange · POC = Proof of Concept · CLI = Command-Line Interface ·
UI = User Interface · FR / fr-FR = French (language code).

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
# Replay a wav as a real-time stream (Azure ASR + Smart Turn + CamemBERT):
python demo.py --wav path/to/clip.wav

# Live microphone:
python demo.py --mic

# Rule-based lexical instead of the fine-tuned CamemBERT:
python demo.py --mic --lexical rules

# Fully offline smoke (no creds, no models):
python demo.py --wav path/to/clip.wav --no-realtime --asr scripted --acoustic stub --lexical rules
```

The lexical branch defaults to the fine-tuned **CamemBERT** (`models/camembert-eou`,
the model behind the reported fusion result), guarded by the French heuristic veto
for the hard incomplete classes; pass `--lexical rules` for the pure rule engine.

## Live web front

A one-page browser UI to record from the mic and watch the pipeline live
(partial transcript, `p_ac` / `p_lex` / `p_eou` bars, per-pause ENDPOINT/WAIT):

```bash
export $(grep -v '^#' .env | xargs)     # load Azure creds (needed for the text branch)
python webdemo.py                        # then open http://127.0.0.1:8970
```

It drives the exact same `EouSession` as the CLI demo and the production path
(Smart Turn + CamemBERT → fusion → endpoint); heavy models load once at startup.

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
