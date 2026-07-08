import argparse
import asyncio
import os

from eou_detector.config import load_settings
from eou_detector.audio.source import WavStreamSource
from eou_detector.vad.silero_vad import SileroVAD
from eou_detector.asr.scripted_asr import ScriptedASR
from eou_detector.asr.azure_asr import AzureSpeechASR
from eou_detector.eou.acoustic import (
    SmartTurnV3, StubAcousticEOU, find_smart_turn_model)
from eou_detector.eou.lexical import FrenchSemanticEOU
from eou_detector.fusion.rules import RuleFusion
from eou_detector.endpoint.controller import EndpointController
from eou_detector.orchestrator import Orchestrator
from eval.metrics import ClipResult, evaluate_clip, summarize

CLIPS = os.path.join(os.path.dirname(__file__), "clips")


def _build(wav, args, settings):
    # Append trailing silence so a finished clip presents the pause the endpoint
    # logic evaluates (clipped fixtures often lack one).
    source = WavStreamSource(wav, settings.sample_rate, settings.frame_samples,
                             realtime=False, pad_silence_ms=args.pad_silence_ms)
    vad = SileroVAD(settings.sample_rate, settings.vad_threshold)
    asr = (ScriptedASR(script=[(0, "")]) if args.asr == "scripted"
           else AzureSpeechASR(settings))

    model_path = args.smart_turn_onnx or find_smart_turn_model()
    if args.acoustic == "smart_turn" and model_path:
        acoustic = SmartTurnV3(model_path, settings.sample_rate,
                               settings.smart_turn_window_s)
    else:
        acoustic = StubAcousticEOU(0.5)

    lexical = FrenchSemanticEOU()
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
    ap.add_argument("--smart-turn-onnx", default=None,
                    help="explicit path (default: auto-discover in models/)")
    ap.add_argument("--pad-silence-ms", type=float, default=1500.0,
                    help="trailing silence appended to each clip (the pause)")
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
