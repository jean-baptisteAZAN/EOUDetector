import argparse
import asyncio
import os

from eou_detector.config import load_settings
from eou_detector.types import Decision
from eou_detector.audio.source import WavStreamSource, MicSource
from eou_detector.vad.silero_vad import SileroVAD
from eou_detector.asr.azure_asr import AzureSpeechASR
from eou_detector.asr.scripted_asr import ScriptedASR
from eou_detector.eou.acoustic import (
    SmartTurnV3, StubAcousticEOU, find_smart_turn_model)
from eou_detector.eou.lexical import FrenchSemanticEOU, CamembertLexicalEOU
from eou_detector.fusion.rules import RuleFusion
from eou_detector.endpoint.controller import EndpointController
from eou_detector.orchestrator import Orchestrator


def format_decision(d: Decision) -> str:
    return (f"t={d.ts_ms:7.0f}ms p_ac={d.p_ac:.2f} p_lex={d.p_lex:.2f} "
            f"p_eou={d.p_eou:.2f} dec={d.decision} "
            f"req={d.required_silence_ms:.0f}ms sil={d.silence_ms:.0f}ms "
            f"lat={d.latency_ms:.0f}ms reason={d.reason} lex={d.lex_reason} "
            f"| '{d.partial_text}'")


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

    model_path = args.smart_turn_onnx or find_smart_turn_model()
    if args.acoustic == "smart_turn" and model_path:
        print(f"[info] acoustic: Smart Turn v3 ({model_path})")
        acoustic = SmartTurnV3(model_path, settings.sample_rate,
                               settings.smart_turn_window_s)
    else:
        if args.acoustic == "smart_turn":
            print("[warn] no Smart Turn ONNX in models/ "
                  "(run: python scripts/fetch_models.py); using stub acoustic")
        acoustic = StubAcousticEOU(0.5)

    if args.lexical == "camembert":
        lexical = CamembertLexicalEOU()
        print("[info] lexical: CamemBERT fine-tuned (models/camembert-eou)")
    else:
        lexical = FrenchSemanticEOU()
        print("[info] lexical: French semantic rules (homemade, prod-safe)")
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
    ap.add_argument("--lexical", choices=["camembert", "rules"], default="camembert",
                    help="lexical branch: fine-tuned CamemBERT (real POC, default) "
                         "or the homemade French rules")
    ap.add_argument("--smart-turn-onnx", default=None,
                    help="explicit Smart Turn ONNX path "
                         "(default: auto-discover best variant in models/)")
    args = ap.parse_args()

    settings = load_settings()
    comps = build_components(args, settings)
    orch = Orchestrator(settings, *comps,
                        on_decision=lambda d: print(format_decision(d)))
    asyncio.run(orch.run())


if __name__ == "__main__":
    main()
