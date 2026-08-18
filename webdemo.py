"""Tiny web front for the live EOU demo.

Serves a one-page UI (web/index.html) and a WebSocket that carries the browser
microphone (16 kHz mono int16 PCM) into the very same decision core the CLI demo
and the production path use (EouSession: Silero VAD -> Smart Turn + CamemBERT ->
fusion -> endpoint). Per-pause decisions (p_ac / p_lex / p_eou / ENDPOINT|WAIT)
and the live ASR partial are streamed back to the page.

Run:  python webdemo.py            # then open http://127.0.0.1:8970
Heavy models (Smart Turn ONNX, CamemBERT) load once at startup and are shared
across connections; the VAD / fusion / endpoint / ASR are per connection.
"""
import asyncio
import os

from aiohttp import web, WSMsgType

from eou_detector.config import load_settings
from eou_detector.vad.silero_vad import SileroVAD
from eou_detector.asr.azure_asr import AzureSpeechASR
from eou_detector.asr.scripted_asr import ScriptedASR
from eou_detector.eou.acoustic import SmartTurnV3, StubAcousticEOU, find_smart_turn_model
from eou_detector.eou.lexical import CamembertLexicalEOU
from eou_detector.fusion.rules import RuleFusion
from eou_detector.endpoint.controller import EndpointController
from eou_detector.session import EouSession

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "web", "index.html")


def build_shared(settings):
    """Heavy, stateless models shared across every connection."""
    model_path = find_smart_turn_model()
    if model_path:
        acoustic = SmartTurnV3(model_path, settings.sample_rate,
                               settings.smart_turn_window_s)
        ac_name = f"Smart Turn v3 ({os.path.basename(model_path)})"
    else:
        acoustic = StubAcousticEOU(0.5)
        ac_name = "stub (no ONNX in models/)"
    lexical = CamembertLexicalEOU()
    return acoustic, lexical, ac_name


async def on_startup(app):
    s = app["settings"]
    app["acoustic"], app["lexical"], app["ac_name"] = build_shared(s)
    app["azure"] = bool(s.azure_key and s.azure_region)
    print(f"[web] acoustic: {app['ac_name']}", flush=True)
    print("[web] lexical : CamemBERT fine-tuned (models/camembert-eou)", flush=True)
    print(f"[web] ASR     : {'Azure Speech (fr-FR)' if app['azure'] else 'DISABLED (no Azure creds -> lexical stays blank)'}", flush=True)
    print(f"[web] open    : http://127.0.0.1:{s.eou_port}", flush=True)


async def index(request):
    return web.FileResponse(INDEX)


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    app = request.app
    s = app["settings"]
    loop = asyncio.get_running_loop()

    vad = SileroVAD(s.sample_rate, s.vad_threshold)
    fusion = RuleFusion(s)
    endpoint = EndpointController(s)
    session = EouSession(s, vad, app["acoustic"], app["lexical"], fusion, endpoint)
    asr = AzureSpeechASR(s) if app["azure"] else ScriptedASR(script=[(0, "")])
    await asr.start()

    await ws.send_json({"type": "hello", "acoustic": app["ac_name"],
                        "asr": "azure" if app["azure"] else "disabled"})
    last_partial = None
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                pcm = msg.data
                asr.send_audio(pcm)
                partial = asr.latest_partial().text
                session.set_partial(partial)
                if partial != last_partial:
                    last_partial = partial
                    await ws.send_json({"type": "partial", "text": partial})
                for d in await session.process(pcm, loop):
                    await ws.send_json({
                        "type": "decision", "decision": d.decision,
                        "p_ac": d.p_ac, "p_lex": d.p_lex, "p_eou": d.p_eou,
                        "silence_ms": d.silence_ms,
                        "required_ms": d.required_silence_ms,
                        "latency_ms": d.latency_ms, "reason": d.reason,
                        "lex_reason": d.lex_reason, "ts_ms": d.ts_ms,
                        "text": d.partial_text})
            elif msg.type == WSMsgType.TEXT and msg.data == "stop":
                break
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        await asr.stop()
    return ws


def main():
    settings = load_settings()
    app = web.Application()
    app["settings"] = settings
    app.on_startup.append(on_startup)
    app.add_routes([web.get("/", index), web.get("/ws", ws_handler)])
    web.run_app(app, host="127.0.0.1", port=settings.eou_port, print=None)


if __name__ == "__main__":
    main()
