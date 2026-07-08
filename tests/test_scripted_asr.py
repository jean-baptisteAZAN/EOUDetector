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
