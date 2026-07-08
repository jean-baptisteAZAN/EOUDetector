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
