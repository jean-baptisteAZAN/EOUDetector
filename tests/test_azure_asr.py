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
