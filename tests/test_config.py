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
