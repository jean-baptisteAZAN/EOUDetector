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
    eou_host: str = "0.0.0.0"
    eou_port: int = 8970


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    e = env if env is not None else os.environ
    return Settings(
        azure_key=e.get("AZURE_STT_API_KEY"),
        azure_region=e.get("AZURE_STT_REGION"),
        lang=e.get("EOU_LANG", "fr-FR"),
        eou_host=e.get("EOU_SERVICE_HOST", "0.0.0.0"),
        eou_port=int(e.get("EOU_SERVICE_PORT", "8970")),
    )
