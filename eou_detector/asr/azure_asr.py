import threading
from typing import List, Optional
from .base import ASR
from eou_detector.config import Settings
from eou_detector.types import Partial


class AzureSpeechASR(ASR):
    """Azure Speech STT mirroring the prod CareCallHouseMade config, with an
    added `recognizing` handler to surface real-time partials for the lexical
    branch. Credentials come from Settings (AZURE_STT_API_KEY / _REGION)."""

    def __init__(self, settings: Settings,
                 phrase_list: Optional[List[str]] = None,
                 segmentation_silence_ms: int = 500):
        self._s = settings
        self._phrase_list = phrase_list or []
        self._seg_ms = segmentation_silence_ms
        self._partial = Partial(text="")
        self._lock = threading.Lock()
        self._recognizer = None
        self._push_stream = None

    # --- handler logic, unit-testable without the SDK ---
    def _on_recognizing_text(self, text: str, is_final: bool) -> None:
        with self._lock:
            self._partial = Partial(text=text, is_final=is_final)

    def latest_partial(self) -> Partial:
        with self._lock:
            return self._partial

    async def start(self) -> None:
        if not self._s.azure_key or not self._s.azure_region:
            raise RuntimeError("AZURE_STT_API_KEY / AZURE_STT_REGION not set")
        import azure.cognitiveservices.speech as speechsdk

        speech_config = speechsdk.SpeechConfig(
            subscription=self._s.azure_key, region=self._s.azure_region)
        speech_config.speech_recognition_language = self._s.lang
        speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
            str(self._seg_ms))
        speech_config.output_format = speechsdk.OutputFormat.Detailed

        fmt = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self._s.sample_rate, bits_per_sample=16, channels=1)
        self._push_stream = speechsdk.audio.PushAudioInputStream(stream_format=fmt)
        audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config)

        if self._phrase_list:
            pl = speechsdk.PhraseListGrammar.from_recognizer(self._recognizer)
            for ph in self._phrase_list:
                pl.addPhrase(ph)

        self._recognizer.recognizing.connect(
            lambda evt: self._on_recognizing_text(evt.result.text, is_final=False))
        self._recognizer.recognized.connect(
            lambda evt: self._on_recognizing_text(evt.result.text, is_final=True))
        self._recognizer.start_continuous_recognition_async().get()

    def send_audio(self, frame: bytes) -> None:
        if self._push_stream is not None:
            self._push_stream.write(frame)

    async def stop(self) -> None:
        if self._recognizer is not None:
            self._recognizer.stop_continuous_recognition_async().get()
        if self._push_stream is not None:
            self._push_stream.close()
