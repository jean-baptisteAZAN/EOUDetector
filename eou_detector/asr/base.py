import abc
from eou_detector.types import Partial


class ASR(abc.ABC):
    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    def send_audio(self, frame: bytes) -> None: ...

    @abc.abstractmethod
    def latest_partial(self) -> Partial: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...
