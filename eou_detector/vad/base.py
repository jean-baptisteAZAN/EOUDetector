import abc


class VAD(abc.ABC):
    @abc.abstractmethod
    def process(self, frame: bytes) -> bool:
        """Return True if the frame contains speech."""

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state between utterances/streams."""
