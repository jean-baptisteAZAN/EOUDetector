import threading
import numpy as np


class RingBuffer:
    """Fixed-capacity rolling buffer of int16 PCM, safe across threads."""

    def __init__(self, capacity_samples: int):
        self._cap = int(capacity_samples)
        self._buf = np.zeros(self._cap, dtype=np.int16)
        self._size = 0
        self._lock = threading.Lock()

    def extend(self, samples: np.ndarray) -> None:
        s = np.asarray(samples, dtype=np.int16).reshape(-1)
        with self._lock:
            n = s.size
            if n >= self._cap:
                self._buf[:] = s[-self._cap:]
                self._size = self._cap
                return
            self._buf = np.roll(self._buf, -n)
            self._buf[-n:] = s
            self._size = min(self._cap, self._size + n)

    def snapshot(self, n_samples: int) -> np.ndarray:
        n = int(n_samples)
        with self._lock:
            avail = min(self._size, self._cap)
            data = self._buf[-avail:] if avail else np.zeros(0, dtype=np.int16)
        if data.size >= n:
            return data[-n:].copy()
        out = np.zeros(n, dtype=np.int16)
        if data.size:
            out[-data.size:] = data
        return out

    def __len__(self) -> int:
        with self._lock:
            return self._size
