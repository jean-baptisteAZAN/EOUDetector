import numpy as np
from eou_detector.audio.ring_buffer import RingBuffer


def test_extend_and_snapshot_recent():
    rb = RingBuffer(capacity_samples=10)
    rb.extend(np.arange(1, 6, dtype=np.int16))   # 1..5
    snap = rb.snapshot(3)
    assert snap.tolist() == [3, 4, 5]


def test_snapshot_left_pads_when_short():
    rb = RingBuffer(capacity_samples=10)
    rb.extend(np.array([7, 8], dtype=np.int16))
    snap = rb.snapshot(4)
    assert snap.tolist() == [0, 0, 7, 8]


def test_overwrites_when_over_capacity():
    rb = RingBuffer(capacity_samples=4)
    rb.extend(np.arange(1, 7, dtype=np.int16))   # 1..6, only last 4 kept
    assert rb.snapshot(4).tolist() == [3, 4, 5, 6]
    assert len(rb) == 4
