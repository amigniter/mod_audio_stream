from bridge.audio import frame_bytes, drop_oldest_frame_aligned


def test_frame_bytes_16k_mono_20ms():
    assert frame_bytes(16000, 1, 20) == 640


def test_drop_oldest_is_frame_aligned():
    frame = 640
    buf = bytearray(b"a" * (frame * 10))
    dropped = drop_oldest_frame_aligned(buf, drop_bytes=1, frame=frame)
    assert dropped == frame
    assert len(buf) == frame * 9
