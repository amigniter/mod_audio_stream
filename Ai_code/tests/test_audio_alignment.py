from bridge.audio import frame_bytes, drop_oldest_frame_aligned


def test_frame_bytes_16k_mono_20ms():
    assert frame_bytes(16000, 1, 20) == 640


def test_frame_bytes_8k_mono_20ms():
    """FS sends 8kHz mono -> 320 bytes per 20ms frame."""
    assert frame_bytes(8000, 1, 20) == 320


def test_frame_bytes_24k_mono_20ms():
    """OpenAI Realtime uses 24kHz -> 960 bytes per 20ms frame."""
    assert frame_bytes(24000, 1, 20) == 960


def test_drop_oldest_is_frame_aligned():
    frame = 640
    buf = bytearray(b"a" * (frame * 10))
    dropped = drop_oldest_frame_aligned(buf, drop_bytes=1, frame=frame)
    assert dropped == frame
    assert len(buf) == frame * 9


def test_drop_oldest_8k_frame_aligned():
    frame = 320  # 8kHz mono 20ms
    buf = bytearray(b"\x00" * (frame * 5))
    dropped = drop_oldest_frame_aligned(buf, drop_bytes=100, frame=frame)
    assert dropped == frame  # rounds up to one full frame
    assert len(buf) == frame * 4
