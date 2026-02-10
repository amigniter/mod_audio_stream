import math

from bridge.resample import PCM16Resampler, guess_pcm16_duration_ms


def _tone_pcm16_mono(sample_rate: int, hz: float, dur_ms: int) -> bytes:
    # Simple sine wave in PCM16
    n = int(sample_rate * dur_ms / 1000)
    out = bytearray()
    for i in range(n):
        t = i / sample_rate
        v = int(0.2 * 32767 * math.sin(2 * math.pi * hz * t))
        out.extend(int(v).to_bytes(2, "little", signed=True))
    return bytes(out)


def test_guess_duration_ms() -> None:
    pcm = _tone_pcm16_mono(8000, 440.0, 100)
    ms = guess_pcm16_duration_ms(len(pcm), 8000, 1)
    assert 95.0 <= ms <= 105.0


def test_resample_8k_to_16k_length_scales() -> None:
    pcm8 = _tone_pcm16_mono(8000, 440.0, 200)
    r = PCM16Resampler(src_rate=8000, dst_rate=16000, src_channels=1, dst_channels=1)
    pcm16 = r.convert(pcm8)
    # Expect about 2x bytes (allow for filter transients)
    assert len(pcm16) > len(pcm8) * 1.7
    assert len(pcm16) < len(pcm8) * 2.3


def test_stateful_resample_matches_whole() -> None:
    pcm8 = _tone_pcm16_mono(8000, 440.0, 400)
    whole = PCM16Resampler(src_rate=8000, dst_rate=16000).convert(pcm8)

    r = PCM16Resampler(src_rate=8000, dst_rate=16000)
    a = r.convert(pcm8[: len(pcm8) // 2])
    b = r.convert(pcm8[len(pcm8) // 2 :])
    chunked = a + b

    # Outputs won't be byte-identical due to chunk boundaries, but lengths should be close.
    assert abs(len(chunked) - len(whole)) <= 2000


def test_resample_8k_to_24k_length_scales() -> None:
    """Production path: FS 8kHz -> OpenAI 24kHz. Expect ~3x bytes."""
    pcm8 = _tone_pcm16_mono(8000, 440.0, 200)
    r = PCM16Resampler(src_rate=8000, dst_rate=24000, src_channels=1, dst_channels=1)
    pcm24 = r.convert(pcm8)
    # 3x ratio, allow for filter transients
    assert len(pcm24) > len(pcm8) * 2.5
    assert len(pcm24) < len(pcm8) * 3.5


def test_resample_noop_same_rate() -> None:
    """No resample when src == dst."""
    pcm = _tone_pcm16_mono(24000, 440.0, 100)
    r = PCM16Resampler(src_rate=24000, dst_rate=24000)
    out = r.convert(pcm)
    assert out == pcm
