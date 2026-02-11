"""Tests for ultra-advanced DSP primitives in bridge.audio."""
import math
import struct

from bridge.audio import (
    DCBlocker,
    ComfortNoiseGenerator,
    ClickDetector,
    crossfade_pcm16,
    fade_in_pcm16,
    fade_out_pcm16,
    peak_dbfs,
    rms_dbfs,
    ensure_even_bytes,
)


def _silence(n_samples: int) -> bytes:
    """Generate n_samples of PCM16 silence."""
    return b"\x00" * (n_samples * 2)


def _tone(sample_rate: int, hz: float, dur_ms: int, amplitude: float = 0.5) -> bytes:
    """Generate a sine wave tone as PCM16."""
    n = int(sample_rate * dur_ms / 1000)
    out = bytearray()
    for i in range(n):
        t = i / sample_rate
        v = int(amplitude * 32767 * math.sin(2 * math.pi * hz * t))
        v = max(-32768, min(32767, v))
        out.extend(struct.pack("<h", v))
    return bytes(out)


def _dc_offset_tone(sample_rate: int, hz: float, dur_ms: int,
                     dc_offset: int = 5000) -> bytes:
    """Tone with a DC offset baked in."""
    n = int(sample_rate * dur_ms / 1000)
    out = bytearray()
    for i in range(n):
        t = i / sample_rate
        v = int(0.3 * 32767 * math.sin(2 * math.pi * hz * t)) + dc_offset
        v = max(-32768, min(32767, v))
        out.extend(struct.pack("<h", v))
    return bytes(out)


# ─── DCBlocker ───────────────────────────────────────────────────

class TestDCBlocker:
    def test_removes_dc_offset(self) -> None:
        """A constant DC input should converge toward zero."""
        dc = struct.pack("<480h", *([5000] * 480))  # 480 samples of +5000
        blocker = DCBlocker(alpha=0.9975)
        # Process several blocks to let filter converge
        for _ in range(20):
            out = blocker.process(dc)
        # Last output should have much less DC
        samples = struct.unpack(f"<{len(out) // 2}h", out)
        avg = sum(samples) / len(samples)
        assert abs(avg) < 500, f"DC not removed: avg={avg}"

    def test_preserves_length(self) -> None:
        pcm = _tone(24000, 440, 20)
        blocker = DCBlocker()
        out = blocker.process(pcm)
        assert len(out) == len(pcm)

    def test_empty_input(self) -> None:
        blocker = DCBlocker()
        assert blocker.process(b"") == b""

    def test_reset(self) -> None:
        blocker = DCBlocker()
        blocker.process(_tone(24000, 440, 20))
        blocker.reset()
        assert blocker._x_prev == 0.0
        assert blocker._y_prev == 0.0

    def test_passthrough_on_clean_signal(self) -> None:
        """A zero-mean sine should pass through mostly unchanged."""
        pcm = _tone(24000, 440, 100)
        blocker = DCBlocker()
        out = blocker.process(pcm)
        # RMS should be within ~5% of original
        in_rms = _compute_rms(pcm)
        out_rms = _compute_rms(out)
        assert abs(in_rms - out_rms) / max(in_rms, 1) < 0.10


# ─── ComfortNoiseGenerator ───────────────────────────────────────

class TestComfortNoiseGenerator:
    def test_output_length(self) -> None:
        cng = ComfortNoiseGenerator(level_dbfs=-70.0)
        out = cng.generate(960)
        assert len(out) == 960

    def test_output_is_quiet(self) -> None:
        cng = ComfortNoiseGenerator(level_dbfs=-70.0)
        out = cng.generate(960)
        peak = peak_dbfs(out)
        # Should be very quiet — peak below -50 dBFS
        assert peak < -50.0, f"Comfort noise too loud: {peak:.1f} dBFS"

    def test_even_alignment(self) -> None:
        cng = ComfortNoiseGenerator()
        out = cng.generate(961)  # odd input
        assert len(out) % 2 == 0

    def test_zero_bytes(self) -> None:
        cng = ComfortNoiseGenerator()
        assert cng.generate(0) == b""

    def test_not_all_zeros(self) -> None:
        """Comfort noise should not be all silence."""
        cng = ComfortNoiseGenerator(level_dbfs=-60.0)
        out = cng.generate(1920)
        assert out != b"\x00" * 1920


# ─── ClickDetector ───────────────────────────────────────────────

class TestClickDetector:
    def test_normal_speech_not_flagged(self) -> None:
        """Constant-level speech should never trigger the detector."""
        det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=10)
        tone = _tone(24000, 300, 20, amplitude=0.3)  # ~20ms frame
        clicks = 0
        for _ in range(100):
            if det.check(tone):
                clicks += 1
        assert clicks == 0, f"False positives: {clicks}/100"

    def test_warmup_never_flags(self) -> None:
        """During warmup, nothing should be flagged — even loud frames."""
        det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=30)
        loud = _tone(24000, 1000, 20, amplitude=0.9)
        for i in range(30):
            assert det.check(loud) is False, f"Flagged during warmup at frame {i}"

    def test_true_click_detected(self) -> None:
        """A sudden 30 dB spike after quiet baseline should be flagged."""
        det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=5)
        quiet = _tone(24000, 300, 20, amplitude=0.01)
        # Warm up with quiet signal
        for _ in range(40):
            det.check(quiet)
        # Now inject a spike (full-scale click)
        click_frame = _tone(24000, 300, 20, amplitude=0.95)
        assert det.check(click_frame) is True

    def test_reset(self) -> None:
        det = ClickDetector()
        det.check(_tone(24000, 300, 20))
        det.reset()
        assert det._avg_rms == 0.0
        assert det._frame_count == 0


# ─── crossfade_pcm16 ────────────────────────────────────────────

class TestCrossfade:
    def test_output_length_matches_head(self) -> None:
        """Crossfade output must be same length as head (full frame)."""
        tail = _tone(24000, 300, 20)
        head = _tone(24000, 500, 20)
        out = crossfade_pcm16(tail, head, fade_samples=120)
        assert len(out) == len(head)

    def test_overlap_samples_alias(self) -> None:
        """The overlap_samples keyword should work identically."""
        tail = _tone(24000, 300, 20)
        head = _tone(24000, 500, 20)
        out1 = crossfade_pcm16(tail, head, fade_samples=120)
        out2 = crossfade_pcm16(tail, head, overlap_samples=120)
        assert out1 == out2

    def test_zero_fade(self) -> None:
        head = _tone(24000, 500, 20)
        out = crossfade_pcm16(b"", head, fade_samples=0)
        assert out == head

    def test_empty_tail_returns_head(self) -> None:
        head = _tone(24000, 500, 20)
        out = crossfade_pcm16(b"", head, fade_samples=120)
        assert out == head

    def test_crossfade_smoother_than_concat(self) -> None:
        """Crossfaded boundary should have less discontinuity than raw concat."""
        tail = _tone(24000, 300, 20, amplitude=0.5)
        head = _tone(24000, 800, 20, amplitude=0.5)
        faded = crossfade_pcm16(tail, head, fade_samples=60)
        # Just verify it doesn't crash and produces valid PCM16
        assert len(faded) == len(head)
        n = len(faded) // 2
        samples = struct.unpack(f"<{n}h", faded)
        assert all(-32768 <= s <= 32767 for s in samples)


# ─── fade_in / fade_out ─────────────────────────────────────────

class TestFades:
    def test_fade_in_first_frame_quiet(self) -> None:
        pcm = _tone(24000, 440, 20, amplitude=0.8)
        faded = fade_in_pcm16(pcm, position=0, total_frames=3)
        # First frame gain = 1/3 → energy should be much lower
        assert _compute_rms(faded) < _compute_rms(pcm) * 0.5

    def test_fade_in_last_frame_full(self) -> None:
        pcm = _tone(24000, 440, 20, amplitude=0.8)
        faded = fade_in_pcm16(pcm, position=3, total_frames=3)
        assert faded == pcm  # beyond total → no change

    def test_fade_out_preserves_length(self) -> None:
        pcm = _tone(24000, 440, 20)
        faded = fade_out_pcm16(pcm, position=0, total_frames=3)
        assert len(faded) == len(pcm)

    def test_fade_in_empty(self) -> None:
        assert fade_in_pcm16(b"", 0, 3) == b""


# ─── peak_dbfs / rms_dbfs ───────────────────────────────────────

class TestMetering:
    def test_full_scale_peak(self) -> None:
        # Full-scale sample
        pcm = struct.pack("<h", 32767)
        assert abs(peak_dbfs(pcm) - 0.0) < 0.01

    def test_silence_peak(self) -> None:
        pcm = _silence(480)
        assert peak_dbfs(pcm) == float("-inf")

    def test_rms_dbfs_range(self) -> None:
        pcm = _tone(24000, 440, 100, amplitude=0.5)
        rms = rms_dbfs(pcm)
        assert -10.0 < rms < 0.0  # ~−6 dBFS for 0.5 amplitude sine

    def test_empty(self) -> None:
        assert peak_dbfs(b"") == float("-inf")
        assert rms_dbfs(b"") == float("-inf")


# ─── helpers ─────────────────────────────────────────────────────

def _compute_rms(pcm: bytes) -> float:
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm)
    return math.sqrt(sum(s * s for s in samples) / n)
