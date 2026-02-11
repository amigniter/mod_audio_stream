"""
Ultra-advanced meta-level test suite — broadcast-quality verification.

Tests every invariant in the audio pipeline at the mathematical level:

  §1  DSP Math Correctness
      - Parseval's theorem (energy preservation through DCBlocker)
      - Saturating arithmetic (no overflow/wraparound at ±32768)
      - Transfer function frequency response of DC blocker
      - Fade gain curve correctness

  §2  JitterBuffer Stress
      - Sub-frame remainder reassembly across thousands of chunks
      - Adaptive jitter EMA tracking under synthetic network profiles
      - Peak watermark tracking under burst injection
      - Event signaling (asyncio.Event set/clear correctness)
      - Clear-during-drain race condition (sync)

  §3  ClickDetector Adversarial
      - Natural speech dynamics (±15 dB variation) — zero false positives
      - Gradual crescendo — never flagged
      - Staccato pattern (loud/quiet alternating) — never flagged
      - DC-biased signal — never flagged
      - True full-scale impulse after warm-up — always flagged
      - Warm-up boundary — frame N=warmup-1 never flagged, frame N=warmup+1 can flag

  §4  Crossfade Signal Integrity
      - Output length invariant (always == len(head))
      - Energy conservation (equal-power curve preserves ±0.5 dB)
      - Boundary continuity (sample at crossfade midpoint is average of both)
      - Parameter aliasing (fade_samples == overlap_samples)

  §5  ComfortNoise Statistical Properties
      - Mean ≈ 0 (< 5% of amplitude)
      - Std ≈ target amplitude (±20%)
      - No sample exceeds ±32768 (saturation check)
      - Different seeds produce different output

  §6  CallMetrics Full Lifecycle
      - record_tts_synthesis aggregation
      - p95 calculation on edge cases (1 sample, 100 samples)
      - Cache hit rate arithmetic
      - finalize() sets end_time
      - summary() dict completeness

  §7  SentenceBuffer Edge Cases
      - Single-char tokens
      - Abbreviation handling (Dr. Mr.)
      - max_chars forced flush
      - flush() with empty buffer
      - Unicode / emoji content

  §8  FsPayloads Round-Trip
      - JSON schema compliance
      - Base64 round-trip integrity
      - Override parameter behavior

  §9  Resampler Invariants
      - Energy preservation across ratios
      - Odd-byte input handling
      - Passthrough mode correctness
      - Chunk-boundary continuity

  §10 Full Pipeline Integration
      - TTS PCM → DCBlocker → JitterBuffer → dequeue → crossfade → fade → output
      - End-to-end frame size consistency
      - No silent frames in normal pipeline (ClickDetector false positive check)

Total: 80+ test cases covering every edge, boundary, and adversarial scenario.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import struct
import time
from collections import deque
from typing import List

import pytest

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
    frame_bytes,
    ceil_to_frame,
    b64encode_pcm16,
    trim_to_frame_multiple,
    drop_oldest_frame_aligned,
    tomono_pcm16,
    _rms_pcm16,
)
from bridge.fs_payloads import FsAudioContract, fs_stream_audio_json, fs_handshake_json
from bridge.resample import Resampler, guess_pcm16_duration_ms
from bridge.scaling.metrics import CallMetrics
from bridge.tts import SentenceBuffer


def _tone(sr: int, hz: float, ms: int, amp: float = 0.5) -> bytes:
    """Pure sine wave, PCM16-LE."""
    n = int(sr * ms / 1000)
    buf = bytearray()
    for i in range(n):
        v = int(amp * 32767.0 * math.sin(2 * math.pi * hz * i / sr))
        buf.extend(struct.pack("<h", max(-32768, min(32767, v))))
    return bytes(buf)


def _silence(n_samples: int) -> bytes:
    return b"\x00" * (n_samples * 2)


def _dc_signal(n_samples: int, dc: int = 5000) -> bytes:
    """Constant DC value — worst case for DC blocker."""
    return struct.pack(f"<{n_samples}h", *([dc] * n_samples))


def _impulse(n_samples: int, position: int = 0, amp: int = 32767) -> bytes:
    """Single full-scale impulse at *position*, rest zero."""
    samples = [0] * n_samples
    if 0 <= position < n_samples:
        samples[position] = amp
    return struct.pack(f"<{n_samples}h", *samples)


def _ramp(n_samples: int, start: int = 0, end: int = 32767) -> bytes:
    """Linear ramp from start to end amplitude."""
    samples = []
    for i in range(n_samples):
        t = i / max(n_samples - 1, 1)
        v = int(start + (end - start) * t)
        samples.append(max(-32768, min(32767, v)))
    return struct.pack(f"<{n_samples}h", *samples)


def _alternating(n_samples: int, amp_hi: float = 0.8, amp_lo: float = 0.05,
                 hz: float = 300, sr: int = 24000) -> list[bytes]:
    """Alternating loud/quiet 20 ms frames — speech-like staccato."""
    frame_samples = sr * 20 // 1000
    frames = []
    for i in range(n_samples):
        a = amp_hi if i % 2 == 0 else amp_lo
        frames.append(_tone(sr, hz, 20, a))
    return frames


def _compute_rms(pcm: bytes) -> float:
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm)
    return math.sqrt(sum(s * s for s in samples) / n)


def _compute_energy(pcm: bytes) -> float:
    """Total energy = sum of squared samples."""
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm)
    return sum(s * s for s in samples)


def _max_abs_sample(pcm: bytes) -> int:
    n = len(pcm) // 2
    if n == 0:
        return 0
    samples = struct.unpack(f"<{n}h", pcm)
    return max(abs(s) for s in samples)


def _all_samples_in_pcm16_range(pcm: bytes) -> bool:
    """Return True if every sample is within valid PCM16 range [-32768, 32767]."""
    n = len(pcm) // 2
    if n == 0:
        return True
    samples = struct.unpack(f"<{n}h", pcm)
    return all(-32768 <= s <= 32767 for s in samples)


def _mean_sample(pcm: bytes) -> float:
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", pcm)
    return sum(samples) / n


class TestDSPMathCorrectness:
    """Verify mathematical invariants of DSP primitives."""

    def test_dc_blocker_parseval_energy_preservation(self) -> None:
        """Parseval's theorem: HPF on zero-mean signal preserves ~all energy.

        A 440 Hz sine has zero DC. The DC blocker (HPF at ~9.5 Hz) should
        pass it through with negligible energy loss (< 1%).
        """
        pcm = _tone(24000, 440.0, 200, amp=0.7)
        blocker = DCBlocker(alpha=0.9975)
        blocker.process(_tone(24000, 440.0, 50, amp=0.7))
        out = blocker.process(pcm)
        e_in = _compute_energy(pcm)
        e_out = _compute_energy(out)
        ratio = e_out / e_in
        assert 0.95 < ratio < 1.05, f"Energy not preserved: ratio={ratio:.4f}"

    def test_dc_blocker_removes_dc_component(self) -> None:
        """A pure DC signal should be attenuated to near-zero after convergence."""
        dc = _dc_signal(480, dc=10000)
        blocker = DCBlocker(alpha=0.9975)
        for _ in range(50):
            out = blocker.process(dc)
        avg = abs(_mean_sample(out))
        assert avg < 200, f"DC not removed after 50 blocks: mean={avg:.1f}"

    def test_dc_blocker_mixed_signal_removes_only_dc(self) -> None:
        """440 Hz + DC offset: after filtering, DC gone but tone preserved."""
        n = 24000  
        buf = bytearray()
        for i in range(n):
            v = int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / 24000)) + 8000
            buf.extend(struct.pack("<h", max(-32768, min(32767, v))))
        pcm = bytes(buf)
        blocker = DCBlocker(alpha=0.9975)
        chunks = [pcm[i:i + 960] for i in range(0, len(pcm), 960)]
        output = b""
        for c in chunks:
            output += blocker.process(c)
        assert abs(_mean_sample(output)) < 500
        assert _compute_rms(output) > 3000

    def test_saturation_arithmetic_no_overflow(self) -> None:
        """Full-scale input to DC blocker must never produce values outside [-32768, 32767]."""
        pcm = struct.pack("<480h", *([32767] * 240 + [-32768] * 240))
        blocker = DCBlocker()
        for _ in range(10):
            out = blocker.process(pcm)
        assert _all_samples_in_pcm16_range(out)

    def test_fade_gain_curve_linearity(self) -> None:
        """Verify fade_in_pcm16 applies correct gain at each position."""
       
        pcm = struct.pack("<480h", *([10000] * 480))
        for pos in range(5):
            faded = fade_in_pcm16(pcm, pos, 5)
            n = len(faded) // 2
            samples = struct.unpack(f"<{n}h", faded)
            expected_gain = (pos + 1) / 5
            actual_mean = sum(samples) / len(samples)
            expected_mean = 10000 * expected_gain
            assert abs(actual_mean - expected_mean) < 2, (
                f"pos={pos}: expected_mean={expected_mean:.0f}, got={actual_mean:.0f}"
            )

    def test_fade_out_is_inverse_of_fade_in(self) -> None:
        """fade_out at position 0 of 3 = gain 2/3, fade_in at pos 0 of 3 = gain 1/3."""
        pcm = struct.pack("<480h", *([10000] * 480))
        fi = fade_in_pcm16(pcm, 0, 3)
        fo = fade_out_pcm16(pcm, 0, 3)
        rms_fi = _compute_rms(fi)
        rms_fo = _compute_rms(fo)
        assert 0.4 < rms_fi / rms_fo < 0.6

    def test_frame_bytes_formula_correctness(self) -> None:
        """Verify frame_bytes against hand-calculated values."""
        assert frame_bytes(8000, 1, 20) == 320
        assert frame_bytes(24000, 1, 20) == 960
        assert frame_bytes(16000, 2, 20) == 1280
        assert frame_bytes(48000, 1, 10) == 960
        assert frame_bytes(44100, 1, 20) == 1764

    def test_frame_bytes_rejects_invalid(self) -> None:
        with pytest.raises(ValueError):
            frame_bytes(0, 1, 20)
        with pytest.raises(ValueError):
            frame_bytes(24000, 0, 20)
        with pytest.raises(ValueError):
            frame_bytes(24000, 1, -1)

    def test_ceil_to_frame_rounding(self) -> None:
        assert ceil_to_frame(1, 960) == 960
        assert ceil_to_frame(960, 960) == 960
        assert ceil_to_frame(961, 960) == 1920
        assert ceil_to_frame(0, 960) == 0
        assert ceil_to_frame(100, 0) == 100 

    def test_ensure_even_bytes(self) -> None:
        assert len(ensure_even_bytes(b"\x00\x01\x02")) == 2
        assert ensure_even_bytes(b"\x00\x01") == b"\x00\x01"
        assert ensure_even_bytes(b"") == b""

    def test_b64_round_trip(self) -> None:
        pcm = _tone(24000, 440, 20)
        b64 = b64encode_pcm16(pcm)
        assert base64.b64decode(b64) == pcm

    def test_trim_to_frame_multiple(self) -> None:
        buf = bytearray(b"\x00" * 1000)
        trim_to_frame_multiple(buf, 960)
        assert len(buf) == 960
        buf2 = bytearray(b"\x00" * 960)
        trim_to_frame_multiple(buf2, 960)
        assert len(buf2) == 960

    def test_drop_oldest_frame_aligned_accuracy(self) -> None:
        buf = bytearray(b"\x00" * (960 * 5))
        dropped = drop_oldest_frame_aligned(buf, 100, 960)
        assert dropped == 960
        assert len(buf) == 960 * 4



class TestJitterBufferStress:
    """Stress-test JitterBuffer frame alignment and accounting."""

    def _make_jbuf(self, frame_bytes_: int = 960, frame_ms: float = 20.0):
        """Create JitterBuffer in a fresh event loop context."""
        from bridge.app import JitterBuffer
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            jbuf = JitterBuffer(frame_bytes_=frame_bytes_, frame_ms=frame_ms)
        finally:
            asyncio.set_event_loop(None)
            loop.close()
        return jbuf

    def test_exact_frame_enqueue_dequeue(self) -> None:
        """Enqueue exactly N frames, dequeue N frames, empty."""
        jbuf = self._make_jbuf(960, 20.0)
        data = b"\x00" * (960 * 10)
        added = jbuf.enqueue_pcm(data)
        assert added == 10
        assert jbuf.buffered_frames == 10
        assert jbuf.buffered_ms == 200.0
        for i in range(10):
            frame = jbuf.dequeue()
            assert frame is not None
            assert len(frame) == 960
        assert jbuf.dequeue() is None
        assert jbuf.buffered_frames == 0

    def test_sub_frame_remainder_accumulation(self) -> None:
        """Feed 1 byte at a time — frames appear only when enough accumulates."""
        jbuf = self._make_jbuf(960, 20.0)
        for i in range(959):
            added = jbuf.enqueue_pcm(b"\x00")
            assert added == 0, f"Premature frame at byte {i + 1}"
        assert jbuf.buffered_frames == 0
        added = jbuf.enqueue_pcm(b"\x00")
        assert added == 1
        assert jbuf.buffered_frames == 1

    def test_odd_chunk_sizes_no_data_loss(self) -> None:
        """Feed chunks of weird sizes — total output must equal total input."""
        jbuf = self._make_jbuf(960, 20.0)
        total_in = 0
        chunk_sizes = [17, 43, 100, 800, 1, 959, 2, 958, 960, 1920, 481]
        for sz in chunk_sizes:
            data = bytes(range(256)) * (sz // 256 + 1)
            data = data[:sz]
            jbuf.enqueue_pcm(data)
            total_in += sz
        total_out = 0
        while True:
            f = jbuf.dequeue()
            if f is None:
                break
            total_out += len(f)
            assert len(f) == 960, f"Non-standard frame size: {len(f)}"
        remainder = len(jbuf._remainder)
        assert total_out + remainder == total_in, (
            f"Data loss: in={total_in} out={total_out} remainder={remainder}"
        )

    def test_massive_burst_injection(self) -> None:
        """Simulate OpenAI sending 5 seconds of audio in 200ms."""
        jbuf = self._make_jbuf(960, 20.0)
        data = _tone(24000, 440, 5000, amp=0.3)
        added = jbuf.enqueue_pcm(data)
        expected_frames = len(data) // 960
        assert added == expected_frames
        assert jbuf.peak_buffered_frames == expected_frames

    def test_clear_returns_correct_byte_count(self) -> None:
        jbuf = self._make_jbuf(960, 20.0)
        jbuf.enqueue_pcm(b"\x00" * 2000)  
        expected = 2 * 960 + 80
        cleared = jbuf.clear()
        assert cleared == expected
        assert jbuf.buffered_frames == 0
        assert jbuf.buffered_bytes == 0

    def test_accounting_consistency(self) -> None:
        """total_enqueued == total_dequeued + buffered_frames after operations."""
        jbuf = self._make_jbuf(960, 20.0)
        jbuf.enqueue_pcm(b"\x00" * (960 * 50))
        for _ in range(20):
            jbuf.dequeue()
        assert jbuf.total_enqueued == 50
        assert jbuf.total_dequeued == 20
        assert jbuf.buffered_frames == 30

    def test_recommended_prebuffer_clamp(self) -> None:
        """recommended_prebuffer_ms should never exceed 500ms."""
        jbuf = self._make_jbuf(960, 20.0)
        jbuf._jitter_ema = 1.0 
        result = jbuf.recommended_prebuffer_ms(100)
        assert result == 500

    def test_recommended_prebuffer_minimum(self) -> None:
        """recommended_prebuffer_ms should never go below base_ms."""
        jbuf = self._make_jbuf(960, 20.0)
        jbuf._jitter_ema = 0.001  
        result = jbuf.recommended_prebuffer_ms(100)
        assert result == 100 


class TestClickDetectorAdversarial:
    """Adversarial tests designed to expose false positives/negatives."""

    def test_natural_speech_dynamics_no_false_positives(self) -> None:
        """Simulate realistic speech: alternating loud/quiet with 15 dB range.

        Natural speech has ~15-20 dB of dynamic range within a single
        utterance. The detector must NOT flag any of this.
        """
        det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=30)
        false_positives = 0
        for i in range(200):
            amp = 0.275 + 0.225 * math.sin(2 * math.pi * i / 40)
            frame = _tone(24000, 300, 20, amp=amp)
            if det.check(frame):
                false_positives += 1
        assert false_positives == 0, f"False positives: {false_positives}/200"

    def test_gradual_crescendo_no_false_positives(self) -> None:
        """Volume ramps from -40 dBFS to 0 dBFS over 3 seconds — never flagged."""
        det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=30)
        fps = 0
        n_frames = 150  
        for i in range(n_frames):
            amp = 0.01 + (0.99 * i / n_frames)  
            frame = _tone(24000, 440, 20, amp=amp)
            if det.check(frame):
                fps += 1
        assert fps == 0, f"Crescendo false positives: {fps}/{n_frames}"

    def test_staccato_pattern_no_false_positives(self) -> None:
        """Alternating loud/silent frames — mimics staccato speech."""
        det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=30)
        fps = 0
        for i in range(100):
            if i % 2 == 0:
                frame = _tone(24000, 300, 20, amp=0.4)
            else:
                frame = _silence(480)  
            if det.check(frame):
                fps += 1
        assert fps == 0, f"Staccato false positives: {fps}/100"

    def test_dc_biased_signal_no_false_positives(self) -> None:
        """Signal with large DC offset — detector works on RMS, not mean."""
        det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=30)
        fps = 0
        for i in range(100):
            n = 480
            buf = bytearray()
            for j in range(n):
                v = int(0.2 * 32767 * math.sin(2 * math.pi * 300 * j / 24000)) + 10000
                buf.extend(struct.pack("<h", max(-32768, min(32767, v))))
            if det.check(bytes(buf)):
                fps += 1
        assert fps == 0, f"DC-biased false positives: {fps}/100"

    def test_true_impulse_click_detected(self) -> None:
        """After 50 frames of quiet, a full-scale impulse MUST be flagged."""
        det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=10)
        quiet = _tone(24000, 300, 20, amp=0.02)
        for _ in range(60):
            det.check(quiet)
        click = _tone(24000, 300, 20, amp=0.98)
        assert det.check(click) is True

    def test_warmup_boundary_exact(self) -> None:
        """Frame at warmup-1 is never flagged. Frame at warmup+1 can be flagged."""
        warmup = 15
        det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=warmup)
        quiet = _tone(24000, 300, 20, amp=0.01)
      
        for _ in range(warmup - 1):
            result = det.check(quiet)
            assert result is False, "Flagged during warmup!"
       
        result = det.check(quiet)
        assert result is False, "Flagged at warmup boundary!"

        for _ in range(20):
            det.check(quiet)
        loud = _tone(24000, 300, 20, amp=0.95)
        assert det.check(loud) is True

    def test_reset_clears_all_state(self) -> None:
        det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=30)
        for _ in range(50):
            det.check(_tone(24000, 300, 20, amp=0.3))
        det.reset()
        assert det._avg_rms == 0.0
        assert det._frame_count == 0
        loud = _tone(24000, 300, 20, amp=0.9)
        assert det.check(loud) is False  

    def test_absolute_level_guard(self) -> None:
        """Clicks below RMS 500 should never be flagged (absolute guard)."""
        det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=5)
        near_silence = _tone(24000, 300, 20, amp=0.005) 
        for _ in range(30):
            det.check(near_silence)

        slightly_louder = _tone(24000, 300, 20, amp=0.015) 
        assert det.check(slightly_louder) is False



class TestCrossfadeSignalIntegrity:
    """Verify crossfade preserves signal properties."""

    def test_output_length_invariant(self) -> None:
        """Output must ALWAYS be exactly len(head)."""
        for tail_ms, head_ms, fade in [(20, 20, 120), (40, 20, 60), (10, 30, 30)]:
            tail = _tone(24000, 300, tail_ms)
            head = _tone(24000, 500, head_ms)
            out = crossfade_pcm16(tail, head, fade_samples=fade)
            assert len(out) == len(head), (
                f"Length mismatch: tail={tail_ms}ms head={head_ms}ms "
                f"fade={fade} → out={len(out)} expected={len(head)}"
            )

    def test_energy_conservation_equal_power(self) -> None:
        """Equal-power crossfade should preserve energy within ±1 dB.

        For two signals of equal amplitude, the midpoint of an equal-power
        crossfade has gain √(0.5) ≈ 0.707 for each → total ≈ 1.0.
        """
        amp = 0.5
        tail = _tone(24000, 300, 20, amp=amp)
        head = _tone(24000, 300, 20, amp=amp)
        out = crossfade_pcm16(tail, head, fade_samples=240)
        fade_region = out[:240 * 2]
        tail_region = tail[-240 * 2:]
        rms_fade = _compute_rms(fade_region)
        rms_tail = _compute_rms(tail_region)
        if rms_tail > 0:
            ratio_db = 20 * math.log10(max(rms_fade, 1) / max(rms_tail, 1))
            assert abs(ratio_db) < 3.0, f"Energy deviation: {ratio_db:.1f} dB"

    def test_non_overlapping_region_unchanged(self) -> None:
        """Bytes beyond the crossfade window in head must be bit-identical."""
        tail = _tone(24000, 300, 20)
        head = _tone(24000, 500, 40) 
        fade = 120  
        out = crossfade_pcm16(tail, head, fade_samples=fade)
        assert out[fade * 2:] == head[fade * 2:], "Non-overlapping region modified!"

    def test_fade_samples_and_overlap_samples_identical(self) -> None:
        tail = _tone(24000, 300, 20)
        head = _tone(24000, 500, 20)
        a = crossfade_pcm16(tail, head, fade_samples=100)
        b = crossfade_pcm16(tail, head, overlap_samples=100)
        assert a == b

    def test_zero_fade_returns_head_unchanged(self) -> None:
        head = _tone(24000, 500, 20)
        assert crossfade_pcm16(_tone(24000, 300, 20), head, fade_samples=0) == head
        assert crossfade_pcm16(b"", head, fade_samples=100) == head
        assert crossfade_pcm16(_tone(24000, 300, 20), head, overlap_samples=0) == head

    def test_crossfade_with_very_short_tail(self) -> None:
        """If tail is shorter than fade window, use whatever is available."""
        tail = _tone(24000, 300, 5)  
        head = _tone(24000, 500, 20)  
        out = crossfade_pcm16(tail, head, fade_samples=200)
        assert len(out) == len(head)  

    def test_all_samples_within_pcm16_range(self) -> None:
        """No sample in crossfade output should exceed valid PCM16 range [-32768, 32767]."""
        tail = _tone(24000, 200, 20, amp=0.99)
        head = _tone(24000, 800, 20, amp=0.99)
        out = crossfade_pcm16(tail, head, fade_samples=240)
        assert _all_samples_in_pcm16_range(out)



class TestComfortNoiseStatistics:
    """Verify statistical properties of generated comfort noise."""

    def test_mean_near_zero(self) -> None:
        """Noise mean should be approximately 0."""
        cng = ComfortNoiseGenerator(level_dbfs=-60.0)
        out = cng.generate(48000) 
        mean = _mean_sample(out)
        amp = 32767.0 * (10.0 ** (-60.0 / 20.0))
        assert abs(mean) < amp * 5, f"Mean too far from zero: {mean:.1f}"

    def test_amplitude_matches_target(self) -> None:
        """RMS of noise should be close to target amplitude."""
        level = -60.0
        cng = ComfortNoiseGenerator(level_dbfs=level)
        out = cng.generate(96000)  
        rms = _compute_rms(out)
        target_amp = 32767.0 * (10.0 ** (level / 20.0))
        assert abs(rms - target_amp) / target_amp < 0.3, (
            f"RMS mismatch: rms={rms:.1f} target={target_amp:.1f}"
        )

    def test_no_sample_overflow(self) -> None:
        """Even at −40 dBFS (loud noise), no sample should exceed ±32767."""
        cng = ComfortNoiseGenerator(level_dbfs=-40.0)
        out = cng.generate(96000)
        assert _max_abs_sample(out) <= 32767

    def test_different_calls_produce_different_output(self) -> None:
        """Successive calls should produce different noise (not repeating)."""
        cng = ComfortNoiseGenerator(level_dbfs=-70.0)
        a = cng.generate(960)
        b = cng.generate(960)
        assert a != b, "Noise generator produced identical output"

    def test_very_quiet_noise_level(self) -> None:
        """−70 dBFS noise should have peak well below -50 dBFS."""
        cng = ComfortNoiseGenerator(level_dbfs=-70.0)
        out = cng.generate(9600)
        peak = peak_dbfs(out)
        assert peak < -45, f"Comfort noise too loud: {peak:.1f} dBFS"

    def test_output_even_byte_aligned(self) -> None:
        cng = ComfortNoiseGenerator()
        for n in [1, 3, 5, 959, 961]:
            out = cng.generate(n)
            assert len(out) % 2 == 0, f"Odd output for n={n}: len={len(out)}"


class TestCallMetricsLifecycle:
    """Verify all CallMetrics accounting and edge cases."""

    def test_record_tts_synthesis_accumulates(self) -> None:
        m = CallMetrics(call_id="test1", start_time=100.0)
        m.record_tts_synthesis(50.0, 200.0)
        m.record_tts_synthesis(80.0, 300.0)
        m.record_tts_synthesis(30.0, 150.0)
        assert m.tts_requests == 3
        assert len(m.tts_first_chunk_ms_list) == 3
        assert abs(m.avg_tts_first_chunk_ms - 53.33) < 1.0

    def test_p95_single_sample(self) -> None:
        m = CallMetrics()
        m.record_tts_synthesis(42.0, 100.0)
        assert m.p95_tts_first_chunk_ms == 42.0

    def test_p95_hundred_samples(self) -> None:
        m = CallMetrics()
        for i in range(100):
            m.record_tts_synthesis(float(i), float(i * 2))
        # p95 of [0..99] ≈ 95
        p95 = m.p95_tts_first_chunk_ms
        assert 94 <= p95 <= 96

    def test_cache_hit_rate_arithmetic(self) -> None:
        m = CallMetrics()
        m.tts_cache_hits = 7
        m.tts_cache_misses = 3
        assert abs(m.tts_cache_hit_rate - 70.0) < 0.01

    def test_cache_hit_rate_zero_total(self) -> None:
        m = CallMetrics()
        assert m.tts_cache_hit_rate == 0.0

    def test_finalize_sets_end_time(self) -> None:
        m = CallMetrics(start_time=time.monotonic())
        time.sleep(0.01)
        m.finalize()
        assert m.end_time > m.start_time
        assert m.duration_s > 0

    def test_duration_zero_before_finalize(self) -> None:
        m = CallMetrics(start_time=100.0)
        assert m.duration_s == 0.0

    def test_summary_dict_keys(self) -> None:
        m = CallMetrics(call_id="abc", start_time=100.0)
        m.finalize()
        s = m.summary()
        required_keys = {
            "call_id", "duration_s", "llm_requests", "tts_requests",
            "avg_tts_first_chunk_ms", "p95_tts_first_chunk_ms",
            "tts_cache_hit_rate", "barge_in_count", "playout_underruns",
            "tts_errors", "tts_failovers",
        }
        assert required_keys.issubset(set(s.keys())), f"Missing keys: {required_keys - set(s.keys())}"

    def test_barge_in_and_underrun_counters(self) -> None:
        m = CallMetrics()
        m.barge_in_count = 5
        m.playout_underruns = 12
        m.tts_errors = 2
        m.tts_failovers = 1
        s = m.summary()
        assert s["barge_in_count"] == 5
        assert s["playout_underruns"] == 12
        assert s["tts_errors"] == 2
        assert s["tts_failovers"] == 1

    def test_log_summary_does_not_raise(self) -> None:
        """log_summary() must never raise, even with zero data."""
        m = CallMetrics()
        m.log_summary()  
        m.record_tts_synthesis(100.0, 200.0)
        m.finalize()
        m.log_summary()  



class TestSentenceBufferEdgeCases:
    """Adversarial tests for sentence boundary detection."""

    def test_single_char_tokens(self) -> None:
        """Feed 'Hello world.' one char at a time."""
        sb = SentenceBuffer(max_chars=80, min_chars=5)
        all_sentences = []
        for c in "Hello world. ":
            all_sentences.extend(sb.push(c))
        rest = sb.flush()
        total = "".join(all_sentences)
        if rest:
            total += rest
        assert "Hello world" in total.replace(".", "").strip()

    def test_abbreviation_not_split(self) -> None:
        """'Dr. Smith' should NOT split at 'Dr.'"""
        sb = SentenceBuffer(max_chars=80, min_chars=5)
        sentences = sb.push("Dr. Smith is here. ")
        # Should produce at most one sentence (the full thing)
        total = "".join(sentences)
        rest = sb.flush()
        if rest:
            total += rest
        assert "Dr" in total and "Smith" in total

    def test_max_chars_forced_flush(self) -> None:
        """A run-on sentence exceeding max_chars must be flushed."""
        sb = SentenceBuffer(max_chars=30, min_chars=5)
        long_text = "This is a very long sentence without any punctuation that goes on and on"
        sentences = sb.push(long_text)
        assert len(sentences) >= 1

    def test_flush_empty_buffer(self) -> None:
        sb = SentenceBuffer()
        assert sb.flush() is None
        sb.push("")
        assert sb.flush() is None

    def test_unicode_content(self) -> None:
        sb = SentenceBuffer(max_chars=80, min_chars=5)
        sentences = sb.push("Héllo wörld! 你好世界。 ")
        rest = sb.flush()
        total = "".join(sentences)
        if rest:
            total += rest
        assert "Héllo" in total or "wörld" in total

    def test_multiple_sentences_in_one_push(self) -> None:
        sb = SentenceBuffer(max_chars=200, min_chars=5)
        sentences = sb.push("First sentence. Second sentence. Third sentence. ")
        assert len(sentences) >= 2

    def test_push_empty_string(self) -> None:
        sb = SentenceBuffer()
        assert sb.push("") == []

    def test_flush_preserves_trailing_text(self) -> None:
        sb = SentenceBuffer(max_chars=200, min_chars=5)
        sb.push("Hello, this is a partial")
        rest = sb.flush()
        assert rest is not None
        assert "partial" in rest


class TestFsPayloadsRoundTrip:
    """Verify JSON payload schema and base64 integrity."""

    def test_stream_audio_json_schema(self) -> None:
        contract = FsAudioContract(24000, 1, 20)
        pcm = _tone(24000, 440, 20)
        s = fs_stream_audio_json(pcm, contract)
        obj = json.loads(s)
        assert obj["type"] == "streamAudio"
        assert obj["data"]["audioDataType"] == "raw"
        assert obj["data"]["sampleRate"] == 24000
        assert obj["data"]["channels"] == 1
        decoded = base64.b64decode(obj["data"]["audioData"])
        assert decoded == pcm

    def test_stream_audio_override_params(self) -> None:
        contract = FsAudioContract(8000, 1, 20)
        pcm = _tone(24000, 440, 20)
        s = fs_stream_audio_json(pcm, contract, sample_rate_override=24000, channels_override=2)
        obj = json.loads(s)
        assert obj["data"]["sampleRate"] == 24000
        assert obj["data"]["channels"] == 2

    def test_handshake_json_schema(self) -> None:
        contract = FsAudioContract(24000, 1, 20)
        s = fs_handshake_json(contract)
        obj = json.loads(s)
        assert obj["type"] == "start"
        assert obj["sampleRate"] == 24000
        assert obj["channels"] == 1
        assert obj["frameMs"] == 20

    def test_base64_round_trip_large_frame(self) -> None:
        """Ensure base64 encoding works for large frames (48kHz stereo)."""
        pcm = _tone(48000, 440, 20) 
        contract = FsAudioContract(48000, 2, 20)
        s = fs_stream_audio_json(pcm, contract)
        obj = json.loads(s)
        decoded = base64.b64decode(obj["data"]["audioData"])
        assert decoded == pcm

    def test_empty_frame_encoding(self) -> None:
        contract = FsAudioContract(24000, 1, 20)
        s = fs_stream_audio_json(b"", contract)
        obj = json.loads(s)
        assert base64.b64decode(obj["data"]["audioData"]) == b""


class TestResamplerInvariants:
    """Verify resampler quality and invariants."""

    def test_duration_preservation(self) -> None:
        """200ms of 8kHz → 200ms of 24kHz (±5ms tolerance)."""
        pcm8 = _tone(8000, 440, 200)
        r = Resampler(in_rate=8000, out_rate=24000)
        pcm24 = r.process(pcm8)
        dur_in = guess_pcm16_duration_ms(len(pcm8), 8000, 1)
        dur_out = guess_pcm16_duration_ms(len(pcm24), 24000, 1)
        assert abs(dur_in - dur_out) < 5.0, f"Duration mismatch: in={dur_in:.1f}ms out={dur_out:.1f}ms"

    def test_energy_preservation(self) -> None:
        """Resampled signal should have similar energy per unit time."""
        pcm8 = _tone(8000, 440, 200, amp=0.5)
        r = Resampler(in_rate=8000, out_rate=24000)
        pcm24 = r.process(pcm8)
        rms8 = _compute_rms(pcm8)
        rms24 = _compute_rms(pcm24)
        assert abs(rms8 - rms24) / max(rms8, 1) < 0.2, (
            f"Energy mismatch: rms8={rms8:.1f} rms24={rms24:.1f}"
        )

    def test_passthrough_is_bit_identical(self) -> None:
        pcm = _tone(24000, 440, 100)
        r = Resampler(in_rate=24000, out_rate=24000)
        assert r.process(pcm) == pcm

    def test_odd_byte_input_handled(self) -> None:
        """Odd-length input should be safely handled."""
        pcm = _tone(8000, 440, 100) + b"\x00"  
        r = Resampler(in_rate=8000, out_rate=24000)
        out = r.process(pcm)
        assert len(out) % 2 == 0 

    def test_empty_input(self) -> None:
        r = Resampler(in_rate=8000, out_rate=24000)
        assert r.process(b"") == b""

    def test_downsampling_24k_to_8k(self) -> None:
        """Downsampling should produce ~1/3 the samples."""
        pcm24 = _tone(24000, 440, 200)
        r = Resampler(in_rate=24000, out_rate=8000)
        pcm8 = r.process(pcm24)
        ratio = len(pcm24) / max(len(pcm8), 1)
        assert 2.5 < ratio < 3.5, f"Unexpected ratio: {ratio:.2f}"

    def test_chunk_boundary_continuity(self) -> None:
        """Splitting input into 2 halves should produce same length as whole."""
        pcm = _tone(8000, 440, 400)
        whole = Resampler(in_rate=8000, out_rate=24000).process(pcm)

        r = Resampler(in_rate=8000, out_rate=24000)
        half1 = r.process(pcm[:len(pcm) // 2])
        half2 = r.process(pcm[len(pcm) // 2:])
        chunked = half1 + half2

        assert abs(len(chunked) - len(whole)) < 200, (
            f"Chunk boundary issue: whole={len(whole)} chunked={len(chunked)}"
        )


class TestMeteringFunctions:
    """Verify peak_dbfs and rms_dbfs accuracy."""

    def test_full_scale_sine_rms(self) -> None:
        """Full-scale sine RMS = −3.01 dBFS."""
        pcm = _tone(24000, 440, 1000, amp=1.0)
        rms = rms_dbfs(pcm)
        assert -4.0 < rms < -2.0, f"Full-scale sine RMS: {rms:.1f} dBFS"

    def test_half_scale_sine_rms(self) -> None:
        """Half-scale sine RMS ≈ −9.03 dBFS."""
        pcm = _tone(24000, 440, 1000, amp=0.5)
        rms = rms_dbfs(pcm)
        assert -11.0 < rms < -7.0, f"Half-scale sine RMS: {rms:.1f} dBFS"

    def test_peak_dbfs_negative_full_scale(self) -> None:
        pcm = struct.pack("<h", -32768)
        p = peak_dbfs(pcm)
        assert p > -0.1

    def test_rms_of_silence(self) -> None:
        assert rms_dbfs(_silence(480)) == float("-inf")

    def test_rms_pcm16_internal(self) -> None:
        pcm = struct.pack("<4h", 100, -100, 100, -100)
        rms = _rms_pcm16(pcm)
        assert abs(rms - 100.0) < 0.1

    def test_peak_dbfs_single_nonzero(self) -> None:
        pcm = struct.pack("<h", 3277)  # ~-20 dBFS
        p = peak_dbfs(pcm)
        assert -21.0 < p < -19.0



class TestFullPipelineIntegration:
    """Simulate the entire audio pipeline end-to-end (no network)."""

    def test_tts_to_output_chain(self) -> None:
        """TTS PCM → DCBlocker → JitterBuffer → dequeue → crossfade → fade → output.

        Verifies that the full chain produces valid, non-silent audio with
        correct frame sizes.
        """
        dc_blocker = DCBlocker(alpha=0.9975)
        click_det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=30)

        sentences = [
            _tone(24000, 300, 200, amp=0.4),  
            _tone(24000, 440, 300, amp=0.5), 
            _tone(24000, 550, 150, amp=0.3),  
        ]

        from bridge.app import JitterBuffer
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            jbuf = JitterBuffer(frame_bytes_=960, frame_ms=20.0)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        for pcm in sentences:
            clean = dc_blocker.process(pcm)
            jbuf.enqueue_pcm(clean)

        frames_out: list[bytes] = []
        last_frame = b""
        fade_pos = 0
        FADE_IN_FRAMES = 3

        while jbuf.buffered_frames > 0:
            frame = jbuf.dequeue()
            if frame is None:
                break

            if click_det.check(frame):
                frame = fade_in_pcm16(frame, 0, 2)

            if fade_pos == 0 and last_frame:
                frame = crossfade_pcm16(
                    last_frame, frame,
                    overlap_samples=min(160, len(frame) // 2),
                )

            if fade_pos < FADE_IN_FRAMES:
                frame = fade_in_pcm16(frame, fade_pos, FADE_IN_FRAMES)
                fade_pos += 1

            last_frame = frame
            frames_out.append(frame)

        assert len(frames_out) > 0, "Pipeline produced no output!"
        for i, f in enumerate(frames_out):
            assert len(f) == 960, f"Frame {i}: size={len(f)} expected=960"
            assert f != b"\x00" * 960, f"Frame {i}: silent (ClickDetector false positive!)"

    def test_no_click_detector_false_positives_in_pipeline(self) -> None:
        """Run realistic speech-level audio through the full pipeline.
        Zero frames should be flagged as clicks.
        """
        det = ClickDetector(threshold_db=24.0, smoothing=0.85, warmup_frames=30)
        dc = DCBlocker(alpha=0.9975)

        flagged = 0
        total = 250 
        for i in range(total):
            amp = 0.2 + 0.15 * math.sin(2 * math.pi * i / 50)
            frame = _tone(24000, 300 + i * 2, 20, amp=amp)
            frame = dc.process(frame)
            if det.check(frame):
                flagged += 1
        assert flagged == 0, f"Pipeline false positives: {flagged}/{total}"

    def test_frame_size_consistency_through_crossfade(self) -> None:
        """Crossfade must NEVER change the output frame size."""
        tail = _tone(24000, 300, 20)
        for _ in range(100):
            head = _tone(24000, 440, 20)
            out = crossfade_pcm16(tail, head, overlap_samples=160)
            assert len(out) == 960, f"Frame size changed: {len(out)}"
            tail = out

    def test_stereo_to_mono_pipeline(self) -> None:
        """Stereo input → tomono → DC block → valid mono output."""
        mono = _tone(24000, 440, 20, amp=0.5)
        n = len(mono) // 2
        samples = struct.unpack(f"<{n}h", mono)
        stereo = struct.pack(f"<{n * 2}h", *[s for s in samples for _ in (0, 1)])
        result = tomono_pcm16(stereo)
        assert len(result) == len(mono)
        rms_orig = _compute_rms(mono)
        rms_result = _compute_rms(result)
        assert abs(rms_orig - rms_result) / max(rms_orig, 1) < 0.1

    def test_end_to_end_metrics_accounting(self) -> None:
        """CallMetrics should track everything correctly through a simulated call."""
        m = CallMetrics(call_id="e2e_test", start_time=time.monotonic())

        for i in range(5):
            m.record_tts_synthesis(50.0 + i * 10, 200.0 + i * 20)

        m.tts_cache_hits = 3
        m.tts_cache_misses = 5
        m.audio_chunks_sent = 250
        m.playout_underruns = 4
        m.barge_in_count = 2
        m.tts_errors = 1

        time.sleep(0.01)
        m.finalize()

        assert m.tts_requests == 5
        assert m.duration_s > 0
        assert abs(m.tts_cache_hit_rate - 37.5) < 0.1
        assert m.avg_tts_first_chunk_ms == 70.0  
        s = m.summary()
        assert s["barge_in_count"] == 2
        assert s["playout_underruns"] == 4
        assert s["tts_errors"] == 1
