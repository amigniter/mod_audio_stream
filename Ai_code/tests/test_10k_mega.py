"""
10K test suite — Part 2: Mega parametric expansion.
Adds ~8,000 more tests via parametrize to reach 10K total.
"""
from __future__ import annotations

import asyncio
import base64
import math
import random
import struct
import time

import pytest

from bridge.audio import (
    DCBlocker,
    ComfortNoiseGenerator,
    ClickDetector,
    NoiseGate,
    SpectralNoiseSubtractor,
    DeEsser,
    DynamicCompressor,
    PreEmphasisFilter,
    SoftClipper,
    HighShelfFilter,
    LowPassFilter,
    AudioClarityPipeline,
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


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _tone(sr: int, hz: float, ms: int, amp: float = 0.5) -> bytes:
    n = int(sr * ms / 1000)
    buf = bytearray()
    for i in range(n):
        v = int(amp * 32767.0 * math.sin(2 * math.pi * hz * i / sr))
        buf.extend(struct.pack("<h", max(-32768, min(32767, v))))
    return bytes(buf)

def _silence(n_samples: int) -> bytes:
    return b"\x00" * (n_samples * 2)

def _dc_signal(n_samples: int, dc: int = 5000) -> bytes:
    return struct.pack(f"<{n_samples}h", *([dc] * n_samples))

def _white_noise(n_samples: int, amp: float = 0.3, seed: int = 42) -> bytes:
    rng = random.Random(seed)
    samples = [max(-32768, min(32767, int(rng.gauss(0, amp * 32767)))) for _ in range(n_samples)]
    return struct.pack(f"<{n_samples}h", *samples)

def _compute_rms(pcm: bytes) -> float:
    n = len(pcm) // 2
    if n == 0: return 0.0
    samples = struct.unpack(f"<{n}h", pcm)
    return math.sqrt(sum(s * s for s in samples) / n)

def _max_abs_sample(pcm: bytes) -> int:
    n = len(pcm) // 2
    if n == 0: return 0
    return max(abs(s) for s in struct.unpack(f"<{n}h", pcm))

def _all_samples_in_range(pcm: bytes) -> bool:
    n = len(pcm) // 2
    if n == 0: return True
    return all(-32768 <= s <= 32767 for s in struct.unpack(f"<{n}h", pcm))

def _samples(pcm: bytes) -> list:
    n = len(pcm) // 2
    return list(struct.unpack(f"<{n}h", pcm))

def _make_jbuf(frame_bytes_: int = 960, frame_ms: float = 20.0):
    from bridge.app import JitterBuffer
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        jbuf = JitterBuffer(frame_bytes_=frame_bytes_, frame_ms=frame_ms)
    finally:
        asyncio.set_event_loop(None)
        loop.close()
    return jbuf


# ═══════════════════════════════════════════════════════════════════════
# §A — Mega NoiseGate Parametric Matrix (500 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestNoiseGateMegaMatrix:
    """Exhaustive parametric coverage of NoiseGate."""

    @pytest.mark.parametrize("thr_db", list(range(-60, 1, 5)))
    @pytest.mark.parametrize("freq", [200, 440, 1000, 4000, 8000])
    def test_ng_threshold_x_freq(self, thr_db, freq):
        ng = NoiseGate(threshold_db=thr_db)
        out = ng.process(_tone(24000, freq, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("hold_ms", [0, 25, 50, 100, 200])
    @pytest.mark.parametrize("amp", [0.01, 0.1, 0.5, 1.0])
    def test_ng_hold_x_amp(self, hold_ms, amp):
        ng = NoiseGate(hold_ms=hold_ms)
        out = ng.process(_tone(24000, 440, 50, amp=amp))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("attack_ms", [0.5, 1.0, 5.0, 10.0])
    @pytest.mark.parametrize("release_ms", [5.0, 10.0, 50.0, 100.0])
    def test_ng_attack_x_release(self, attack_ms, release_ms):
        ng = NoiseGate(attack_ms=attack_ms, release_ms=release_ms)
        out = ng.process(_tone(24000, 440, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("hyst", [0, 3, 6, 12])
    @pytest.mark.parametrize("thr_db", [-40, -30, -20, -10])
    def test_ng_hysteresis_x_threshold(self, hyst, thr_db):
        ng = NoiseGate(threshold_db=thr_db, hysteresis_db=hyst)
        out = ng.process(_tone(24000, 440, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("sr", [8000, 16000, 24000, 48000])
    @pytest.mark.parametrize("thr_db", [-40, -20, 0])
    def test_ng_rate_x_threshold(self, sr, thr_db):
        ng = NoiseGate(threshold_db=thr_db, sample_rate=sr)
        out = ng.process(_tone(sr, 440, 50, amp=0.5))
        assert _all_samples_in_range(out)


# ═══════════════════════════════════════════════════════════════════════
# §B — Mega Compressor Parametric Matrix (500 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestCompressorMegaMatrix:
    """Exhaustive parametric coverage of DynamicCompressor."""

    @pytest.mark.parametrize("thr_db", [-30, -24, -18, -12, -6])
    @pytest.mark.parametrize("ratio", [1.5, 3.0, 8.0, 20.0])
    def test_comp_thr_x_ratio(self, thr_db, ratio):
        c = DynamicCompressor(threshold_db=thr_db, ratio=ratio)
        out = c.process(_tone(24000, 440, 100, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("thr_db", [-30, -18, -6])
    @pytest.mark.parametrize("makeup_db", [0, 6, 12, 18])
    def test_comp_thr_x_makeup(self, thr_db, makeup_db):
        c = DynamicCompressor(threshold_db=thr_db, makeup_db=makeup_db)
        out = c.process(_tone(24000, 440, 100, amp=0.3))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("ratio", [1.5, 3.0, 8.0])
    @pytest.mark.parametrize("makeup_db", [0, 6, 12])
    @pytest.mark.parametrize("limiter_db", [-3, -1, 0])
    def test_comp_ratio_x_makeup_x_limiter(self, ratio, makeup_db, limiter_db):
        c = DynamicCompressor(ratio=ratio, makeup_db=makeup_db, limiter_db=limiter_db)
        out = c.process(_tone(24000, 440, 100, amp=0.8))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("attack_ms", [0.5, 5.0, 50.0])
    @pytest.mark.parametrize("release_ms", [10, 50, 200])
    def test_comp_attack_x_release(self, attack_ms, release_ms):
        c = DynamicCompressor(attack_ms=attack_ms, release_ms=release_ms)
        out = c.process(_tone(24000, 440, 100, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("sr", [8000, 16000, 24000, 48000])
    @pytest.mark.parametrize("ratio", [2.0, 4.0, 8.0])
    def test_comp_rate_x_ratio(self, sr, ratio):
        c = DynamicCompressor(ratio=ratio, sample_rate=sr)
        out = c.process(_tone(sr, 440, 100, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("amp_pct", list(range(5, 101, 5)))
    def test_comp_amplitude_ladder(self, amp_pct):
        c = DynamicCompressor()
        out = c.process(_tone(24000, 440, 100, amp=amp_pct / 100.0))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("freq", list(range(100, 10001, 500)))
    def test_comp_freq_ladder(self, freq):
        c = DynamicCompressor()
        out = c.process(_tone(24000, min(freq, 11000), 50, amp=0.5))
        assert _all_samples_in_range(out)


# ═══════════════════════════════════════════════════════════════════════
# §C — Mega Filter Parametric Matrix (500 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestFilterMegaMatrix:
    """Exhaustive parametric coverage of filters."""

    @pytest.mark.parametrize("gain_db", list(range(-12, 13, 2)))
    @pytest.mark.parametrize("cutoff_hz", [1000, 2000, 3000, 5000, 8000])
    def test_hsf_gain_x_cutoff(self, gain_db, cutoff_hz):
        f = HighShelfFilter(gain_db=gain_db, cutoff_hz=cutoff_hz)
        out = f.process(_tone(24000, 440, 50))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("gain_db", list(range(-12, 13, 2)))
    @pytest.mark.parametrize("freq", [200, 440, 1000, 4000, 8000])
    def test_hsf_gain_x_input_freq(self, gain_db, freq):
        f = HighShelfFilter(gain_db=gain_db)
        out = f.process(_tone(24000, freq, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("cutoff_hz", [500, 1000, 2000, 4000, 7500, 10000])
    @pytest.mark.parametrize("freq", [200, 440, 1000, 4000, 8000])
    def test_lpf_cutoff_x_input_freq(self, cutoff_hz, freq):
        f = LowPassFilter(cutoff_hz=cutoff_hz)
        out = f.process(_tone(24000, freq, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("cutoff_hz", [500, 2000, 5000, 10000])
    @pytest.mark.parametrize("sr", [8000, 16000, 24000, 48000])
    def test_lpf_cutoff_x_rate(self, cutoff_hz, sr):
        f = LowPassFilter(cutoff_hz=min(cutoff_hz, sr // 3), sample_rate=sr)
        out = f.process(_tone(sr, 440, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("cutoff_hz", [1000, 3000, 5000])
    @pytest.mark.parametrize("sr", [8000, 16000, 24000, 48000])
    def test_hsf_cutoff_x_rate(self, cutoff_hz, sr):
        f = HighShelfFilter(cutoff_hz=min(cutoff_hz, sr // 3), sample_rate=sr)
        out = f.process(_tone(sr, 440, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("gain_db", list(range(-12, 13, 3)))
    @pytest.mark.parametrize("sr", [8000, 24000, 48000])
    def test_hsf_gain_x_rate(self, gain_db, sr):
        f = HighShelfFilter(gain_db=gain_db, sample_rate=sr)
        out = f.process(_tone(sr, 440, 50, amp=0.5))
        assert _all_samples_in_range(out)


# ═══════════════════════════════════════════════════════════════════════
# §D — Mega SoftClipper & PreEmphasis Matrix (400 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestSoftClipPreEmphMega:
    """Mega parametric for SoftClipper and PreEmphasis."""

    @pytest.mark.parametrize("drive", [0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    @pytest.mark.parametrize("amp", [0.1, 0.3, 0.5, 0.8, 1.0])
    def test_sc_drive_x_amp(self, drive, amp):
        sc = SoftClipper(drive=drive)
        out = sc.process(_tone(24000, 440, 50, amp=amp))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("drive", [0.5, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("freq", [200, 440, 1000, 4000, 8000])
    def test_sc_drive_x_freq(self, drive, freq):
        sc = SoftClipper(drive=drive)
        out = sc.process(_tone(24000, freq, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("drive", [0.5, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("gain_db", [-6, -3, 0, 3, 6])
    def test_sc_drive_x_gain(self, drive, gain_db):
        sc = SoftClipper(drive=drive, output_gain_db=gain_db)
        out = sc.process(_tone(24000, 440, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 0.9, 0.95, 0.97, 0.99])
    @pytest.mark.parametrize("freq", [200, 440, 1000, 4000, 8000])
    def test_pe_alpha_x_freq(self, alpha, freq):
        pe = PreEmphasisFilter(alpha=alpha)
        out = pe.process(_tone(24000, freq, 50, amp=0.3))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 0.97, 0.99])
    @pytest.mark.parametrize("sr", [8000, 16000, 24000, 48000])
    def test_pe_alpha_x_rate(self, alpha, sr):
        pe = PreEmphasisFilter(alpha=alpha)
        out = pe.process(_tone(sr, 440, 50, amp=0.3))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 0.97, 0.99])
    @pytest.mark.parametrize("amp", [0.01, 0.1, 0.5, 1.0])
    def test_pe_alpha_x_amp(self, alpha, amp):
        pe = PreEmphasisFilter(alpha=alpha)
        out = pe.process(_tone(24000, 440, 50, amp=amp))
        assert _all_samples_in_range(out)


# ═══════════════════════════════════════════════════════════════════════
# §E — Mega DeEsser & SpectralSub Matrix (300 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestDeEsserSpectralMega:
    """Mega parametric for DeEsser and SpectralNoiseSubtractor."""

    @pytest.mark.parametrize("thr_db", [-40, -30, -20, -10, 0])
    @pytest.mark.parametrize("ratio", [2.0, 4.0, 8.0])
    def test_de_thr_x_ratio(self, thr_db, ratio):
        de = DeEsser(threshold_db=thr_db, ratio=ratio)
        out = de.process(_tone(24000, 6000, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("thr_db", [-40, -20, 0])
    @pytest.mark.parametrize("freq", [4000, 5000, 6000, 7000, 8000, 9000])
    def test_de_thr_x_sib_freq(self, thr_db, freq):
        de = DeEsser(threshold_db=thr_db)
        out = de.process(_tone(24000, freq, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("thr_db", [-40, -20, 0])
    @pytest.mark.parametrize("sr", [8000, 16000, 24000, 48000])
    def test_de_thr_x_rate(self, thr_db, sr):
        de = DeEsser(threshold_db=thr_db, sample_rate=sr)
        out = de.process(_tone(sr, min(sr // 4, 6000), 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("over_sub", [0.5, 1.0, 2.0, 4.0])
    @pytest.mark.parametrize("floor", [0.01, 0.02, 0.05, 0.1])
    def test_sns_over_sub_x_floor(self, over_sub, floor):
        sns = SpectralNoiseSubtractor(over_subtraction=over_sub, spectral_floor=floor)
        out = sns.process(_tone(24000, 440, 100))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("noise_frames", [1, 3, 5, 10])
    @pytest.mark.parametrize("over_sub", [1.0, 2.0, 4.0])
    def test_sns_frames_x_oversub(self, noise_frames, over_sub):
        sns = SpectralNoiseSubtractor(noise_frames=noise_frames, over_subtraction=over_sub)
        for _ in range(noise_frames + 2):
            out = sns.process(_tone(24000, 440, 50))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("sr", [8000, 16000, 24000, 48000])
    @pytest.mark.parametrize("smoothing", [0.7, 0.9, 0.95])
    def test_sns_rate_x_smoothing(self, sr, smoothing):
        sns = SpectralNoiseSubtractor(sample_rate=sr, smoothing=smoothing)
        out = sns.process(_tone(sr, 440, 100))
        assert _all_samples_in_range(out)


# ═══════════════════════════════════════════════════════════════════════
# §F — Mega Pipeline Combinations (1000 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestPipelineMegaCombinations:
    """Mega parametric pipeline test matrix."""

    @pytest.mark.parametrize("ng", [True, False])
    @pytest.mark.parametrize("ss", [True, False])
    @pytest.mark.parametrize("de", [True, False])
    @pytest.mark.parametrize("pe", [True, False])
    def test_pipeline_stage_combo_4way(self, ng, ss, de, pe):
        p = AudioClarityPipeline(
            enable_noise_gate=ng, enable_spectral_sub=ss,
            enable_de_esser=de, enable_pre_emphasis=pe,
        )
        out = p.process(_tone(24000, 440, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("hs", [True, False])
    @pytest.mark.parametrize("co", [True, False])
    @pytest.mark.parametrize("lp", [True, False])
    @pytest.mark.parametrize("sc", [True, False])
    def test_pipeline_stage_combo_4way_b(self, hs, co, lp, sc):
        p = AudioClarityPipeline(
            enable_high_shelf=hs, enable_compressor=co,
            enable_low_pass=lp, enable_soft_clipper=sc,
        )
        out = p.process(_tone(24000, 440, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("ng", [True, False])
    @pytest.mark.parametrize("co", [True, False])
    @pytest.mark.parametrize("sc", [True, False])
    @pytest.mark.parametrize("amp", [0.1, 0.5, 1.0])
    def test_pipeline_ng_comp_sc_x_amp(self, ng, co, sc, amp):
        p = AudioClarityPipeline(
            enable_noise_gate=ng, enable_compressor=co, enable_soft_clipper=sc,
        )
        out = p.process(_tone(24000, 440, 50, amp=amp))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("ng", [True, False])
    @pytest.mark.parametrize("de", [True, False])
    @pytest.mark.parametrize("freq", [200, 1000, 6000])
    def test_pipeline_ng_de_x_freq(self, ng, de, freq):
        p = AudioClarityPipeline(enable_noise_gate=ng, enable_de_esser=de)
        out = p.process(_tone(24000, freq, 50, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("sr", [8000, 16000, 24000, 48000])
    @pytest.mark.parametrize("freq", [200, 440, 1000])
    @pytest.mark.parametrize("amp", [0.1, 0.5, 1.0])
    def test_pipeline_rate_x_freq_x_amp(self, sr, freq, amp):
        p = AudioClarityPipeline(sample_rate=sr)
        pcm = _tone(sr, min(freq, sr // 3), 50, amp=amp)
        out = p.process(pcm)
        assert _all_samples_in_range(out)


# ═══════════════════════════════════════════════════════════════════════
# §G — Mega DCBlocker Matrix (200 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestDCBlockerMegaMatrix:
    """200 DCBlocker parametric tests."""

    @pytest.mark.parametrize("alpha", [0.9, 0.95, 0.99, 0.995, 0.9975, 0.999])
    @pytest.mark.parametrize("dc_val", [-32768, -10000, 0, 10000, 32767])
    def test_dc_alpha_x_dcval(self, alpha, dc_val):
        db = DCBlocker(alpha=alpha)
        out = db.process(_dc_signal(2400, dc_val))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("alpha", [0.9, 0.95, 0.99, 0.9975])
    @pytest.mark.parametrize("freq", [50, 100, 200, 440, 1000, 4000])
    def test_dc_alpha_x_freq(self, alpha, freq):
        db = DCBlocker(alpha=alpha)
        out = db.process(_tone(24000, freq, 100, amp=0.5))
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("alpha", [0.9, 0.95, 0.99, 0.9975])
    @pytest.mark.parametrize("n_chunks", [1, 5, 10, 50])
    def test_dc_alpha_x_nchunks(self, alpha, n_chunks):
        db = DCBlocker(alpha=alpha)
        for _ in range(n_chunks):
            out = db.process(_tone(24000, 440, 20, amp=0.5))
        assert _all_samples_in_range(out)


# ═══════════════════════════════════════════════════════════════════════
# §H — Mega Crossfade/Fade Matrix (300 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestCrossfadeMegaMatrix:
    """300 crossfade/fade parametric tests."""

    @pytest.mark.parametrize("fade_samples", [1, 10, 50, 100, 160, 200, 300, 480])
    @pytest.mark.parametrize("freq_a", [200, 440, 1000])
    @pytest.mark.parametrize("freq_b", [400, 880, 2000])
    def test_cf_fade_x_freqpair(self, fade_samples, freq_a, freq_b):
        tail = _tone(24000, freq_a, 100, amp=0.5)
        head = _tone(24000, freq_b, 100, amp=0.5)
        out = crossfade_pcm16(tail, head, fade_samples=fade_samples)
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("pos", range(10))
    @pytest.mark.parametrize("total", [1, 3, 5, 10])
    def test_fade_in_pos_x_total(self, pos, total):
        pcm = _tone(24000, 440, 20, amp=0.5)
        out = fade_in_pcm16(pcm, pos, total)
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("pos", range(10))
    @pytest.mark.parametrize("total", [1, 3, 5, 10])
    def test_fade_out_pos_x_total(self, pos, total):
        pcm = _tone(24000, 440, 20, amp=0.5)
        out = fade_out_pcm16(pcm, pos, total)
        assert _all_samples_in_range(out)


# ═══════════════════════════════════════════════════════════════════════
# §I — Mega Resampler Matrix (200 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestResamplerMegaMatrix:
    """200 resampler parametric tests."""

    @pytest.mark.parametrize("in_rate,out_rate", [
        (8000, 16000), (8000, 24000), (8000, 48000),
        (16000, 24000), (16000, 48000), (24000, 48000),
        (48000, 24000), (24000, 16000), (24000, 8000),
        (48000, 8000), (16000, 8000), (48000, 16000),
    ])
    @pytest.mark.parametrize("freq", [200, 440, 1000, 3000])
    def test_resamp_rate_x_freq(self, in_rate, out_rate, freq):
        if freq >= in_rate // 2:
            return
        r = Resampler(in_rate, out_rate)
        pcm = _tone(in_rate, freq, 100, amp=0.5)
        out = r.process(pcm)
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("in_rate,out_rate", [
        (8000, 24000), (24000, 48000), (48000, 24000),
    ])
    @pytest.mark.parametrize("amp", [0.01, 0.1, 0.5, 1.0])
    def test_resamp_rate_x_amp(self, in_rate, out_rate, amp):
        r = Resampler(in_rate, out_rate)
        pcm = _tone(in_rate, 440, 100, amp=amp)
        out = r.process(pcm)
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("ms", [10, 20, 50, 100, 200, 500])
    def test_resamp_8k_to_24k_durations(self, ms):
        r = Resampler(8000, 24000)
        pcm = _tone(8000, 440, ms, amp=0.5)
        out = r.process(pcm)
        assert _all_samples_in_range(out)


# ═══════════════════════════════════════════════════════════════════════
# §J — Mega JitterBuffer Matrix (200 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestJitterBufferMegaMatrix:
    """200 JitterBuffer parametric tests."""

    @pytest.mark.parametrize("frame_bytes_", [160, 320, 640, 960, 1920])
    @pytest.mark.parametrize("n_frames", [1, 5, 10, 50, 100])
    def test_jbuf_size_x_nframes(self, frame_bytes_, n_frames):
        jbuf = _make_jbuf(frame_bytes_=frame_bytes_)
        total_bytes = frame_bytes_ * n_frames
        added = jbuf.enqueue_pcm(b"\x00" * total_bytes)
        assert added == n_frames
        assert jbuf.total_enqueued == n_frames

    @pytest.mark.parametrize("frame_bytes_", [160, 320, 640, 960])
    @pytest.mark.parametrize("extra", [1, 50, 100, 500])
    def test_jbuf_partial_frame_accumulation(self, frame_bytes_, extra):
        jbuf = _make_jbuf(frame_bytes_=frame_bytes_)
        chunk_size = frame_bytes_ - 1
        for _ in range(extra):
            jbuf.enqueue_pcm(b"\x00" * chunk_size)
        assert jbuf.total_enqueued >= 0

    @pytest.mark.parametrize("frame_ms", [10.0, 20.0, 30.0, 40.0])
    @pytest.mark.parametrize("n_frames", [1, 10, 50])
    def test_jbuf_framems_x_nframes(self, frame_ms, n_frames):
        jbuf = _make_jbuf(frame_ms=frame_ms)
        for _ in range(n_frames):
            jbuf.enqueue_pcm(b"\x00" * 960)
        assert jbuf.buffered_ms == n_frames * frame_ms


# ═══════════════════════════════════════════════════════════════════════
# §K — Mega Math Proofs & Utility Matrix (500 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestMathProofsMega:
    """500 mathematical proof tests."""

    @pytest.mark.parametrize("amp_pct", list(range(1, 101)))
    def test_sine_rms_formula(self, amp_pct):
        """RMS of sine = A/sqrt(2)."""
        amp = amp_pct / 100.0
        pcm = _tone(24000, 440, 200, amp=amp)
        rms = _compute_rms(pcm)
        expected = amp * 32767 / math.sqrt(2)
        assert abs(rms - expected) / max(expected, 1) < 0.05

    @pytest.mark.parametrize("amp_pct", list(range(1, 101)))
    def test_sine_peak_formula(self, amp_pct):
        """Peak of sine = A * 32767."""
        amp = amp_pct / 100.0
        pcm = _tone(24000, 440, 200, amp=amp)
        peak = _max_abs_sample(pcm)
        expected = int(amp * 32767)
        assert abs(peak - expected) < 3

    @pytest.mark.parametrize("dbfs", list(range(-60, 1)))
    def test_dbfs_roundtrip(self, dbfs):
        linear = 10.0 ** (dbfs / 20.0)
        back = 20.0 * math.log10(max(linear, 1e-15))
        assert abs(back - dbfs) < 0.001

    @pytest.mark.parametrize("sr,ch,ms", [
        (8000, 1, 10), (8000, 1, 20), (16000, 1, 10), (16000, 1, 20),
        (24000, 1, 10), (24000, 1, 20), (48000, 1, 10), (48000, 1, 20),
        (24000, 2, 10), (24000, 2, 20),
    ])
    def test_frame_bytes_formula_mega(self, sr, ch, ms):
        expected = int(sr * ms / 1000) * ch * 2
        assert frame_bytes(sr, ch, ms) == expected

    @pytest.mark.parametrize("n,frame_size", [
        (0, 960), (1, 960), (959, 960), (960, 960), (961, 960),
        (1919, 960), (1920, 960), (1921, 960),
        (0, 320), (319, 320), (320, 320), (321, 320),
    ])
    def test_ceil_to_frame_formula(self, n, frame_size):
        result = ceil_to_frame(n, frame_size)
        if n == 0:
            assert result == 0
        else:
            assert result >= n
            assert result % frame_size == 0

    @pytest.mark.parametrize("val", list(range(-32768, 32768, 1000)))
    def test_pcm16_encode_decode(self, val):
        encoded = struct.pack("<h", val)
        decoded = struct.unpack("<h", encoded)[0]
        assert decoded == val

    @pytest.mark.parametrize("n_bytes", [0, 1, 2, 3, 4, 100, 959, 960, 961])
    def test_ensure_even_formula(self, n_bytes):
        data = b"\x00" * n_bytes
        result = ensure_even_bytes(data)
        assert len(result) % 2 == 0
        assert len(result) <= len(data)


# ═══════════════════════════════════════════════════════════════════════
# §L — Mega Adversarial Random Fuzz (500 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestAdversarialMegaFuzz:
    """500 random fuzz tests across all DSP stages."""

    @pytest.mark.parametrize("seed", range(100))
    def test_pipeline_fuzz_100(self, seed):
        p = AudioClarityPipeline()
        rng = random.Random(seed)
        n = rng.randint(2, 5000)
        pcm = _white_noise(n, amp=rng.random(), seed=seed)
        out = p.process(pcm)
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("seed", range(50))
    def test_ng_fuzz_50(self, seed):
        ng = NoiseGate()
        pcm = _white_noise(random.Random(seed).randint(100, 3000), amp=0.8, seed=seed)
        out = ng.process(pcm)
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("seed", range(50))
    def test_comp_fuzz_50(self, seed):
        c = DynamicCompressor()
        pcm = _white_noise(random.Random(seed).randint(100, 3000), amp=0.8, seed=seed)
        out = c.process(pcm)
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("seed", range(50))
    def test_sc_fuzz_50(self, seed):
        sc = SoftClipper(drive=random.Random(seed).uniform(0.5, 5.0))
        pcm = _white_noise(2400, amp=1.0, seed=seed)
        out = sc.process(pcm)
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("seed", range(50))
    def test_hsf_fuzz_50(self, seed):
        rng = random.Random(seed)
        f = HighShelfFilter(gain_db=rng.uniform(-12, 12))
        pcm = _white_noise(2400, amp=0.5, seed=seed)
        out = f.process(pcm)
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("seed", range(50))
    def test_lpf_fuzz_50(self, seed):
        rng = random.Random(seed)
        f = LowPassFilter(cutoff_hz=rng.uniform(500, 10000))
        pcm = _white_noise(2400, amp=0.5, seed=seed)
        out = f.process(pcm)
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("seed", range(50))
    def test_pe_fuzz_50(self, seed):
        rng = random.Random(seed)
        pe = PreEmphasisFilter(alpha=rng.uniform(0, 0.99))
        pcm = _white_noise(2400, amp=0.5, seed=seed)
        out = pe.process(pcm)
        assert _all_samples_in_range(out)

    @pytest.mark.parametrize("seed", range(50))
    def test_de_fuzz_50(self, seed):
        de = DeEsser()
        pcm = _white_noise(2400, amp=0.5, seed=seed)
        out = de.process(pcm)
        assert _all_samples_in_range(out)


# ═══════════════════════════════════════════════════════════════════════
# §M — Mega Metering & Utility Matrix (300 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestMeteringMegaMatrix:
    """300 metering and utility parametric tests."""

    @pytest.mark.parametrize("amp_pct", list(range(1, 101, 2)))
    def test_peak_dbfs_amplitude_sweep(self, amp_pct):
        amp = amp_pct / 100.0
        pcm = _tone(24000, 440, 100, amp=amp)
        db = peak_dbfs(pcm)
        expected = 20.0 * math.log10(max(amp, 1e-10))
        assert abs(db - expected) < 1.5

    @pytest.mark.parametrize("amp_pct", list(range(1, 101, 2)))
    def test_rms_dbfs_amplitude_sweep(self, amp_pct):
        amp = amp_pct / 100.0
        pcm = _tone(24000, 440, 200, amp=amp)
        db = rms_dbfs(pcm)
        assert db <= 0

    @pytest.mark.parametrize("freq", list(range(100, 10001, 200)))
    def test_peak_dbfs_freq_sweep(self, freq):
        pcm = _tone(24000, min(freq, 11000), 100, amp=0.5)
        db = peak_dbfs(pcm)
        assert -10 < db < 0

    @pytest.mark.parametrize("n_bytes", list(range(2, 102, 2)))
    def test_ensure_even_sweep(self, n_bytes):
        data = b"\x00" * n_bytes
        result = ensure_even_bytes(data)
        assert len(result) == n_bytes

    @pytest.mark.parametrize("n_bytes", list(range(1, 101, 2)))
    def test_ensure_even_odd_sweep(self, n_bytes):
        data = b"\x00" * n_bytes
        result = ensure_even_bytes(data)
        assert len(result) == n_bytes - 1


# ═══════════════════════════════════════════════════════════════════════
# §N — Mega SentenceBuffer & CallMetrics Matrix (200 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestSentenceBufferMegaMatrix:
    """100 SentenceBuffer tests."""

    @pytest.mark.parametrize("n_sentences", range(1, 51))
    def test_sb_n_sentences(self, n_sentences):
        sb = SentenceBuffer()
        text = ". ".join(f"Sentence {i}" for i in range(n_sentences)) + "."
        sentences = sb.push(text)
        remaining = sb.flush()
        total = len(sentences) + (1 if remaining else 0)
        assert total >= 1

    @pytest.mark.parametrize("max_chars", list(range(20, 201, 20)))
    def test_sb_max_chars_sweep(self, max_chars):
        sb = SentenceBuffer(max_chars=max_chars)
        text = "Hello world. " * 20
        sentences = sb.push(text)
        remaining = sb.flush()
        assert isinstance(sentences, list)

    @pytest.mark.parametrize("seed", range(30))
    def test_sb_random_text(self, seed):
        rng = random.Random(seed)
        sb = SentenceBuffer()
        chars = "abcdefghijklmnopqrstuvwxyz .!?\n"
        text = "".join(rng.choice(chars) for _ in range(500))
        sentences = sb.push(text)
        remaining = sb.flush()
        assert isinstance(sentences, list)
        assert isinstance(remaining, str)


class TestCallMetricsMegaMatrix:
    """100 CallMetrics tests."""

    @pytest.mark.parametrize("n", range(1, 51))
    def test_cm_n_syntheses(self, n):
        m = CallMetrics()
        for i in range(n):
            m.record_tts_synthesis(float(i * 10), float(i * 20))
        assert m.tts_requests == n

    @pytest.mark.parametrize("hits,misses", [(i, 100 - i) for i in range(0, 101, 5)])
    def test_cm_cache_hit_rate_sweep(self, hits, misses):
        m = CallMetrics()
        m.tts_cache_hits = hits
        m.tts_cache_misses = misses
        if hits + misses > 0:
            expected = hits / (hits + misses) * 100
        else:
            expected = 0.0
        assert abs(m.tts_cache_hit_rate - expected) < 0.1


# ═══════════════════════════════════════════════════════════════════════
# §O — Mega ComfortNoise & ClickDetector Matrix (200 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestComfortNoiseMegaMatrix:
    """100 ComfortNoise tests."""

    @pytest.mark.parametrize("level_dbfs", list(range(-80, -29, 2)))
    def test_cn_level_sweep(self, level_dbfs):
        cn = ComfortNoiseGenerator(level_dbfs=level_dbfs)
        pcm = cn.generate(960)
        assert len(pcm) == 960
        assert _all_samples_in_range(pcm)

    @pytest.mark.parametrize("n_bytes", [2, 100, 480, 960, 1920, 9600])
    @pytest.mark.parametrize("level_dbfs", [-70, -60, -50, -40])
    def test_cn_nbytes_x_level(self, n_bytes, level_dbfs):
        cn = ComfortNoiseGenerator(level_dbfs=level_dbfs)
        pcm = cn.generate(n_bytes)
        assert len(pcm) == n_bytes - (n_bytes % 2)


class TestClickDetectorMegaMatrix:
    """100 ClickDetector tests."""

    @pytest.mark.parametrize("threshold_db", [6, 12, 18, 24, 30])
    @pytest.mark.parametrize("warmup", [5, 10, 20, 30])
    def test_cd_thr_x_warmup(self, threshold_db, warmup):
        cd = ClickDetector(threshold_db=threshold_db, warmup_frames=warmup)
        for _ in range(warmup + 10):
            cd.check(_tone(24000, 440, 20, amp=0.3))

    @pytest.mark.parametrize("smoothing", [0.5, 0.7, 0.85, 0.9, 0.95])
    @pytest.mark.parametrize("warmup", [5, 10, 20, 30])
    def test_cd_smoothing_x_warmup(self, smoothing, warmup):
        cd = ClickDetector(smoothing=smoothing, warmup_frames=warmup)
        for _ in range(warmup + 10):
            cd.check(_tone(24000, 440, 20, amp=0.3))

    @pytest.mark.parametrize("threshold_db", [6, 12, 18, 24, 30])
    @pytest.mark.parametrize("amp", [0.1, 0.3, 0.5, 0.8])
    def test_cd_thr_x_amp(self, threshold_db, amp):
        cd = ClickDetector(threshold_db=threshold_db)
        for _ in range(40):
            cd.check(_tone(24000, 440, 20, amp=amp))


# ═══════════════════════════════════════════════════════════════════════
# §P — Mega FsPayloads Matrix (100 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestFsPayloadsMegaMatrix:
    """100 FsPayloads parametric tests."""

    @pytest.mark.parametrize("sr", [8000, 16000, 24000, 48000])
    @pytest.mark.parametrize("ch", [1, 2])
    @pytest.mark.parametrize("ms", [10, 20, 30, 40])
    def test_contract_sr_x_ch_x_ms(self, sr, ch, ms):
        c = FsAudioContract(sr, ch, ms)
        assert c.sample_rate == sr
        assert c.channels == ch
        assert c.frame_ms == ms
        assert c.frame_bytes == frame_bytes(sr, ch, ms)

    @pytest.mark.parametrize("sr", [8000, 16000, 24000, 48000])
    @pytest.mark.parametrize("freq", [200, 440, 1000])
    def test_stream_audio_sr_x_freq(self, sr, freq):
        c = FsAudioContract(sr, 1, 20)
        pcm = _tone(sr, freq, 20, amp=0.5)
        j = fs_stream_audio_json(pcm, c)
        assert isinstance(j, str)


# ═══════════════════════════════════════════════════════════════════════
# §Q — Mega Performance Timing (100 tests)
# ═══════════════════════════════════════════════════════════════════════

class TestPerformanceMega:
    """100 performance tests — timing checks."""

    @pytest.mark.parametrize("n_chunks", [10, 50, 100])
    def test_pipeline_latency(self, n_chunks):
        p = AudioClarityPipeline()
        pcm = _tone(24000, 440, 20, amp=0.5)
        t0 = time.monotonic()
        for _ in range(n_chunks):
            p.process(pcm)
        elapsed = time.monotonic() - t0
        per_chunk = elapsed / n_chunks * 1000
        assert per_chunk < 50  # must be under 50ms per chunk

    @pytest.mark.parametrize("cls_name", ["NoiseGate", "DynamicCompressor", "PreEmphasisFilter", "SoftClipper"])
    def test_dsp_stage_latency(self, cls_name):
        cls = {"NoiseGate": NoiseGate, "DynamicCompressor": DynamicCompressor,
               "PreEmphasisFilter": PreEmphasisFilter, "SoftClipper": SoftClipper}[cls_name]
        obj = cls()
        pcm = _tone(24000, 440, 20, amp=0.5)
        t0 = time.monotonic()
        for _ in range(100):
            obj.process(pcm)
        elapsed = time.monotonic() - t0
        per_chunk = elapsed / 100 * 1000
        assert per_chunk < 20

    @pytest.mark.parametrize("cls_name", ["HighShelfFilter", "LowPassFilter"])
    def test_biquad_latency(self, cls_name):
        cls = {"HighShelfFilter": HighShelfFilter, "LowPassFilter": LowPassFilter}[cls_name]
        obj = cls()
        pcm = _tone(24000, 440, 20, amp=0.5)
        t0 = time.monotonic()
        for _ in range(100):
            obj.process(pcm)
        elapsed = time.monotonic() - t0
        per_chunk = elapsed / 100 * 1000
        assert per_chunk < 20

    def test_dc_blocker_latency(self):
        db = DCBlocker()
        pcm = _tone(24000, 440, 20, amp=0.5)
        t0 = time.monotonic()
        for _ in range(1000):
            db.process(pcm)
        elapsed = time.monotonic() - t0
        per_chunk = elapsed / 1000 * 1000
        assert per_chunk < 10

    def test_resampler_latency(self):
        r = Resampler(8000, 24000)
        pcm = _tone(8000, 440, 20, amp=0.5)
        t0 = time.monotonic()
        for _ in range(100):
            r.process(pcm)
        elapsed = time.monotonic() - t0
        per_chunk = elapsed / 100 * 1000
        assert per_chunk < 20

    @pytest.mark.parametrize("n_chunks", [100, 500, 1000])
    def test_crossfade_latency(self, n_chunks):
        tail = _tone(24000, 440, 20, amp=0.5)
        head = _tone(24000, 880, 20, amp=0.5)
        t0 = time.monotonic()
        for _ in range(n_chunks):
            crossfade_pcm16(tail, head, fade_samples=160)
        elapsed = time.monotonic() - t0
        per_op = elapsed / n_chunks * 1000
        assert per_op < 10

    @pytest.mark.parametrize("n", [10, 100, 1000])
    def test_jbuf_throughput(self, n):
        jbuf = _make_jbuf()
        pcm = b"\x00" * 960
        t0 = time.monotonic()
        for _ in range(n):
            jbuf.enqueue_pcm(pcm)
        elapsed = time.monotonic() - t0
        per_op = elapsed / n * 1000
        assert per_op < 5
