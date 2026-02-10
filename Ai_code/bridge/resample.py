"""
High-quality PCM16 resampler with automatic backend selection.

Fallback chain (best → worst):
  1. soxr   — sinc-based, broadcast quality (pip install soxr numpy)
  2. samplerate — libsamplerate bindings (pip install samplerate numpy)
  3. audioop — linear interpolation (built-in, lowest quality)

app.py imports:
  from .resample import Resampler, get_backend as get_resample_backend
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

from .audio import ensure_even_bytes

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
#  Detect available backends
# ─────────────────────────────────────────────────────────────────
_HAS_NUMPY = False
try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:
    pass

_HAS_SOXR = False
try:
    import soxr as _soxr
    _HAS_SOXR = True
except ImportError:
    pass

_HAS_SAMPLERATE = False
try:
    import samplerate as _samplerate
    _HAS_SAMPLERATE = True
except ImportError:
    pass

_HAS_AUDIOOP = False
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import audioop as _audioop
    _HAS_AUDIOOP = True
except ImportError:
    try:
        import audioop_lts as _audioop  # type: ignore
        _HAS_AUDIOOP = True
    except ImportError:
        pass


def get_backend() -> str:
    """Return the name of the resampler backend that will be used."""
    if _HAS_SOXR and _HAS_NUMPY:
        return "soxr"
    if _HAS_SAMPLERATE and _HAS_NUMPY:
        return "samplerate"
    if _HAS_AUDIOOP:
        return "audioop"
    return "none"


# ─────────────────────────────────────────────────────────────────
#  Unified Resampler class
# ─────────────────────────────────────────────────────────────────
class Resampler:
    """Stateful PCM16 resampler.

    Usage:
        rs = Resampler(8000, 24000, channels=1)
        out_pcm = rs.process(in_pcm)   # bytes in, bytes out

    The backend is automatically selected (soxr > samplerate > audioop).
    If input rate == output rate, data passes through unchanged.
    """

    def __init__(self, in_rate: int, out_rate: int, *, channels: int = 1) -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.channels = channels
        self._backend = get_backend()
        self._state: object = None  # backend-specific state

        if in_rate == out_rate:
            self._backend = "passthrough"
            logger.info("Resampler: passthrough (same rate %d Hz)", in_rate)
            return

        if self._backend == "soxr":
            # soxr.ResampleStream for stateful streaming
            try:
                self._state = _soxr.ResampleStream(
                    in_rate, out_rate,
                    num_channels=channels,
                    dtype="int16",
                )
                logger.info(
                    "Resampler: soxr %d->%d Hz (sinc/broadcast quality)",
                    in_rate, out_rate,
                )
                return
            except Exception as exc:
                logger.warning("soxr.ResampleStream failed: %s — trying block mode", exc)
                # Fall through to block mode
                self._state = None

        if self._backend == "soxr" and self._state is None:
            # Block mode fallback (no state, but still high quality)
            self._backend = "soxr_block"
            logger.info(
                "Resampler: soxr block mode %d->%d Hz",
                in_rate, out_rate,
            )
            return

        if self._backend == "samplerate":
            try:
                self._state = _samplerate.Resampler("sinc_medium", channels=channels)
                logger.info(
                    "Resampler: samplerate/libsamplerate %d->%d Hz",
                    in_rate, out_rate,
                )
                return
            except Exception as exc:
                logger.warning("samplerate init failed: %s — falling back to audioop", exc)
                self._backend = "audioop"

        if self._backend == "audioop":
            self._state = None  # audioop.ratecv state
            logger.info(
                "Resampler: audioop (linear interp) %d->%d Hz — "
                "install soxr+numpy for better quality",
                in_rate, out_rate,
            )
            return

        logger.error(
            "No resampler available for %d->%d Hz — audio will pass through WRONG RATE",
            in_rate, out_rate,
        )
        self._backend = "passthrough"

    def process(self, pcm: bytes) -> bytes:
        """Resample PCM16 bytes. Returns PCM16 bytes at the output rate."""
        if not pcm:
            return b""

        pcm = ensure_even_bytes(pcm)

        if self._backend == "passthrough":
            return pcm

        if self._backend == "soxr":
            return self._process_soxr_stream(pcm)

        if self._backend == "soxr_block":
            return self._process_soxr_block(pcm)

        if self._backend == "samplerate":
            return self._process_samplerate(pcm)

        if self._backend == "audioop":
            return self._process_audioop(pcm)

        return pcm

    # ── soxr streaming (stateful, best quality) ──
    def _process_soxr_stream(self, pcm: bytes) -> bytes:
        arr = _np.frombuffer(pcm, dtype=_np.int16)
        if self.channels > 1:
            arr = arr.reshape(-1, self.channels)
        out = self._state.resample_chunk(arr)  # type: ignore
        return out.astype(_np.int16).tobytes()

    # ── soxr block mode (stateless, still high quality) ──
    def _process_soxr_block(self, pcm: bytes) -> bytes:
        arr = _np.frombuffer(pcm, dtype=_np.int16)
        if self.channels > 1:
            arr = arr.reshape(-1, self.channels)
        out = _soxr.resample(arr, self.in_rate, self.out_rate)
        return out.astype(_np.int16).tobytes()

    # ── samplerate/libsamplerate (stateful, good quality) ──
    def _process_samplerate(self, pcm: bytes) -> bytes:
        arr = _np.frombuffer(pcm, dtype=_np.int16).astype(_np.float32) / 32768.0
        if self.channels > 1:
            arr = arr.reshape(-1, self.channels)
        ratio = self.out_rate / self.in_rate
        out = self._state.process(arr, ratio, end_of_input=False)  # type: ignore
        out_i16 = _np.clip(out * 32768.0, -32768, 32767).astype(_np.int16)
        return out_i16.tobytes()

    # ── audioop (stateful, lowest quality — linear interpolation) ──
    def _process_audioop(self, pcm: bytes) -> bytes:
        converted, self._state = _audioop.ratecv(
            pcm, 2, self.channels,
            self.in_rate, self.out_rate,
            self._state,
        )
        return ensure_even_bytes(converted)


def guess_pcm16_duration_ms(n_bytes: int, sample_rate: int, channels: int) -> float:
    """Utility for logging/tests."""
    if sample_rate <= 0 or channels <= 0:
        return 0.0
    samples = n_bytes / (2 * channels)
    return (samples / sample_rate) * 1000.0
