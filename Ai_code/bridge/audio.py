from __future__ import annotations
import base64
import logging
import warnings

logger = logging.getLogger(__name__)

_audioop = None
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import audioop as _audioop
except ImportError:
    try:
        import audioop_lts as _audioop 
    except ImportError:
        pass


def tomono_pcm16(pcm: bytes) -> bytes:
    """Convert stereo PCM16 to mono. Returns unchanged if audioop unavailable."""
    if _audioop is not None:
        return _audioop.tomono(pcm, 2, 0.5, 0.5)
    logger.warning("audioop unavailable — cannot downmix stereo input")
    return ensure_even_bytes(pcm)


def frame_bytes(sample_rate: int, channels: int, frame_ms: int) -> int:
    if sample_rate <= 0 or channels <= 0 or frame_ms <= 0:
        raise ValueError("Invalid audio parameters")
    samples = int(sample_rate * frame_ms / 1000)
    return samples * channels * 2 


def b64encode_pcm16(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def ceil_to_frame(n: int, frame: int) -> int:
    if frame <= 0:
        return n
    return ((n + frame - 1) // frame) * frame


def ensure_even_bytes(pcm: bytes) -> bytes:
    if len(pcm) % 2:
        return pcm[:-1]
    return pcm


def trim_to_frame_multiple(buf: bytearray, frame: int) -> None:
    """Trim tail so len(buf) becomes a multiple of frame."""
    if frame <= 0:
        return
    rem = len(buf) % frame
    if rem:
        del buf[-rem:]


def drop_oldest_frame_aligned(buf: bytearray, drop_bytes: int, frame: int) -> int:
    """Drop at least drop_bytes from the front, rounded up to full frames."""
    if drop_bytes <= 0 or not buf:
        return 0
    if frame > 0:
        drop = ((drop_bytes + frame - 1) // frame) * frame
    else:
        drop = drop_bytes
    if drop > len(buf):
        drop = len(buf)
    del buf[:drop]
    return drop
