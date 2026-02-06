from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import audioop

from .audio import ensure_even_bytes


@dataclass
class PCM16Resampler:
    """Stateful PCM16 resampler.

    - Only supports 16-bit little-endian PCM.
    - Keeps internal state so successive chunks are continuous.
    - Can optionally downmix stereo->mono.
    """

    src_rate: int
    dst_rate: int
    src_channels: int = 1
    dst_channels: int = 1
    _ratecv_state: Any = None

    def convert(self, pcm: bytes) -> bytes:
        if not pcm:
            return b""

        pcm = ensure_even_bytes(pcm)

        # Channels: best-effort downmix only (common for telephony)
        if self.src_channels != self.dst_channels:
            if self.src_channels == 2 and self.dst_channels == 1:
                try:
                    pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
                except Exception:
                    # If we can't mix, proceed with original bytes.
                    pass
            elif self.src_channels == 1 and self.dst_channels == 2:
                # Rare for this project; implement if needed.
                raise ValueError("Mono->stereo upmix not supported")
            else:
                raise ValueError(f"Unsupported channel conversion {self.src_channels}->{self.dst_channels}")

        # Rate conversion
        if self.src_rate == self.dst_rate:
            return ensure_even_bytes(pcm)

        converted, self._ratecv_state = audioop.ratecv(
            pcm,
            2,  # width
            self.dst_channels,
            self.src_rate,
            self.dst_rate,
            self._ratecv_state,
        )

        return ensure_even_bytes(converted)


def guess_pcm16_duration_ms(n_bytes: int, sample_rate: int, channels: int) -> float:
    """Utility for logging/tests.

    For PCM16: bytes_per_sample = 2.
    """
    if sample_rate <= 0 or channels <= 0:
        return 0.0
    samples = n_bytes / (2 * channels)
    return (samples / sample_rate) * 1000.0
