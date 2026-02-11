"""
Abstract TTS engine interface.

All TTS backends (ElevenLabs, Cartesia, self-hosted, OpenAI fallback)
implement this interface. The bridge code is backend-agnostic.

Key design principles:
  1. Streaming — synthesize_stream() yields PCM chunks as they arrive.
  2. Async — all I/O is non-blocking for asyncio compatibility.
  3. Stateless — each call to synthesize_stream is independent.
  4. PCM16 output — all backends output raw PCM16 at a known sample rate.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass(frozen=True)
class TTSChunk:
    """A chunk of synthesized audio from the TTS engine.

    Attributes:
        pcm16: Raw PCM16 little-endian audio bytes.
        sample_rate: Sample rate in Hz (e.g., 24000).
        channels: Number of audio channels (typically 1).
        is_final: True if this is the last chunk for the current synthesis.
        text_fragment: The text that produced this audio (for logging).
    """
    pcm16: bytes
    sample_rate: int
    channels: int = 1
    is_final: bool = False
    text_fragment: str = ""


class TTSEngine(abc.ABC):
    """Abstract streaming TTS engine.

    Subclasses must implement:
      - synthesize_stream(): async generator yielding TTSChunk
      - close(): cleanup resources
      - properties: name, output_sample_rate, output_channels

    The bridge calls synthesize_stream() with a sentence-sized text chunk
    and drains the yielded PCM chunks into the JitterBuffer.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable engine name (for logging)."""
        ...

    @property
    @abc.abstractmethod
    def output_sample_rate(self) -> int:
        """Output PCM sample rate in Hz."""
        ...

    @property
    @abc.abstractmethod
    def output_channels(self) -> int:
        """Output channel count (1=mono, 2=stereo)."""
        ...

    @abc.abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        *,
        voice_id: str | None = None,
    ) -> AsyncIterator[TTSChunk]:
        """Stream-synthesize text into PCM16 audio chunks.

        Args:
            text: The sentence/phrase to synthesize.
            voice_id: Optional per-request voice override.

        Yields:
            TTSChunk objects containing PCM16 audio data.

        The caller expects chunks to arrive as fast as the TTS produces them
        (streaming). For managed APIs this means HTTP chunked-transfer or
        WebSocket streaming. For self-hosted this means GPU inference with
        streaming decode.

        Example:
            async for chunk in engine.synthesize_stream("Hello!"):
                jitter_buffer.enqueue_pcm(chunk.pcm16)
        """
        ...
        # Make this an async generator (yield required in abstract body)
        if False:
            yield  # pragma: no cover

    async def warm_up(self) -> None:
        """Optional: pre-load model / warm connection pool.

        Called once at bridge startup. Override to pre-load TTS models
        or establish persistent connections.
        """

    async def close(self) -> None:
        """Release resources (HTTP sessions, GPU memory, etc.)."""

    async def health_check(self) -> bool:
        """Return True if the engine is healthy and ready to serve."""
        return True
