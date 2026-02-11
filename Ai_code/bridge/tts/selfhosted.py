"""
Self-hosted TTS backend for XTTS v2, Fish Speech, StyleTTS2, or VITS.

Connects to a local or remote TTS inference server over HTTP.
The server must implement a simple streaming API:

  POST /synthesize
  Body: {"text": "...", "voice_id": "...", "sample_rate": 24000}
  Response: Chunked PCM16 audio stream

This is a generic client — the actual TTS model runs in a separate
GPU inference server (see selfhosted_server.py for the server).

Scaling:
  - Each A10G GPU handles ~50 concurrent 24kHz TTS streams
  - Use Kubernetes with GPU node pools
  - Load balance across multiple GPU pods
  - Model is loaded once per pod, inference is batched

Required:
  pip install aiohttp

Environment:
  TTS_SELFHOSTED_URL=http://tts-service:8080
  TTS_VOICE_ID=your-voice-id
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

import aiohttp

from .base import TTSChunk, TTSEngine

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 24000
_CHANNELS = 1


class SelfHostedTTS(TTSEngine):
    """Streaming TTS client for self-hosted inference server.

    Connects to a TTS inference server (XTTS, Fish Speech, etc.)
    via HTTP with chunked transfer encoding for streaming output.

    The server must return raw PCM16 LE audio in chunked responses.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        voice_id: str = "default",
        sample_rate: int = _SAMPLE_RATE,
        timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._voice_id = voice_id
        self._sample_rate = sample_rate
        self._timeout = timeout_s
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return "selfhosted"

    @property
    def output_sample_rate(self) -> int:
        return self._sample_rate

    @property
    def output_channels(self) -> int:
        return _CHANNELS

    async def warm_up(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout, connect=5),
            )
        try:
            async with self._session.get(
                f"{self._base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    logger.info("Self-hosted TTS server healthy: %s", self._base_url)
                else:
                    logger.warning("Self-hosted TTS health check returned %d", resp.status)
        except Exception as e:
            logger.warning("Self-hosted TTS health check failed: %s", e)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def health_check(self) -> bool:
        try:
            if self._session is None:
                await self.warm_up()
            async with self._session.get(
                f"{self._base_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice_id: str | None = None,
    ) -> AsyncIterator[TTSChunk]:
        """Stream-synthesize via self-hosted TTS server."""
        if not text.strip():
            return

        if self._session is None or self._session.closed:
            await self.warm_up()

        vid = voice_id or self._voice_id
        url = f"{self._base_url}/synthesize"

        payload = {
            "text": text,
            "voice_id": vid,
            "sample_rate": self._sample_rate,
            "channels": _CHANNELS,
            "format": "pcm16",
        }

        chunk_index = 0
        chunk_size = self._sample_rate * _CHANNELS * 2 * 20 // 1000  # 20ms

        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        "Self-hosted TTS error %d: %s (text=%.50s)",
                        resp.status, body[:200], text,
                    )
                    return

                buffer = bytearray()
                async for data in resp.content.iter_any():
                    buffer.extend(data)

                    while len(buffer) >= chunk_size:
                        pcm_chunk = bytes(buffer[:chunk_size])
                        del buffer[:chunk_size]
                        chunk_index += 1
                        yield TTSChunk(
                            pcm16=pcm_chunk,
                            sample_rate=self._sample_rate,
                            channels=_CHANNELS,
                            is_final=False,
                            text_fragment=text if chunk_index == 1 else "",
                        )

                if buffer:
                    if len(buffer) % 2:
                        buffer = buffer[:-1]
                    if buffer:
                        chunk_index += 1
                        yield TTSChunk(
                            pcm16=bytes(buffer),
                            sample_rate=self._sample_rate,
                            channels=_CHANNELS,
                            is_final=True,
                            text_fragment="",
                        )

        except aiohttp.ClientError as e:
            logger.error("Self-hosted TTS stream error: %s (text=%.50s)", e, text)
        except Exception:
            logger.exception("Self-hosted TTS unexpected error (text=%.50s)", text)

        logger.debug("Self-hosted: synthesized %d chunks for %.40s...", chunk_index, text)
