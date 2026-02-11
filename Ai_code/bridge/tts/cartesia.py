"""
Cartesia Sonic streaming TTS backend.

Cartesia provides ultra-low-latency streaming TTS via WebSocket.
First-chunk latency is typically 80-150ms — the fastest managed service.

Voice cloning:
  - Upload audio sample via API → get voice_id
  - Or use their voice library

Required:
  pip install aiohttp websockets

Environment:
  CARTESIA_API_KEY=your-key
  TTS_VOICE_ID=your-voice-id
  TTS_MODEL=sonic-2          # or sonic-english, sonic-multilingual
"""
from __future__ import annotations

import base64
import json
import logging
import ssl
from typing import AsyncIterator, Optional

import aiohttp

from .base import TTSChunk, TTSEngine

logger = logging.getLogger(__name__)


def _make_ssl_context() -> ssl.SSLContext:
    """Create SSL context that works on macOS (uses certifi CA bundle)."""
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(cafile=certifi.where())
    except ImportError:
        pass
    return ctx

_SAMPLE_RATE = 24000
_CHANNELS = 1


class CartesiaTTS(TTSEngine):
    """Streaming TTS via Cartesia HTTP streaming API.

    Uses the /tts/bytes endpoint with streaming output for
    low-latency PCM generation.

    Cartesia's Sonic model is optimized for real-time applications
    and typically achieves <150ms first-chunk latency.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "sonic-2",
        sample_rate: int = _SAMPLE_RATE,
        language: str = "en",
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._sample_rate = sample_rate
        self._language = language
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return f"cartesia/{self._model}"

    @property
    def output_sample_rate(self) -> int:
        return self._sample_rate

    @property
    def output_channels(self) -> int:
        return _CHANNELS

    async def warm_up(self) -> None:
        if self._session is None or self._session.closed:
            ssl_ctx = _make_ssl_context()
            conn = aiohttp.TCPConnector(ssl=ssl_ctx)
            self._session = aiohttp.ClientSession(
                connector=conn,
                headers={
                    "X-API-Key": self._api_key,
                    "Cartesia-Version": "2024-06-10",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30, connect=5, sock_read=10),
            )
        logger.info("Cartesia TTS warmed up: model=%s voice=%s", self._model, self._voice_id)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def health_check(self) -> bool:
        try:
            if self._session is None:
                await self.warm_up()
            async with self._session.get(
                "https://api.cartesia.ai/voices",
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
        """Stream-synthesize via Cartesia's bytes streaming endpoint."""
        if not text.strip():
            return

        if self._session is None or self._session.closed:
            await self.warm_up()

        vid = voice_id or self._voice_id
        url = "https://api.cartesia.ai/tts/bytes"

        payload = {
            "transcript": text,
            "model_id": self._model,
            "voice": {
                "mode": "id",
                "id": vid,
            },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self._sample_rate,
            },
            "language": self._language,
        }

        chunk_index = 0
        # 20ms chunk size
        chunk_size = self._sample_rate * _CHANNELS * 2 * 20 // 1000

        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        "Cartesia TTS error %d: %s (text=%.50s)",
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

                # Flush remainder
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
            logger.error("Cartesia stream error: %s (text=%.50s)", e, text)
        except Exception:
            logger.exception("Cartesia unexpected error (text=%.50s)", text)

        logger.debug("Cartesia: synthesized %d chunks for %.40s...", chunk_index, text)
