"""
OpenAI TTS-1 streaming backend — universal fallback.

Uses OpenAI's standard TTS API (not Realtime) as a last-resort fallback.
This is NOT the Realtime API voice — it's the separate TTS-1/TTS-1-HD API
that accepts text and returns audio.

Pros: Always available if you have an OpenAI key, decent quality.
Cons: Higher latency (~300-500ms first chunk), limited voice selection,
      no voice cloning.

This serves as the final link in the failover chain:
  Primary (ElevenLabs) → Secondary (Cartesia) → Fallback (OpenAI TTS-1)

Required:
  pip install aiohttp

Environment:
  OPENAI_API_KEY=your-key  (same key as Realtime API)
"""
from __future__ import annotations

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
        pass  # Fall back to system certs
    return ctx

_SAMPLE_RATE = 24000
_CHANNELS = 1


class OpenAITTS(TTSEngine):
    """Streaming TTS via OpenAI TTS-1 API (fallback).

    Uses the /v1/audio/speech endpoint with response_format=pcm
    which returns raw PCM16 at 24kHz.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "tts-1",
        voice: str = "alloy",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._voice = voice
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def name(self) -> str:
        return f"openai/{self._model}"

    @property
    def output_sample_rate(self) -> int:
        return _SAMPLE_RATE

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
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30, connect=5, sock_read=15),
            )
        logger.info("OpenAI TTS fallback warmed up: model=%s voice=%s", self._model, self._voice)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def health_check(self) -> bool:
        try:
            if self._session is None:
                await self.warm_up()
            async with self._session.get(
                "https://api.openai.com/v1/models",
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
        """Stream-synthesize via OpenAI TTS-1.

        Note: OpenAI TTS doesn't support true streaming output for pcm format
        in all cases. We stream the response body as it arrives.
        """
        if not text.strip():
            return

        if self._session is None or self._session.closed:
            await self.warm_up()

        url = "https://api.openai.com/v1/audio/speech"
        payload = {
            "model": self._model,
            "input": text,
            "voice": voice_id or self._voice,
            "response_format": "pcm",  
        }

        chunk_index = 0
        chunk_size = _SAMPLE_RATE * _CHANNELS * 2 * 20 // 1000  # 20ms

        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        "OpenAI TTS error %d: %s (text=%.50s)",
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
                            sample_rate=_SAMPLE_RATE,
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
                            sample_rate=_SAMPLE_RATE,
                            channels=_CHANNELS,
                            is_final=True,
                            text_fragment="",
                        )

        except aiohttp.ClientError as e:
            logger.error("OpenAI TTS stream error: %s (text=%.50s)", e, text)
        except Exception:
            logger.exception("OpenAI TTS unexpected error (text=%.50s)", text)

        logger.debug("OpenAI TTS: synthesized %d chunks for %.40s...", chunk_index, text)
