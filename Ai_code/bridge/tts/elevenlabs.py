"""
ElevenLabs streaming TTS backend.

Uses ElevenLabs v1 text-to-speech streaming API with WebSocket or
chunked HTTP streaming for minimal latency.

ElevenLabs returns audio chunks as they're generated — typically
the first chunk arrives in ~150-250ms.

Voice cloning:
  - Instant Voice Clone: upload 30s-5min of audio → get voice_id
  - Professional Voice Clone: upload 30min+ → fine-tuned model
  - Use the voice_id from your cloned voice in config

Required:
  pip install aiohttp

Environment:
  ELEVENLABS_API_KEY=your-key
  TTS_VOICE_ID=your-cloned-voice-id
  TTS_MODEL=eleven_turbo_v2_5         # fastest streaming model
"""
from __future__ import annotations

import io
import logging
import ssl
import struct
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

# ElevenLabs streaming output settings
# PCM signed 16-bit LE at 24kHz mono — matches our pipeline
_OUTPUT_FORMAT = "pcm_24000"
_SAMPLE_RATE = 24000
_CHANNELS = 1


class ElevenLabsTTS(TTSEngine):
    """Streaming TTS via ElevenLabs HTTP API.

    Uses the /v1/text-to-speech/{voice_id}/stream endpoint with
    chunked transfer encoding for streaming PCM output.

    The latency optimization settings (optimize_streaming_latency)
    trade off quality for speed:
      0 = max quality (default)
      1 = optimized latency
      2 = more optimized
      3 = max optimization (lowest quality)
      4 = max optimization + text normalizer disabled

    For IVR, level 2-3 is recommended — telephony masks quality
    differences above 8kHz anyway.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "eleven_turbo_v2_5",
        latency_optimization: int = 3,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        output_format: str = _OUTPUT_FORMAT,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._latency_opt = latency_optimization
        self._stability = stability
        self._similarity = similarity_boost
        self._style = style
        self._output_format = output_format
        self._session: Optional[aiohttp.ClientSession] = None

        # Parse sample rate from output format
        self._sample_rate = _SAMPLE_RATE
        if "16000" in output_format:
            self._sample_rate = 16000
        elif "22050" in output_format:
            self._sample_rate = 22050
        elif "44100" in output_format:
            self._sample_rate = 44100

    @property
    def name(self) -> str:
        return f"elevenlabs/{self._model}"

    @property
    def output_sample_rate(self) -> int:
        return self._sample_rate

    @property
    def output_channels(self) -> int:
        return _CHANNELS

    async def warm_up(self) -> None:
        """Create persistent HTTP session."""
        if self._session is None or self._session.closed:
            ssl_ctx = _make_ssl_context()
            conn = aiohttp.TCPConnector(ssl=ssl_ctx)
            self._session = aiohttp.ClientSession(
                connector=conn,
                headers={
                    "xi-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    connect=5,
                    sock_read=10,
                ),
            )
        logger.info("ElevenLabs TTS warmed up: model=%s voice=%s", self._model, self._voice_id)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def health_check(self) -> bool:
        try:
            if self._session is None:
                await self.warm_up()
            async with self._session.get(
                "https://api.elevenlabs.io/v1/models",
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
        """Stream-synthesize text via ElevenLabs.

        Yields PCM16 chunks as they arrive from the API.
        First chunk typically arrives in 150-250ms.
        """
        if not text.strip():
            return

        if self._session is None or self._session.closed:
            await self.warm_up()

        vid = voice_id or self._voice_id
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{vid}/stream"
            f"?output_format={self._output_format}"
            f"&optimize_streaming_latency={self._latency_opt}"
        )

        payload = {
            "text": text,
            "model_id": self._model,
            "voice_settings": {
                "stability": self._stability,
                "similarity_boost": self._similarity,
                "style": self._style,
                "use_speaker_boost": True,
            },
        }

        chunk_index = 0
        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        "ElevenLabs TTS error %d: %s (text=%.50s)",
                        resp.status, body[:200], text,
                    )
                    return

                # Stream PCM chunks as they arrive
                buffer = bytearray()
                # Yield in chunks of ~20ms (960 bytes at 24kHz mono 16-bit)
                chunk_size = self._sample_rate * _CHANNELS * 2 * 20 // 1000

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

                # Flush remaining buffer
                if buffer:
                    # Ensure even byte count for PCM16
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
            logger.error("ElevenLabs stream error: %s (text=%.50s)", e, text)
        except Exception:
            logger.exception("ElevenLabs unexpected error (text=%.50s)", text)

        logger.debug(
            "ElevenLabs: synthesized %d chunks for %.40s...",
            chunk_index, text,
        )
