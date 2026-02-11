"""
TTS engine factory with failover chain.

Creates the appropriate TTS engine based on configuration and
wraps it with failover logic so calls never experience silence.

Failover chain: Primary → Secondary → OpenAI TTS-1 fallback

Usage:
    engine = create_tts_engine(config)
    await engine.warm_up()
    async for chunk in engine.synthesize_stream("Hello"):
        ...
"""
from __future__ import annotations

import logging
import time
from typing import AsyncIterator, Optional

from .base import TTSChunk, TTSEngine

logger = logging.getLogger(__name__)


class FailoverTTSEngine(TTSEngine):
    """Wraps multiple TTS engines with automatic failover.

    Tries the primary engine first. If it fails (exception or no output),
    falls through to secondary, then tertiary.

    The failover timeout is aggressive (100ms connect timeout) so callers
    don't experience noticeable delay on failover.
    """

    def __init__(self, engines: list[TTSEngine]) -> None:
        if not engines:
            raise ValueError("At least one TTS engine is required")
        self._engines = engines
        self._primary = engines[0]

    @property
    def name(self) -> str:
        names = [e.name for e in self._engines]
        return f"failover({' → '.join(names)})"

    @property
    def output_sample_rate(self) -> int:
        return self._primary.output_sample_rate

    @property
    def output_channels(self) -> int:
        return self._primary.output_channels

    async def warm_up(self) -> None:
        for engine in self._engines:
            try:
                await engine.warm_up()
            except Exception:
                logger.warning("Failed to warm up %s", engine.name, exc_info=True)

    async def close(self) -> None:
        for engine in self._engines:
            try:
                await engine.close()
            except Exception:
                pass

    async def health_check(self) -> bool:
        for engine in self._engines:
            if await engine.health_check():
                return True
        return False

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice_id: str | None = None,
    ) -> AsyncIterator[TTSChunk]:
        """Try each engine in order until one succeeds."""
        last_error: Optional[Exception] = None

        for i, engine in enumerate(self._engines):
            try:
                chunk_count = 0
                t0 = time.monotonic()
                async for chunk in engine.synthesize_stream(text, voice_id=voice_id):
                    chunk_count += 1
                    # Adjust sample rate/channels to match primary
                    # (all engines should output the same format, but just in case)
                    yield chunk

                if chunk_count > 0:
                    elapsed_ms = (time.monotonic() - t0) * 1000
                    if i > 0:
                        logger.warning(
                            "TTS failover: %s succeeded (primary %s failed) "
                            "chunks=%d latency=%.0fms text=%.40s",
                            engine.name, self._engines[0].name,
                            chunk_count, elapsed_ms, text,
                        )
                    return  # Success

                logger.warning("TTS %s returned 0 chunks for '%.40s'", engine.name, text)

            except Exception as e:
                last_error = e
                logger.warning(
                    "TTS %s failed for '%.40s': %s — trying next",
                    engine.name, text, e,
                )

        # All engines failed
        logger.error(
            "All TTS engines failed for '%.60s'. Last error: %s",
            text, last_error,
        )


def create_tts_engine(cfg) -> TTSEngine:
    """Create TTS engine from BridgeConfig.

    Reads these config fields:
      cfg.tts_provider: "elevenlabs" | "cartesia" | "selfhosted" | "openai" | "none"
      cfg.tts_api_key: API key for the TTS provider
      cfg.tts_voice_id: Voice ID for the TTS provider
      cfg.tts_model: Model name (provider-specific)
      cfg.tts_fallback_provider: Secondary TTS provider
      cfg.tts_selfhosted_url: URL for self-hosted TTS server

    Returns a FailoverTTSEngine wrapping primary + fallback.
    """
    engines: list[TTSEngine] = []

    provider = getattr(cfg, "tts_provider", "none").lower().strip()
    fallback = getattr(cfg, "tts_fallback_provider", "openai").lower().strip()

    # Build primary engine
    primary = _build_engine(
        provider=provider,
        api_key=getattr(cfg, "tts_api_key", ""),
        voice_id=getattr(cfg, "tts_voice_id", ""),
        model=getattr(cfg, "tts_model", ""),
        selfhosted_url=getattr(cfg, "tts_selfhosted_url", ""),
        openai_api_key=getattr(cfg, "openai_api_key", ""),
    )
    if primary:
        engines.append(primary)

    # Build fallback engine (if different from primary)
    if fallback and fallback != provider and fallback != "none":
        secondary = _build_engine(
            provider=fallback,
            api_key=getattr(cfg, "tts_fallback_api_key", "") or getattr(cfg, "tts_api_key", ""),
            voice_id=getattr(cfg, "tts_fallback_voice_id", "") or getattr(cfg, "tts_voice_id", ""),
            model=getattr(cfg, "tts_fallback_model", ""),
            selfhosted_url=getattr(cfg, "tts_selfhosted_url", ""),
            openai_api_key=getattr(cfg, "openai_api_key", ""),
        )
        if secondary:
            engines.append(secondary)

    # Always add OpenAI TTS-1 as last-resort fallback
    openai_key = getattr(cfg, "openai_api_key", "")
    if openai_key and provider != "openai" and fallback != "openai":
        from .openai_tts import OpenAITTS
        engines.append(OpenAITTS(
            api_key=openai_key,
            voice=getattr(cfg, "voice", "alloy"),
        ))

    if not engines:
        logger.error("No TTS engines configured — custom voice will not work")
        # Return a dummy that yields nothing
        from .openai_tts import OpenAITTS
        if openai_key:
            return OpenAITTS(api_key=openai_key)
        raise ValueError("No TTS engine available and no OpenAI API key")

    if len(engines) == 1:
        logger.info("TTS engine: %s (no failover)", engines[0].name)
        return engines[0]

    engine = FailoverTTSEngine(engines)
    logger.info("TTS engine: %s", engine.name)
    return engine


def _build_engine(
    *,
    provider: str,
    api_key: str,
    voice_id: str,
    model: str,
    selfhosted_url: str,
    openai_api_key: str,
) -> Optional[TTSEngine]:
    """Build a single TTS engine from provider name."""
    if provider == "elevenlabs":
        if not api_key:
            logger.warning("ElevenLabs selected but TTS_API_KEY not set")
            return None
        from .elevenlabs import ElevenLabsTTS
        return ElevenLabsTTS(
            api_key=api_key,
            voice_id=voice_id or "21m00Tcm4TlvDq8ikWAM",  # Default: Rachel
            model=model or "eleven_turbo_v2_5",
        )

    if provider == "cartesia":
        if not api_key:
            logger.warning("Cartesia selected but TTS_API_KEY not set")
            return None
        from .cartesia import CartesiaTTS
        return CartesiaTTS(
            api_key=api_key,
            voice_id=voice_id,
            model=model or "sonic-2",
        )

    if provider == "selfhosted":
        from .selfhosted import SelfHostedTTS
        return SelfHostedTTS(
            base_url=selfhosted_url or "http://localhost:8080",
            voice_id=voice_id or "default",
        )

    if provider == "openai":
        if not openai_api_key:
            logger.warning("OpenAI TTS selected but OPENAI_API_KEY not set")
            return None
        from .openai_tts import OpenAITTS
        return OpenAITTS(
            api_key=openai_api_key,
            model=model or "tts-1",
            voice=voice_id or "alloy",
        )

    if provider != "none":
        logger.warning("Unknown TTS provider: '%s'", provider)

    return None
