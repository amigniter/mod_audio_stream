from __future__ import annotations

import asyncio
import hashlib
import logging

from bridge.config import load_config
from bridge.logging_utils import setup_logging
from bridge.app import run_server


logger = logging.getLogger(__name__)


def _mask_secret(s: str, prefix: int = 8, suffix: int = 6) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if len(s) <= prefix + suffix:
        return "*" * len(s)
    return f"{s[:prefix]}...{s[-suffix:]}"


def _sha256_prefix(s: str, n: int = 12) -> str:
    if not s:
        return ""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def _log_config(cfg) -> None:
    api_key = getattr(cfg, "openai_api_key", "")
    logging.getLogger("bridge").info(
        "Config: host=%s port=%s model=%s voice=%s fs_sample_rate=%s fs_out_sample_rate=%s fs_channels=%s fs_frame_ms=%s "
        "force_commit_ms=%s(vad_driven) response_min_interval_ms=%s fs_send_json_audio=%s wss_pem=%s openai_wss_insecure=%s "
        "OPENAI_API_KEY_MASKED=%s OPENAI_API_KEY_SHA256_12=%s "
        "tts_provider=%s tts_voice_id=%s",
        getattr(cfg, "host", ""),
        getattr(cfg, "port", ""),
        getattr(cfg, "model", ""),
        getattr(cfg, "voice", ""),
        getattr(cfg, "fs_sample_rate", ""),
        getattr(cfg, "fs_out_sample_rate", ""),
        getattr(cfg, "fs_channels", ""),
        getattr(cfg, "fs_frame_ms", ""),
        getattr(cfg, "force_commit_ms", ""),
        getattr(cfg, "response_min_interval_ms", ""),
        getattr(cfg, "fs_send_json_audio", ""),
        getattr(cfg, "wss_pem", ""),
        getattr(cfg, "openai_wss_insecure", ""),
        _mask_secret(api_key),
        _sha256_prefix(api_key),
        getattr(cfg, "tts_provider", "none"),
        getattr(cfg, "tts_voice_id", ""),
    )


async def _async_main() -> None:
    """Async entry point: loads config, creates TTS engine if needed, starts server."""
    setup_logging()
    cfg = load_config()
    logging.getLogger("bridge").info("=== mod-audio-stream bridge starting (config dump below, secrets masked) ===")
    _log_config(cfg)

    tts_engine = None
    tts_cache = None

    if cfg.tts_provider != "none":
        from bridge.tts import create_tts_engine, TTSCache

        logger.info("Initializing custom TTS: provider=%s", cfg.tts_provider)
        tts_engine = create_tts_engine(cfg)

        try:
            await tts_engine.warm_up()
            logger.info("TTS engine ready: %s", tts_engine.name)
        except Exception:
            logger.warning("TTS warm-up failed (will retry on first call)", exc_info=True)

        if cfg.tts_cache_enabled:
            tts_cache = TTSCache(
                max_entries=cfg.tts_cache_max_entries,
                ttl_seconds=cfg.tts_cache_ttl_s,
            )

            common_phrases = [
                "Thank you for calling.",
                "How can I help you today?",
                "Is there anything else I can help you with?",
                "Let me look that up for you.",
                "One moment please.",
                "I understand.",
                "Have a great day!",
                "Goodbye.",
            ]
            try:
                cached_count = await tts_cache.preload(
                    common_phrases,
                    voice_id=cfg.tts_voice_id or "",
                    tts_engine=tts_engine,
                )
                logger.info("TTS cache: preloaded %d/%d common phrases", cached_count, len(common_phrases))
            except Exception:
                logger.warning("TTS cache preload failed", exc_info=True)

        logger.info(
            "Custom voice pipeline ready: OpenAI text-only → %s → caller hears YOUR voice",
            tts_engine.name,
        )
    else:
        logger.info("Using OpenAI built-in voice (no custom TTS)")

    await run_server(cfg, tts_engine=tts_engine, tts_cache=tts_cache)


def main() -> None:
    try:
        asyncio.run(_async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
