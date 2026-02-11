"""
Custom TTS subsystem for mod_audio_stream bridge.

Provides a pluggable TTS interface so the bridge can use any voice
instead of OpenAI's built-in voices.

Usage:
    from bridge.tts import create_tts_engine
    engine = create_tts_engine(config)

    async for pcm_chunk in engine.synthesize_stream("Hello world"):
        jitter_buffer.enqueue_pcm(pcm_chunk)
"""
from .base import TTSEngine, TTSChunk
from .sentence_buffer import SentenceBuffer
from .cache import TTSCache
from .factory import create_tts_engine

__all__ = [
    "TTSEngine",
    "TTSChunk",
    "SentenceBuffer",
    "TTSCache",
    "create_tts_engine",
]
