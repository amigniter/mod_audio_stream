"""
TTS audio cache — memoize common IVR phrases.

In production IVR, ~30% of AI responses are common phrases:
  "Thank you for calling"
  "Please hold"
  "Is there anything else I can help you with?"
  "Let me look that up for you"
  "Have a great day!"

Caching TTS output for these phrases eliminates TTS latency entirely
for cache hits (0ms vs 150-300ms) and reduces GPU/API cost by ~30%.

Cache is per-voice — different voice IDs get separate cache entries.

Storage:
  - In-memory LRU for hot phrases (fastest, per-process)
  - Optional Redis for shared cache across bridge instances
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedAudio:
    """A cached TTS result."""
    pcm16: bytes
    sample_rate: int
    channels: int
    text: str
    voice_id: str
    cached_at: float       # monotonic clock (for in-process TTL)
    cached_at_wall: float  # wall clock (for serialization / Redis)
    synthesis_ms: float    # How long it took to generate (for metrics)

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    stores: int = 0
    total_hit_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100.0) if total > 0 else 0.0


class TTSCache:
    """In-memory LRU cache for TTS audio.

    Args:
        max_entries: Maximum number of cached phrases.
        max_bytes: Maximum total PCM bytes in cache.
        ttl_seconds: Cache entry time-to-live (0 = infinite).
        normalize_text: If True, normalize text before cache key
                        (lowercase, strip whitespace, remove trailing punctuation variants).
    """

    def __init__(
        self,
        *,
        max_entries: int = 500,
        max_bytes: int = 50 * 1024 * 1024,  
        ttl_seconds: float = 3600.0,  
        normalize_text: bool = True,
    ) -> None:
        self._cache: OrderedDict[str, CachedAudio] = OrderedDict()
        self._max_entries = max(max_entries, 10)
        self._max_bytes = max(max_bytes, 1024 * 1024)
        self._ttl = ttl_seconds
        self._normalize = normalize_text
        self._total_bytes: int = 0
        self._lock = asyncio.Lock()
        self.stats = CacheStats()

    def _make_key(self, text: str, voice_id: str) -> str:
        """Create cache key from text + voice_id."""
        normalized = text
        if self._normalize:
            normalized = text.lower().strip()
            
            while normalized and normalized[-1] in ".!?,;:":
                normalized = normalized[:-1]
            normalized = normalized.strip()

        raw = f"{voice_id}::{normalized}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    async def get(self, text: str, voice_id: str) -> Optional[CachedAudio]:
        """Look up cached audio. Returns None on miss."""
        key = self._make_key(text, voice_id)
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self.stats.misses += 1
                return None

            if self._ttl > 0 and (time.monotonic() - entry.cached_at) > self._ttl:
                self._remove_entry(key)
                self.stats.misses += 1
                return None

            self._cache.move_to_end(key)
            self.stats.hits += 1
            self.stats.total_hit_bytes += len(entry.pcm16)
            return entry

    async def put(
        self,
        text: str,
        voice_id: str,
        pcm16: bytes,
        sample_rate: int,
        channels: int = 1,
        synthesis_ms: float = 0.0,
    ) -> None:
        """Store TTS result in cache."""
        key = self._make_key(text, voice_id)
        entry = CachedAudio(
            pcm16=pcm16,
            sample_rate=sample_rate,
            channels=channels,
            text=text,
            voice_id=voice_id,
            cached_at=time.monotonic(),
            cached_at_wall=time.time(),
            synthesis_ms=synthesis_ms,
        )
        audio_bytes = len(pcm16)

        async with self._lock:
            
            if key in self._cache:
                self._remove_entry(key)

            while (
                len(self._cache) >= self._max_entries
                or self._total_bytes + audio_bytes > self._max_bytes
            ) and self._cache:
                self._evict_oldest()

            self._cache[key] = entry
            self._total_bytes += audio_bytes
            self.stats.stores += 1

    def _remove_entry(self, key: str) -> None:
        """Remove entry by key (caller holds lock)."""
        if key in self._cache:
            self._total_bytes -= len(self._cache[key].pcm16)
            del self._cache[key]

    def _evict_oldest(self) -> None:
        """Evict the least-recently-used entry (caller holds lock)."""
        if self._cache:
            _, evicted = self._cache.popitem(last=False)
            self._total_bytes -= len(evicted.pcm16)
            self.stats.evictions += 1

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._total_bytes = 0

    async def preload(
        self,
        phrases: list[str],
        voice_id: str,
        tts_engine, 
    ) -> int:
        """Pre-populate cache with common IVR phrases.

        Call at startup to ensure zero-latency for common responses.
        Returns number of phrases cached.
        """
        cached = 0
        for phrase in phrases:
            existing = await self.get(phrase, voice_id)
            if existing is not None:
                continue
            try:
                pcm_parts: list[bytes] = []
                t0 = time.monotonic()
                async for chunk in tts_engine.synthesize_stream(phrase, voice_id=voice_id):
                    pcm_parts.append(chunk.pcm16)
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                full_pcm = b"".join(pcm_parts)
                if full_pcm:
                    await self.put(
                        phrase, voice_id, full_pcm,
                        sample_rate=tts_engine.output_sample_rate,
                        channels=tts_engine.output_channels,
                        synthesis_ms=elapsed_ms,
                    )
                    cached += 1
                    logger.debug("Cache preload: '%s' (%.0fms, %d bytes)", phrase, elapsed_ms, len(full_pcm))
            except Exception:
                logger.warning("Cache preload failed for: '%s'", phrase, exc_info=True)
        logger.info("TTS cache preloaded %d/%d phrases for voice=%s", cached, len(phrases), voice_id)
        return cached



DEFAULT_IVR_PHRASES = [
    "Thank you for calling.",
    "How can I help you today?",
    "Please hold while I look that up.",
    "Is there anything else I can help you with?",
    "Let me transfer you to a specialist.",
    "I understand. Let me help you with that.",
    "Have a great day! Goodbye.",
    "I'm sorry, could you repeat that?",
    "One moment please.",
    "Thank you for your patience.",
]
