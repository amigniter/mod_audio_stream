"""
Sentence-boundary text buffer for streaming LLM → TTS pipeline.

LLM tokens stream in one-by-one. Sending each token to TTS individually
would produce choppy, unnatural speech. Waiting for the full response
would add seconds of latency.

The SentenceBuffer accumulates tokens and flushes at natural sentence
boundaries (. ? ! ; — and after ~80 characters with no boundary).
This gives the TTS enough context for natural prosody while keeping
first-chunk latency minimal.

Latency profile:
  - Average sentence: 8-15 words ≈ 50-100 chars
  - TTS needs ~20 chars minimum for natural intonation
  - Flush at sentence end OR after max_chars (whichever comes first)

Usage:
    sb = SentenceBuffer(max_chars=80)
    for token in llm_stream:
        sentences = sb.push(token)
        for sentence in sentences:
            await tts.synthesize_stream(sentence)
    final = sb.flush()
    if final:
        await tts.synthesize_stream(final)
"""
from __future__ import annotations

import re
from typing import List


_SENTENCE_END_RE = re.compile(
    r'(?<=[.!?;])'    
    r'(?:\s|$)'        
)

_ABBREVIATIONS = frozenset({
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "vs.", "etc.", "inc.", "ltd.", "dept.", "est.",
    "approx.", "assn.", "ave.", "blvd.", "st.",
    "u.s.", "u.k.", "e.g.", "i.e.", "a.m.", "p.m.",
})


class SentenceBuffer:
    """Accumulates LLM text tokens and flushes at sentence boundaries.

    Thread-safety: NOT thread-safe. Use from a single asyncio task.

    Args:
        max_chars: Force flush after this many characters even without
                   a sentence boundary. Prevents unbounded buffering
                   for long run-on sentences.
                   Default 80 ≈ one natural sentence.
        min_chars: Don't flush sentences shorter than this (avoids
                   sending fragments like "Oh." to TTS).
                   Default 10.
    """

    def __init__(self, max_chars: int = 80, min_chars: int = 10) -> None:
        self._buffer: str = ""
        self._max_chars = max(max_chars, 20)
        self._min_chars = max(min_chars, 1)
        self._total_pushed: int = 0
        self._total_flushed: int = 0

    def push(self, token: str) -> List[str]:
        """Add an LLM token. Returns list of complete sentences to synthesize.

        May return 0, 1, or multiple sentences depending on token content.
        """
        if not token:
            return []

        self._buffer += token
        self._total_pushed += len(token)

        results: List[str] = []

        while True:
            sentence = self._try_extract_sentence()
            if sentence is None:
                break
            results.append(sentence)

        if len(self._buffer) >= self._max_chars:
            flushed = self._force_flush_at_break()
            if flushed:
                results.append(flushed)

        return results

    def flush(self) -> str | None:
        """Flush any remaining buffered text. Call at end of LLM response."""
        if not self._buffer.strip():
            self._buffer = ""
            return None
        text = self._buffer.strip()
        self._buffer = ""
        self._total_flushed += len(text)
        return text

    @property
    def pending(self) -> str:
        """Current buffered text (not yet flushed)."""
        return self._buffer

    @property
    def pending_chars(self) -> int:
        return len(self._buffer)

    @property
    def stats(self) -> dict:
        return {
            "total_pushed_chars": self._total_pushed,
            "total_flushed_chars": self._total_flushed,
            "pending_chars": len(self._buffer),
        }

    def _try_extract_sentence(self) -> str | None:
        """Try to extract a complete sentence from the buffer."""
        match = _SENTENCE_END_RE.search(self._buffer)
        if match is None:
            return None

        split_pos = match.start()
        candidate = self._buffer[:split_pos].strip()

        last_word = candidate.rsplit(None, 1)[-1].lower() if candidate else ""
        if last_word in _ABBREVIATIONS:
            return None

        if len(candidate) < self._min_chars:
            return None

        self._buffer = self._buffer[match.end():]
        self._total_flushed += len(candidate)
        return candidate

    def _force_flush_at_break(self) -> str | None:
        """Force flush at the best natural break point."""
        buf = self._buffer

        best_pos = -1
        for delim in (",", ";", ":", " — ", " - ", " "):
            pos = buf.rfind(delim, self._min_chars)
            if pos > best_pos:
                best_pos = pos + len(delim)

        if best_pos <= self._min_chars:
            text = buf.strip()
            self._buffer = ""
        else:
            text = buf[:best_pos].strip()
            self._buffer = buf[best_pos:]

        if text:
            self._total_flushed += len(text)
            return text
        return None
