"""
Call-level metrics collection for monitoring and alerting.

Tracks per-call and aggregate metrics for production observability:
  - Call duration, latency percentiles
  - TTS synthesis latency (first chunk, total)
  - LLM response latency
  - Barge-in count
  - Cache hit rate
  - Error rates

Exposes metrics for Prometheus scraping or JSON export.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class CallMetrics:
    """Per-call metrics collected during a single IVR session."""

    call_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0

    llm_requests: int = 0
    llm_first_token_ms: float = 0.0
    llm_total_tokens: int = 0

    tts_requests: int = 0
    tts_first_chunk_ms_list: List[float] = field(default_factory=list)
    tts_total_ms_list: List[float] = field(default_factory=list)
    tts_cache_hits: int = 0
    tts_cache_misses: int = 0

    audio_chunks_sent: int = 0
    audio_bytes_sent: int = 0
    playout_underruns: int = 0
    barge_in_count: int = 0

    tts_errors: int = 0
    tts_failovers: int = 0

    @property
    def duration_s(self) -> float:
        if self.end_time > 0 and self.start_time > 0:
            return self.end_time - self.start_time
        return 0.0

    @property
    def avg_tts_first_chunk_ms(self) -> float:
        if not self.tts_first_chunk_ms_list:
            return 0.0
        return sum(self.tts_first_chunk_ms_list) / len(self.tts_first_chunk_ms_list)

    @property
    def p95_tts_first_chunk_ms(self) -> float:
        if not self.tts_first_chunk_ms_list:
            return 0.0
        sorted_vals = sorted(self.tts_first_chunk_ms_list)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def tts_cache_hit_rate(self) -> float:
        total = self.tts_cache_hits + self.tts_cache_misses
        return (self.tts_cache_hits / total * 100.0) if total > 0 else 0.0

    def record_tts_synthesis(self, first_chunk_ms: float, total_ms: float) -> None:
        self.tts_requests += 1
        self.tts_first_chunk_ms_list.append(first_chunk_ms)
        self.tts_total_ms_list.append(total_ms)

    def finalize(self) -> None:
        self.end_time = time.monotonic()

    def summary(self) -> dict:
        return {
            "call_id": self.call_id,
            "duration_s": round(self.duration_s, 2),
            "llm_requests": self.llm_requests,
            "tts_requests": self.tts_requests,
            "avg_tts_first_chunk_ms": round(self.avg_tts_first_chunk_ms, 1),
            "p95_tts_first_chunk_ms": round(self.p95_tts_first_chunk_ms, 1),
            "tts_cache_hit_rate": round(self.tts_cache_hit_rate, 1),
            "barge_in_count": self.barge_in_count,
            "playout_underruns": self.playout_underruns,
            "tts_errors": self.tts_errors,
            "tts_failovers": self.tts_failovers,
        }

    def log_summary(self) -> None:
        s = self.summary()
        logger.info(
            "CALL_METRICS: call=%s dur=%.1fs llm=%d tts=%d "
            "tts_avg=%.0fms tts_p95=%.0fms cache_hit=%.0f%% "
            "bargein=%d underruns=%d errors=%d failovers=%d",
            s["call_id"], s["duration_s"], s["llm_requests"], s["tts_requests"],
            s["avg_tts_first_chunk_ms"], s["p95_tts_first_chunk_ms"],
            s["tts_cache_hit_rate"],
            s["barge_in_count"], s["playout_underruns"],
            s["tts_errors"], s["tts_failovers"],
        )
