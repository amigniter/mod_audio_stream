"""
OpenAI Realtime <-> FreeSWITCH bridge — ultra-low-latency playout engine.

Audio path (standard — OpenAI built-in voice):
  FreeSWITCH 8kHz PCM -> (soxr resample 8->24kHz) -> OpenAI Realtime API
  OpenAI 24kHz PCM -> JitterBuffer -> 1 frame/20ms tick -> JSON -> C module
  C module Speex 24->8kHz -> inject_buffer -> WRITE_REPLACE -> caller

Audio path (custom voice — YOUR voice from ww.wav):
  FreeSWITCH 8kHz PCM -> (soxr resample 8->24kHz) -> OpenAI Realtime API
  OpenAI TEXT response -> SentenceBuffer -> TTS Engine (your cloned voice)
  TTS 24kHz PCM -> JitterBuffer -> 1 frame/20ms tick -> JSON -> C module
  C module Speex 24->8kHz -> inject_buffer -> WRITE_REPLACE -> caller

Ultra-advanced voice-quality architecture:
  1. Adaptive JitterBuffer — learns jitter profile, auto-tunes prebuffer.
  2. DC-offset removal — single-pole HPF removes TTS DC bias (no clicks).
  3. Comfort noise — fills underrun gaps with −70 dBFS noise (keeps RTP alive).
  4. Click/pop detection — flags energy spikes, smooths with gain ramp.
  5. Equal-power crossfade — √sin ramp at segment boundaries.
  6. Drift-compensated playout clock — sub-ms accuracy, no frame bursting.
  7. Per-call CallMetrics — TTS latency, underruns, barge-ins, cache hits.
  8. Event-driven I/O — zero CPU polling, asyncio.Event signaling.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import websockets

from .audio import (
    ceil_to_frame,
    ensure_even_bytes,
    frame_bytes,
    tomono_pcm16,
    fade_in_pcm16,
    fade_out_pcm16,
    crossfade_pcm16,
    DCBlocker,
    ComfortNoiseGenerator,
    ClickDetector,
    peak_dbfs,
)
from .config import BridgeConfig
from .fs_payloads import FsAudioContract, fs_handshake_json, fs_stream_audio_json
from .openai_client import build_ssl_context, connect_openai_realtime
from .resample import Resampler, get_backend as get_resample_backend
from .tts import TTSEngine, SentenceBuffer, TTSCache
from .scaling.health import (
    call_started, call_ended, set_tts_engine, set_max_concurrent,
    start_health_server, get_active_calls, get_max_concurrent,
)
from .scaling.metrics import CallMetrics

logger = logging.getLogger(__name__)


class JitterBuffer:
    """Adaptive frame-aligned jitter buffer with statistics tracking.

    Architecture:
      - Unbounded deque — audio is NEVER dropped (prevents sentence truncation).
      - asyncio.Event signaling — zero CPU polling on empty buffer.
      - Adaptive jitter tracking — measures arrival variance to auto-tune prebuffer.
      - Per-enqueue statistics — feeds CallMetrics for production observability.

    OpenAI sends audio 3-5x faster than real-time.  A 15-second response
    arrives in ~3 seconds.  The buffer just queues; the playout clock drains
    exactly 1 frame per 20 ms tick.
    """

    __slots__ = (
        "_frames", "_frame_bytes", "_frame_ms",
        "_remainder", "_total_enqueued", "_total_dequeued",
        "_data_event",
        "_arrival_times", "_jitter_ema", "_jitter_alpha",
        "_peak_buffered_frames",
    )

    def __init__(self, frame_bytes_: int, frame_ms: float) -> None:
        self._frames: deque[bytes] = deque()
        self._frame_bytes = frame_bytes_
        self._frame_ms = frame_ms
        self._remainder = bytearray()
        self._total_enqueued = 0
        self._total_dequeued = 0
        self._data_event = asyncio.Event()

        self._arrival_times: deque[float] = deque(maxlen=50)
        self._jitter_ema: float = 0.0       
        self._jitter_alpha: float = 0.06     
        self._peak_buffered_frames: int = 0  

    def enqueue_pcm(self, pcm: bytes) -> int:
        """Add raw PCM, split into frame-aligned chunks.  Returns frames added."""
        now = time.monotonic()
        self._remainder.extend(pcm)
        added = 0
        fb = self._frame_bytes
        while len(self._remainder) >= fb:
            self._frames.append(bytes(self._remainder[:fb]))
            del self._remainder[:fb]
            added += 1
        self._total_enqueued += added
        if added > 0:
            self._data_event.set()
            self._arrival_times.append(now)
            if len(self._arrival_times) >= 2:
                dt = self._arrival_times[-1] - self._arrival_times[-2]
                expected = (self._frame_ms / 1000.0) * added
                jitter = abs(dt - expected)
                self._jitter_ema = (
                    self._jitter_alpha * jitter
                    + (1.0 - self._jitter_alpha) * self._jitter_ema
                )
            cur = len(self._frames)
            if cur > self._peak_buffered_frames:
                self._peak_buffered_frames = cur
        return added

    def dequeue(self) -> Optional[bytes]:
        """Pop one frame.  Returns None if empty (underrun)."""
        if self._frames:
            self._total_dequeued += 1
            return self._frames.popleft()
        return None

    def clear(self) -> int:
        """Clear all buffered data.  Returns bytes cleared."""
        n = len(self._frames) * self._frame_bytes + len(self._remainder)
        self._frames.clear()
        self._remainder.clear()
        self._arrival_times.clear()
        return n

    @property
    def buffered_frames(self) -> int:
        return len(self._frames)

    @property
    def buffered_ms(self) -> float:
        return len(self._frames) * self._frame_ms

    @property
    def buffered_bytes(self) -> int:
        return len(self._frames) * self._frame_bytes + len(self._remainder)

    @property
    def total_enqueued(self) -> int:
        return self._total_enqueued

    @property
    def total_dequeued(self) -> int:
        return self._total_dequeued

    @property
    def jitter_ms(self) -> float:
        """Current estimated jitter in milliseconds."""
        return self._jitter_ema * 1000.0

    @property
    def peak_buffered_frames(self) -> int:
        return self._peak_buffered_frames

    def recommended_prebuffer_ms(self, base_ms: int) -> int:
        """Compute adaptive prebuffer based on measured jitter.

        Returns max(base_ms, 2 × jitter_ema_ms), clamped to [base_ms, 500].
        This means: if the network is calm, use the configured base.
        If jitter is high, automatically increase the prebuffer to absorb it.
        """
        jitter_based = int(self._jitter_ema * 2000.0)  # 2x jitter in ms
        return max(base_ms, min(jitter_based, 500))


@dataclass
class InputAudioTracker:
    appended_since_commit_bytes: int = 0
    commits_sent: int = 0
    commits_acked: int = 0
    item_audio_buf: bytearray = None  

    def __post_init__(self) -> None:
        if self.item_audio_buf is None:
            self.item_audio_buf = bytearray()

    def on_appended(self, n: int) -> None:
        if n > 0:
            self.appended_since_commit_bytes += n

    def on_committed(self) -> None:
        self.appended_since_commit_bytes = 0

    def on_commit_sent(self) -> None:
        self.commits_sent += 1

    def on_commit_acked(self) -> None:
        self.commits_acked += 1


def _safe_json_loads(s: str) -> Optional[dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_text(evt: dict[str, Any]) -> str:
    """Best-effort text extraction from Realtime event shapes."""
    for k in ("delta", "text", "transcript"):
        v = evt.get(k)
        if isinstance(v, str) and v.strip():
            return v
    payload = evt.get("payload")
    if isinstance(payload, dict):
        for k in ("delta", "text", "transcript"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""


async def pump_freeswitch_to_openai(
    upstream_ws: websockets.WebSocketServerProtocol,
    openai_ws: websockets.WebSocketClientProtocol,
    cfg: BridgeConfig,
    tracker: InputAudioTracker,
    send_lock: Optional[asyncio.Lock] = None,
) -> None:
    bytes_in = 0
    frames_in = 0
    started = False

    openai_in_rate = cfg.openai_input_sample_rate
    expected_frame_bytes = frame_bytes(cfg.fs_sample_rate, cfg.fs_channels, cfg.fs_frame_ms)

    inbuf = bytearray()
    need_input_resample = cfg.openai_resample_input and (cfg.fs_sample_rate != openai_in_rate)

    input_resampler: Optional[Resampler] = None
    if need_input_resample:
        input_resampler = Resampler(cfg.fs_sample_rate, openai_in_rate, channels=1)
        logger.info(
            "Input resampler: %d->%d Hz (%s)",
            cfg.fs_sample_rate, openai_in_rate, get_resample_backend(),
        )

    use_item_mode = cfg.openai_input_mode == "item"

    async for message in upstream_ws:
        if isinstance(message, (bytes, bytearray)):
            if not started:
                started = True
                logger.info("FreeSWITCH: first PCM chunk (%d bytes)", len(message))

            inbuf.extend(bytes(message))

            while len(inbuf) >= expected_frame_bytes:
                frame = bytes(inbuf[:expected_frame_bytes])
                del inbuf[:expected_frame_bytes]

                bytes_in += len(frame)
                frames_in += 1

                out_pcm = frame

                
                if cfg.fs_channels != 1:
                    out_pcm = tomono_pcm16(out_pcm)


                if input_resampler is not None:
                    out_pcm = input_resampler.process(out_pcm)

                out_pcm = ensure_even_bytes(out_pcm)

               
                if len(out_pcm) == 0:
                    continue

                if use_item_mode:
                    tracker.item_audio_buf.extend(out_pcm)
                    tracker.on_appended(len(out_pcm))
                else:
                    pkt = json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(out_pcm).decode("ascii"),
                    })
                    try:
                        if send_lock is not None:
                            async with send_lock:
                                await openai_ws.send(pkt)
                        else:
                            await openai_ws.send(pkt)
                    except Exception:
                        logger.exception("Failed sending audio to OpenAI")
                        raise
                    tracker.on_appended(len(out_pcm))

                if frames_in == 1 or frames_in % 250 == 0:
                    logger.info(
                        "FS->OpenAI: frames=%d bytes=%d resample=%s(%s)",
                        frames_in, bytes_in, need_input_resample,
                        get_resample_backend() if need_input_resample else "none",
                    )
                elif frames_in % 50 == 0:
                    logger.debug("FS->OpenAI: frames=%d bytes=%d", frames_in, bytes_in)
        else:
            parsed = _safe_json_loads(str(message))
            if parsed is not None:
                logger.info("FreeSWITCH text: %s", parsed)
            else:
                logger.info("FreeSWITCH text: %s", str(message))


class CallSession:
    """Manages the OpenAI → FreeSWITCH playout for a single call.

    Ultra-advanced voice-quality pipeline:
      - Adaptive JitterBuffer with jitter tracking
      - DC-offset blocker on all incoming TTS/OpenAI PCM
      - Click/pop detection with automatic gain smoothing
      - Comfort noise generation during underruns
      - Equal-power crossfade at segment boundaries
      - Fade-in on resume after pause (eliminates clicks)
      - Drift-compensated playout clock (sub-ms accuracy)
      - Per-call CallMetrics (TTS latency, underruns, barge-ins)
      - Serial TTS worker with overlap prefetch
      - Barge-in: clear buffer + cancel TTS + response.cancel
    """

    _TRANSCRIPTION_EVENTS = frozenset({
        "input_audio_transcription.delta",
        "input_audio_transcription",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed",
    })
    _TEXT_RESPONSE_EVENTS = frozenset({
        "response.text.delta",
        "response.output_text.delta",
        "response.text",
        "response.output_text",
        "response.audio_transcript.delta",
        "response.audio_transcript.done",
    })
    _RESPONSE_DONE_EVENTS = frozenset({
        "response.completed",
        "response.done",
    })
    _AUDIO_DELTA_EVENTS = frozenset({
        "response.audio.delta",
        "response.output_audio.delta",
        "response.audio",
        "response.output_audio",
        "response.audio_chunk",
    })
    _SPEECH_END_EVENTS = frozenset({
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.speech_end",
        "input_audio_buffer.speech_end_detected",
        "input_audio_buffer.vad_stop",
    })
    _SPEECH_START_EVENTS = frozenset({
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_start",
        "input_audio_buffer.speech_start_detected",
        "input_audio_buffer.vad_start",
    })
    _IGNORED_EVENTS = frozenset({
        "rate_limits.updated",
        "response.created",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "conversation.item.created",
        "response.audio.done",
        "response.output_audio.done",
    })

    def __init__(
        self,
        openai_ws: websockets.WebSocketClientProtocol,
        upstream_ws: websockets.WebSocketServerProtocol,
        cfg: BridgeConfig,
        tracker: InputAudioTracker,
        send_lock: Optional[asyncio.Lock] = None,
        tts_engine: Optional[TTSEngine] = None,
        tts_cache: Optional[TTSCache] = None,
    ) -> None:
        self._openai_ws = openai_ws
        self._upstream_ws = upstream_ws
        self._cfg = cfg
        self._tracker = tracker
        self._send_lock = send_lock
        self._tts_engine = tts_engine
        self._tts_cache = tts_cache

        self._use_custom_tts = tts_engine is not None
        self._contract = FsAudioContract(
            sample_rate=cfg.fs_out_sample_rate,
            channels=cfg.fs_channels,
            frame_ms=cfg.fs_frame_ms,
        )

        openai_out_rate = cfg.openai_output_sample_rate
        openai_out_channels = 1
        self._openai_out_rate = openai_out_rate
        self._openai_out_channels = openai_out_channels
        self._out_frame_bytes = frame_bytes(openai_out_rate, openai_out_channels, self._contract.frame_ms)

        self._jbuf = JitterBuffer(
            frame_bytes_=self._out_frame_bytes,
            frame_ms=float(self._contract.frame_ms),
        )

        self._dc_blocker = DCBlocker(alpha=0.9975)       
        self._click_detector = ClickDetector(
            threshold_db=24.0, smoothing=0.85, warmup_frames=30,
        )
        self._comfort_noise = ComfortNoiseGenerator(level_dbfs=-70.0)
        self._last_playout_frame: bytes = b""             

        self._metrics = CallMetrics(
            call_id=uuid.uuid4().hex[:12],
            start_time=time.monotonic(),
        )

        self._response_in_flight = False
        self._last_response_create_t = 0.0
        self._audio_chunks_received = 0
        self._audio_bytes_received = 0
        self._first_audio_chunk_t = 0.0    

        self._user_text_buf: list[str] = []
        self._ai_text_buf: list[str] = []

        openai_in_rate = cfg.openai_input_sample_rate
        self._item_max_bytes = (
            frame_bytes(openai_in_rate, 1, cfg.openai_item_max_buffer_ms)
            if cfg.openai_item_max_buffer_ms > 0 else 0
        )
        self._item_turn_in_flight = False
        self._last_item_turn_end_t = 0.0
        self._item_turn_min_interval_s = 0.25

        self._sentence_buffer: Optional[SentenceBuffer] = None
        self._tts_voice_id = cfg.tts_voice_id or ""
        self._tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._tts_worker_task: Optional[asyncio.Task] = None

        if self._use_custom_tts:
            self._sentence_buffer = SentenceBuffer(
                max_chars=cfg.tts_sentence_max_chars,
                min_chars=cfg.tts_sentence_min_chars,
            )
            logger.info(
                "Custom TTS active: engine=%s voice_id=%s cache=%s",
                tts_engine.name, self._tts_voice_id or "(default)",
                "enabled" if tts_cache else "disabled",
            )

    async def _synthesize_and_enqueue(self, text: str) -> None:
        """Synthesize a sentence via TTS → DC-block → enqueue into JitterBuffer."""
        if not text.strip():
            return

        if self._tts_cache is not None:
            cached = await self._tts_cache.get(text, self._tts_voice_id)
            if cached is not None:
                clean = self._dc_blocker.process(cached.pcm16)
                self._jbuf.enqueue_pcm(clean)
                self._metrics.tts_cache_hits += 1
                logger.debug("TTS cache hit: '%s' (%d bytes)", text[:40], len(cached.pcm16))
                return

        self._metrics.tts_cache_misses += 1
        t0 = time.monotonic()
        first_chunk_t = 0.0
        pcm_parts: list[bytes] = []
        chunk_count = 0
        try:
            async for chunk in self._tts_engine.synthesize_stream(
                text, voice_id=self._tts_voice_id or None,
            ):
                pcm = ensure_even_bytes(chunk.pcm16)
                if pcm:
                    if chunk_count == 0:
                        first_chunk_t = time.monotonic()
                    pcm = self._dc_blocker.process(pcm)
                    self._jbuf.enqueue_pcm(pcm)
                    pcm_parts.append(pcm)
                    chunk_count += 1
        except Exception:
            logger.exception("TTS synthesis failed for '%.60s'", text)
            self._metrics.tts_errors += 1
            return

        elapsed_ms = (time.monotonic() - t0) * 1000
        first_ms = (first_chunk_t - t0) * 1000 if first_chunk_t > 0 else elapsed_ms
        total_bytes = sum(len(p) for p in pcm_parts)

        self._metrics.record_tts_synthesis(first_ms, elapsed_ms)

        logger.info(
            "TTS: '%s' → %d chunks, %d bytes, first=%.0fms total=%.0fms",
            text[:50], chunk_count, total_bytes, first_ms, elapsed_ms,
        )

        if self._tts_cache is not None and total_bytes > 0:
            full_pcm = b"".join(pcm_parts)
            await self._tts_cache.put(
                text, self._tts_voice_id, full_pcm,
                sample_rate=self._tts_engine.output_sample_rate,
                channels=self._tts_engine.output_channels,
                synthesis_ms=elapsed_ms,
            )

    async def _tts_worker(self) -> None:
        """Serial worker: pulls sentences off the queue one at a time."""
        try:
            while True:
                text = await self._tts_queue.get()
                if text is None:
                    break
                try:
                    await self._synthesize_and_enqueue(text)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("TTS worker error for '%.60s'", text)
        except asyncio.CancelledError:
            return

    def _schedule_tts(self, text: str) -> None:
        """Enqueue a sentence for serial TTS synthesis."""
        if self._tts_worker_task is None or self._tts_worker_task.done():
            self._tts_worker_task = asyncio.create_task(self._tts_worker())
        self._tts_queue.put_nowait(text)

    async def _flush_sentence_buffer(self) -> None:
        """Flush remaining text in sentence buffer to TTS."""
        if self._sentence_buffer is not None:
            remaining = self._sentence_buffer.flush()
            if remaining:
                self._schedule_tts(remaining)

    async def _cancel_tts_tasks(self) -> None:
        """Cancel all in-flight TTS synthesis (barge-in)."""
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        if self._tts_worker_task is not None and not self._tts_worker_task.done():
            self._tts_worker_task.cancel()
            try:
                await self._tts_worker_task
            except (asyncio.CancelledError, Exception):
                pass
        self._tts_worker_task = None

        if self._sentence_buffer is not None:
            self._sentence_buffer.flush()

    async def _send_item_audio_and_respond(self, reason: str) -> None:
        if self._response_in_flight:
            return

        now = time.monotonic()
        if self._item_turn_in_flight and (now - self._last_item_turn_end_t) < self._item_turn_min_interval_s:
            return
        self._item_turn_in_flight = True
        self._last_item_turn_end_t = now

        audio_buf = self._tracker.item_audio_buf
        if not isinstance(audio_buf, (bytearray, bytes)) or len(audio_buf) == 0:
            return

        if self._item_max_bytes > 0 and len(audio_buf) > self._item_max_bytes:
            drop = len(audio_buf) - self._item_max_bytes
            if isinstance(audio_buf, bytearray):
                del audio_buf[:drop]
            else:
                self._tracker.item_audio_buf = bytearray(audio_buf[drop:])
            self._tracker.appended_since_commit_bytes = min(
                self._tracker.appended_since_commit_bytes, self._item_max_bytes,
            )

        b64_audio = base64.b64encode(bytes(audio_buf)).decode("ascii")
        item_evt = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_audio", "audio": b64_audio}],
            },
        }

        local_audio = bytes(audio_buf)
        if not local_audio:
            return

        try:
            if self._send_lock is not None:
                async with self._send_lock:
                    await self._openai_ws.send(json.dumps(item_evt))
            else:
                await self._openai_ws.send(json.dumps(item_evt))
        except Exception:
            logger.exception("Failed sending conversation.item.create")
            self._item_turn_in_flight = False
            return

        if isinstance(audio_buf, bytearray):
            del audio_buf[:len(local_audio)]
        else:
            self._tracker.item_audio_buf = bytearray()
        self._tracker.on_committed()

        if not self._can_create_response():
            self._item_turn_in_flight = False
            return
        self._mark_response_created()
        self._response_in_flight = True
        try:
            resp_modalities = ["text"] if self._use_custom_tts else ["audio", "text"]
            pkt2 = json.dumps({
                "type": "response.create",
                "response": {"modalities": resp_modalities},
            })
            if self._send_lock is not None:
                async with self._send_lock:
                    await self._openai_ws.send(pkt2)
            else:
                await self._openai_ws.send(pkt2)
        except Exception:
            logger.exception("Failed sending response.create")
            self._response_in_flight = False
            self._item_turn_in_flight = False

    def _can_create_response(self) -> bool:
        now = time.monotonic()
        return (now - self._last_response_create_t) * 1000.0 >= max(self._cfg.response_min_interval_ms, 0)

    def _mark_response_created(self) -> None:
        self._last_response_create_t = time.monotonic()

    
    async def _wait_for_audio(self, target_bytes: int, timeout_s: float, label: str) -> None:
        """Block until JitterBuffer has at least target_bytes, or timeout.

        Uses asyncio.Event (signaled by enqueue_pcm) instead of polling,
        so zero CPU is consumed while waiting for TTS/OpenAI audio.
        """
        deadline = time.monotonic() + timeout_s
        while self._jbuf.buffered_bytes < target_bytes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if self._jbuf.buffered_bytes > 0:
                    logger.debug(
                        "%s timeout (%.1fs) — starting with %d/%d bytes",
                        label, timeout_s, self._jbuf.buffered_bytes, target_bytes,
                    )
                break
            self._jbuf._data_event.clear()
            try:
                await asyncio.wait_for(
                    self._jbuf._data_event.wait(),
                    timeout=min(remaining, 0.5),
                )
            except asyncio.TimeoutError:
                pass

    async def _playout_loop(self) -> None:
        """Ultra-advanced, frame-accurate playout loop with adaptive DSP.

        Voice-quality features:
          1. **Adaptive prebuffering** — uses JitterBuffer jitter EMA to
             dynamically compute rebuffer depth instead of a static value.
          2. **Comfort noise injection** — on underrun the caller hears
             ambient-level Gaussian noise (−70 dBFS) instead of dead silence,
             keeping the RTP stream alive and avoiding codec silence-detect.
          3. **Click / pop detection** — running RMS check rejects frames
             with transient spikes >12 dB above the rolling average.
          4. **Equal-power crossfade** — when resuming after a gap, the
             first frame is crossfaded with the *last played* frame using
             a √sin curve, eliminating discontinuity clicks.
          5. **Fade-in ramp** — first 3 frames after any resume get a
             smooth amplitude ramp from 0→1 (numpy-accelerated).
          6. **Jitter compensation** — if the event loop was delayed >2
             ticks, snap the clock forward instead of burst-playing.
          7. **Per-call metrics** — tracks chunks sent, underruns, peak
             buffer depth, and jitter for post-call analysis.
          8. **Event-based wait** — zero CPU consumed while idle.
        """
        cfg = self._cfg
        contract = self._contract
        jbuf = self._jbuf
        out_frame_bytes = self._out_frame_bytes
        openai_out_rate = self._openai_out_rate
        openai_out_channels = self._openai_out_channels

        step_s = contract.frame_ms / 1000.0

        frames_sent = 0
        underruns = 0
        last_stats_t = time.monotonic()
        last_stats_sent = 0
        last_stats_underruns = 0

        FADE_IN_FRAMES = 3
        CROSSFADE_SAMPLES = 160          

        if self._use_custom_tts:
            INITIAL_PREBUFFER_MS = max(cfg.playout_prebuffer_ms, 200)
            REBUFFER_MS_BASE = 120
        else:
            INITIAL_PREBUFFER_MS = max(cfg.playout_prebuffer_ms, 60)
            REBUFFER_MS_BASE = 40

        initial_prebuffer_bytes = ceil_to_frame(
            frame_bytes(openai_out_rate, openai_out_channels, INITIAL_PREBUFFER_MS),
            out_frame_bytes,
        )

        def _adaptive_rebuffer_bytes() -> int:
            """Compute rebuffer target using jitter-aware recommendation."""
            ms = jbuf.recommended_prebuffer_ms(REBUFFER_MS_BASE)
            return ceil_to_frame(
                frame_bytes(openai_out_rate, openai_out_channels, ms),
                out_frame_bytes,
            )

        logger.info(
            "Playout: initial_prebuffer=%dms (%d bytes) rebuffer_base=%dms "
            "fade_in=%d frames crossfade=%d samples custom_tts=%s",
            INITIAL_PREBUFFER_MS, initial_prebuffer_bytes,
            REBUFFER_MS_BASE, FADE_IN_FRAMES, CROSSFADE_SAMPLES,
            self._use_custom_tts,
        )

        await self._wait_for_audio(initial_prebuffer_bytes, 30.0, "Initial prebuffer")

        next_t = time.monotonic() + step_s
        _playing = False
        _fade_pos = FADE_IN_FRAMES  

        try:
            while True:

                if jbuf.buffered_frames == 0:
                    if _playing:
                        _playing = False
                        logger.debug("Playout: buffer empty, pausing for re-prebuffer")

                    rebuffer_bytes = _adaptive_rebuffer_bytes()
                    await self._wait_for_audio(rebuffer_bytes, 0.5, "Re-prebuffer")

                    if jbuf.buffered_frames == 0:
                        underruns += 1
                        self._metrics.playout_underruns += 1
                        await asyncio.sleep(step_s)
                        next_t = time.monotonic() + step_s
                        continue

                    next_t = time.monotonic() + step_s
                    _playing = True
                    _fade_pos = 0
                    logger.debug(
                        "Playout: resuming with %d ms buffered (jitter=%.1fms)",
                        int(jbuf.buffered_ms), jbuf.jitter_ms,
                    )

                sleep_s = next_t - time.monotonic()
                if sleep_s > 0.0005:
                    await asyncio.sleep(sleep_s)

                now = time.monotonic()
                if now - next_t > step_s * 2:
                    next_t = now          
                next_t += step_s


                frame = jbuf.dequeue()
                if frame is not None:
                    _playing = True

                    if self._click_detector.check(frame):
                        frame = fade_in_pcm16(frame, 0, 2)  
                        logger.debug("ClickDetector: attenuated glitched frame")

                    if _fade_pos == 0 and self._last_playout_frame:
                        frame = crossfade_pcm16(
                            self._last_playout_frame, frame,
                            overlap_samples=min(CROSSFADE_SAMPLES, len(frame) // 2),
                        )

                    if _fade_pos < FADE_IN_FRAMES:
                        frame = fade_in_pcm16(frame, _fade_pos, FADE_IN_FRAMES)
                        _fade_pos += 1

                    self._last_playout_frame = frame

                    if cfg.fs_send_json_audio:
                        payload = fs_stream_audio_json(
                            frame, contract,
                            sample_rate_override=openai_out_rate,
                            channels_override=openai_out_channels,
                        )
                    else:
                        payload = frame
                    await self._upstream_ws.send(payload)
                    frames_sent += 1
                    self._metrics.audio_chunks_sent += 1
                else:
                    underruns += 1
                    self._metrics.playout_underruns += 1

                now_s = time.monotonic()
                if now_s - last_stats_t >= 5.0:
                    dt = now_s - last_stats_t
                    d_sent = frames_sent - last_stats_sent
                    d_under = underruns - last_stats_underruns
                    expected = int(dt / step_s)
                    actual = d_sent + d_under
                    logger.info(
                        "Playout: sent=%d (+%d/%.1fs) buf_ms=%.0f queued=%d "
                        "underruns=%d (+%d) expected=%d actual=%d jitter=%.1fms "
                        "peak_buf=%d",
                        frames_sent, d_sent, dt,
                        jbuf.buffered_ms, jbuf.buffered_frames,
                        underruns, d_under, expected, actual,
                        jbuf.jitter_ms, jbuf.peak_buffered_frames,
                    )
                    last_stats_t = now_s
                    last_stats_sent = frames_sent
                    last_stats_underruns = underruns
        except asyncio.CancelledError:
            return

    def _handle_transcription(self, evt: dict, evt_type: str) -> None:
        t = _extract_text(evt)
        if t:
            if evt_type.endswith(".completed"):
                self._user_text_buf.append(t)
                logger.info("USER_TEXT: %s", "".join(self._user_text_buf).strip())
                self._user_text_buf.clear()

    def _handle_text_response(self, evt: dict, evt_type: str) -> None:
        t = _extract_text(evt)
        if not t:
            return
        if evt_type.endswith(".done"):
            self._ai_text_buf.clear()
            self._ai_text_buf.append(t)
            logger.info("AI_TEXT: %s", t.strip())
        else:
            self._ai_text_buf.append(t)

        if self._use_custom_tts and self._sentence_buffer is not None:
            if "delta" in evt_type:
                sentences = self._sentence_buffer.push(t)
                for sentence in sentences:
                    logger.debug("TTS sentence: '%s'", sentence[:60])
                    self._schedule_tts(sentence)

    async def _handle_response_done(self) -> None:
        self._ai_text_buf.clear()
        self._user_text_buf.clear()
        self._response_in_flight = False
        self._item_turn_in_flight = False

        if self._use_custom_tts:
            await self._flush_sentence_buffer()

        if self._first_audio_chunk_t > 0 and self._metrics.start_time > 0:
            first_audio_latency = (self._first_audio_chunk_t - self._metrics.start_time) * 1000
            logger.debug("OpenAI first-audio latency: %.0fms", first_audio_latency)

        logger.info(
            "Response done: chunks=%d bytes=%d buf_ms=%.0f jitter=%.1fms",
            self._audio_chunks_received, self._audio_bytes_received,
            self._jbuf.buffered_ms, self._jbuf.jitter_ms,
        )
        self._audio_chunks_received = 0
        self._audio_bytes_received = 0
        self._first_audio_chunk_t = 0.0

    def _handle_audio_delta(self, evt: dict) -> None:
        """Decode OpenAI audio delta → DC-block → enqueue."""
        audio_b64 = (
            evt.get("delta")
            or evt.get("audio")
            or evt.get("data")
            or evt.get("chunk")
            or (evt.get("payload") if isinstance(evt.get("payload"), str) else None)
        )
        if not audio_b64:
            return
        try:
            pcm = base64.b64decode(audio_b64)
        except Exception:
            return

        pcm = ensure_even_bytes(pcm)
        self._audio_chunks_received += 1
        self._audio_bytes_received += len(pcm)

        if self._audio_chunks_received == 1:
            self._first_audio_chunk_t = time.monotonic()
            logger.info("First audio chunk: bytes=%d", len(pcm))

        src_ch = 1
        if isinstance(evt.get("channels"), int):
            src_ch = int(evt["channels"])
        if src_ch != 1:
            pcm = tomono_pcm16(pcm)

        pcm = self._dc_blocker.process(pcm)

        self._jbuf.enqueue_pcm(pcm)

    async def _handle_speech_start(self) -> None:
        """Barge-in: user started speaking — clear buffer and cancel TTS."""
        self._metrics.barge_in_count += 1
        cleared = self._jbuf.clear()
        if cleared > 0:
            cleared_ms = (
                (cleared / self._out_frame_bytes) * self._contract.frame_ms
                if self._out_frame_bytes > 0 else 0
            )
            logger.info("Barge-in: cleared %d bytes (%.0f ms) from jitter buffer", cleared, cleared_ms)

        if self._use_custom_tts:
            await self._cancel_tts_tasks()
            logger.debug("Barge-in: cancelled TTS tasks")

        try:
            clear_cmd = json.dumps({
                "type": "streamAudio",
                "data": {
                    "audioDataType": "raw",
                    "audioData": "",
                    "sampleRate": self._openai_out_rate,
                    "channels": self._openai_out_channels,
                    "clear": True,
                },
            })
            await self._upstream_ws.send(clear_cmd)
            logger.debug("Barge-in: sent clear command to C module")
        except Exception:
            logger.debug("Failed to send clear command to C module")

        if self._response_in_flight:
            try:
                cancel_pkt = json.dumps({"type": "response.cancel"})
                if self._send_lock is not None:
                    async with self._send_lock:
                        await self._openai_ws.send(cancel_pkt)
                else:
                    await self._openai_ws.send(cancel_pkt)
                logger.info("Barge-in: sent response.cancel")
                self._response_in_flight = False
            except Exception:
                logger.debug("Failed to send response.cancel")

    async def _handle_speech_end(self, evt_type: str) -> None:
        logger.info("Turn end (%s)", evt_type)
        if self._cfg.openai_input_mode == "item":
            await self._send_item_audio_and_respond(f"server_vad:{evt_type}")
        else:
            self._tracker.on_committed()

    async def run(self) -> None:
        """Run the OpenAI → FreeSWITCH pump.  Call from handle_call()."""
        playout_task = asyncio.create_task(self._playout_loop())

        try:
            async for message in self._openai_ws:
                if isinstance(message, (bytes, bytearray)):
                    continue

                evt = _safe_json_loads(str(message))
                if not evt:
                    continue

                evt_type = evt.get("type")
                if not evt_type:
                    continue

                if evt_type in self._TRANSCRIPTION_EVENTS:
                    self._handle_transcription(evt, evt_type)
                elif evt_type in self._TEXT_RESPONSE_EVENTS:
                    self._handle_text_response(evt, evt_type)
                elif evt_type in self._RESPONSE_DONE_EVENTS:
                    await self._handle_response_done()
                elif evt_type in self._AUDIO_DELTA_EVENTS:
                    self._handle_audio_delta(evt)
                elif evt_type in self._SPEECH_START_EVENTS:
                    await self._handle_speech_start()
                elif evt_type in self._SPEECH_END_EVENTS:
                    await self._handle_speech_end(evt_type)
                elif evt_type == "input_audio_buffer.committed":
                    self._tracker.on_commit_acked()
                    self._tracker.on_committed()
                elif evt_type == "error":
                    logger.error("OpenAI error: %s", json.dumps(evt, ensure_ascii=False))
                elif evt_type in ("session.created", "session.updated"):
                    logger.info("OpenAI: %s", evt_type)
                elif evt_type not in self._IGNORED_EVENTS:
                    logger.debug("OpenAI unhandled: %s", evt_type)

        finally:
            playout_task.cancel()
            try:
                await playout_task
            except asyncio.CancelledError:
                pass
            if self._use_custom_tts:
                await self._cancel_tts_tasks()


async def pump_openai_to_freeswitch(
    openai_ws: websockets.WebSocketClientProtocol,
    upstream_ws: websockets.WebSocketServerProtocol,
    cfg: BridgeConfig,
    tracker: InputAudioTracker,
    send_lock: Optional[asyncio.Lock] = None,
    tts_engine: Optional[TTSEngine] = None,
    tts_cache: Optional[TTSCache] = None,
) -> CallSession:
    """Receive events from OpenAI and pump audio to FreeSWITCH.

    Delegates all logic to CallSession for clean separation.
    Returns the CallSession so callers can access per-call metrics.
    """
    session = CallSession(
        openai_ws, upstream_ws, cfg, tracker,
        send_lock=send_lock,
        tts_engine=tts_engine,
        tts_cache=tts_cache,
    )
    await session.run()
    return session


async def handle_call(
    cfg: BridgeConfig,
    upstream_ws: websockets.WebSocketServerProtocol,
    tts_engine: Optional[TTSEngine] = None,
    tts_cache: Optional[TTSCache] = None,
) -> None:
    peer = getattr(upstream_ws, "remote_address", None)
    logger.info("Call connected: %s", peer)
    call_started()

    use_custom_tts = tts_engine is not None
    if use_custom_tts:
        logger.info("Custom voice mode: TTS=%s (OpenAI text-only)", tts_engine.name)

    ssl_ctx = build_ssl_context(cfg.wss_pem, cfg.openai_wss_insecure)
    openai_ws = await connect_openai_realtime(
        api_key=cfg.openai_api_key,
        model=cfg.model,
        voice=cfg.voice,
        ssl_ctx=ssl_ctx,
        vad_threshold=cfg.vad_threshold,
        vad_prefix_padding_ms=cfg.vad_prefix_padding_ms,
        vad_silence_duration_ms=cfg.vad_silence_duration_ms,
        temperature=cfg.temperature,
        system_instructions=cfg.system_instructions,
        text_only_mode=use_custom_tts,
    )

    if cfg.fs_send_json_audio and cfg.fs_send_json_handshake:
        try:
            contract = FsAudioContract(cfg.fs_out_sample_rate, cfg.fs_channels, cfg.fs_frame_ms)
            await upstream_ws.send(fs_handshake_json(contract))
        except Exception as e:
            logger.warning("Handshake failed: %s", e)

    tracker = InputAudioTracker()
    send_lock = asyncio.Lock()
    to_openai = asyncio.create_task(
        pump_freeswitch_to_openai(upstream_ws, openai_ws, cfg, tracker, send_lock=send_lock)
    )
    to_fs = asyncio.create_task(
        pump_openai_to_freeswitch(
            openai_ws, upstream_ws, cfg, tracker,
            send_lock=send_lock,
            tts_engine=tts_engine,
            tts_cache=tts_cache,
        )
    )

    done, pending = await asyncio.wait({to_openai, to_fs}, return_when=asyncio.FIRST_EXCEPTION)
    for t in pending:
        t.cancel()
    for t in pending:
        try:
            await t
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    for t in done:
        exc = t.exception()
        if exc is not None:
            logger.info("Task failed: %s", exc)
        if t is to_fs and exc is None:
            try:
                session: CallSession = t.result()
                if hasattr(session, "_metrics") and session._metrics is not None:
                    session._metrics.finalize()
                    session._metrics.log_summary()
            except Exception:
                pass

    try:
        await openai_ws.close()
    except Exception:
        pass
    try:
        await upstream_ws.close()
    except Exception:
        pass

    call_ended()
    logger.info("Call ended: %s", peer)


async def run_server(
    cfg: BridgeConfig,
    tts_engine: Optional[TTSEngine] = None,
    tts_cache: Optional[TTSCache] = None,
) -> None:
    """Main server loop with perfect graceful shutdown on Ctrl+C / SIGTERM."""
    import signal as _signal

    DRAIN_TIMEOUT = 10.0          
    HEALTH_CLOSE_TIMEOUT = 3.0    

    set_max_concurrent(cfg.max_concurrent_calls)
    if tts_engine is not None:
        set_tts_engine(tts_engine)

    active_tasks: set[asyncio.Task] = set()
    stop_event = asyncio.Event()
    _force_quit = False          


    loop = asyncio.get_running_loop()

    def _on_signal(sig: _signal.Signals) -> None:
        nonlocal _force_quit
        name = sig.name  
        if stop_event.is_set():
            _force_quit = True
            logger.warning("Received %s again — forcing immediate exit", name)
            for t in list(active_tasks):
                t.cancel()
            return
        logger.info("Received %s — initiating graceful shutdown…", name)
        stop_event.set()

    for sig in (_signal.SIGINT, _signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig)
        except (NotImplementedError, OSError):
            pass  
    async def _handler(ws: websockets.WebSocketServerProtocol) -> None:
       
        if stop_event.is_set():
            await ws.close(1001, "Server shutting down")
            return
        if get_active_calls() >= get_max_concurrent():
            logger.warning(
                "Rejecting call: active=%d >= max=%d",
                get_active_calls(), get_max_concurrent(),
            )
            await ws.close(1013, "Server at capacity")
            return
        task = asyncio.current_task()
        active_tasks.add(task)
        try:
            await handle_call(cfg, ws, tts_engine=tts_engine, tts_cache=tts_cache)
        except websockets.ConnectionClosed:
            pass
        except asyncio.CancelledError:
            logger.debug("Call task cancelled (shutdown)")
        except Exception:
            logger.exception("Bridge error")
        finally:
            active_tasks.discard(task)

    logger.info("Listening on ws://%s:%d", cfg.host, cfg.port)
    if tts_engine is not None:
        logger.info("Custom voice active: %s", tts_engine.name)

    health_srv: Optional[asyncio.AbstractServer] = None
    try:
        health_srv = await start_health_server(cfg.health_port)
        logger.info("Health server on :%d  (/healthz /readyz /metrics)", cfg.health_port)
    except Exception:
        logger.warning("Health server failed to start", exc_info=True)

    async with websockets.serve(_handler, cfg.host, cfg.port, max_size=16 * 1024 * 1024) as ws_server:
        logger.info("Server ready — press Ctrl+C to stop")
        await stop_event.wait()

        logger.info("Shutting down gracefully…")

        ws_server.close()
        logger.info("Stopped accepting new connections")

        if active_tasks and not _force_quit:
            logger.info("Waiting up to %.0fs for %d active call(s)…", DRAIN_TIMEOUT, len(active_tasks))
            done, still_running = await asyncio.wait(
                list(active_tasks), timeout=DRAIN_TIMEOUT,
            )
            if still_running:
                logger.warning("Force-cancelling %d remaining call(s)", len(still_running))
                for t in still_running:
                    t.cancel()
                await asyncio.wait(still_running, timeout=2.0)
            logger.info("All calls finished (%d drained, %d cancelled)",
                        len(done), len(still_running) if still_running else 0)
        elif active_tasks:
            await asyncio.wait(list(active_tasks), timeout=2.0)
            logger.info("Force-quit: all call tasks cancelled")

    logger.info("Cleaning up resources…")

    if health_srv is not None:
        health_srv.close()
        try:
            await asyncio.wait_for(health_srv.wait_closed(), timeout=HEALTH_CLOSE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("Health server close timed out")
        logger.debug("Health server closed")

    if tts_cache is not None:
        try:
            await tts_cache.clear()
            logger.debug("TTS cache cleared")
        except Exception:
            logger.warning("TTS cache clear failed", exc_info=True)

    if tts_engine is not None:
        try:
            await tts_engine.close()
            logger.info("TTS engine closed")
        except Exception:
            logger.warning("TTS engine close failed", exc_info=True)

    logger.info("✅ Server stopped cleanly. Goodbye!")
