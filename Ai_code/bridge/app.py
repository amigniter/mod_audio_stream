"""
OpenAI Realtime <-> FreeSWITCH bridge — playout engine.

Audio path (standard — OpenAI built-in voice):
  FreeSWITCH 8kHz PCM -> (soxr resample 8->24kHz) -> OpenAI Realtime API
  OpenAI 24kHz PCM -> JitterBuffer -> 1 frame/20ms tick -> JSON -> C module
  C module Speex 24->8kHz -> inject_buffer -> WRITE_REPLACE -> caller

Audio path (custom voice — YOUR voice from ww.wav):
  FreeSWITCH 8kHz PCM -> (soxr resample 8->24kHz) -> OpenAI Realtime API
  OpenAI TEXT response -> SentenceBuffer -> TTS Engine (your cloned voice)
  TTS 24kHz PCM -> JitterBuffer -> 1 frame/20ms tick -> JSON -> C module
  C module Speex 24->8kHz -> inject_buffer -> WRITE_REPLACE -> caller

Design (matching ChatGPT Voice quality):
  1. JitterBuffer is UNBOUNDED — audio is NEVER dropped.
  2. Exactly ONE frame per 20ms tick — no multi-drain, no bursts.
  3. Clock starts AFTER prebuffer is satisfied — no stale-clock catch-up.
  4. Barge-in: clear JitterBuffer + send response.cancel immediately.
  5. High-quality soxr resampler for input (8->24kHz).
  6. Custom TTS: text-only OpenAI + streaming voice synthesis.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import websockets

from .audio import ceil_to_frame, ensure_even_bytes, frame_bytes
from .config import BridgeConfig
from .fs_payloads import FsAudioContract, fs_handshake_json, fs_stream_audio_json
from .openai_client import build_ssl_context, connect_openai_realtime
from .resample import Resampler, get_backend as get_resample_backend
from .tts import TTSEngine, SentenceBuffer, TTSCache

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
#  JitterBuffer — unbounded, frame-aligned, never drops audio
# ─────────────────────────────────────────────────────────────────
class JitterBuffer:
    """Frame-aligned jitter buffer backed by an unbounded deque.

    OpenAI sends audio 3-5x faster than real-time.  A 15-second response
    arrives in ~3 seconds.  The old bytearray with a hard cap was dropping
    the BEGINNING of sentences — that was the main crackling source.

    This buffer NEVER drops.  It just queues.  The playout loop drains
    it at exactly 1 frame per tick (real-time rate).
    """

    __slots__ = (
        "_frames", "_frame_bytes", "_frame_ms",
        "_remainder", "_total_enqueued", "_total_dequeued",
    )

    def __init__(self, frame_bytes_: int, frame_ms: float) -> None:
        self._frames: deque[bytes] = deque()
        self._frame_bytes = frame_bytes_
        self._frame_ms = frame_ms
        self._remainder = bytearray()
        self._total_enqueued = 0
        self._total_dequeued = 0

    def enqueue_pcm(self, pcm: bytes) -> int:
        """Add raw PCM, split into frame-aligned chunks. Returns frames added."""
        self._remainder.extend(pcm)
        added = 0
        fb = self._frame_bytes
        while len(self._remainder) >= fb:
            self._frames.append(bytes(self._remainder[:fb]))
            del self._remainder[:fb]
            added += 1
        self._total_enqueued += added
        return added

    def dequeue(self) -> Optional[bytes]:
        """Pop one frame. Returns None if empty (underrun)."""
        if self._frames:
            self._total_dequeued += 1
            return self._frames.popleft()
        return None

    def clear(self) -> int:
        """Clear all buffered data. Returns bytes cleared."""
        n = len(self._frames) * self._frame_bytes + len(self._remainder)
        self._frames.clear()
        self._remainder.clear()
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

                # Mono downmix if needed
                if cfg.fs_channels != 1:
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", DeprecationWarning)
                            import audioop
                        out_pcm = audioop.tomono(out_pcm, 2, 0.5, 0.5)
                    except ImportError:
                        try:
                            import audioop_lts as audioop  # type: ignore
                            out_pcm = audioop.tomono(out_pcm, 2, 0.5, 0.5)
                        except ImportError:
                            logger.warning("audioop unavailable — cannot downmix stereo input")

                # High-quality resample 8k->24k
                if input_resampler is not None:
                    out_pcm = input_resampler.process(out_pcm)

                out_pcm = ensure_even_bytes(out_pcm)

                # Never send empty audio to OpenAI
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


async def pump_openai_to_freeswitch(
    openai_ws: websockets.WebSocketClientProtocol,
    upstream_ws: websockets.WebSocketServerProtocol,
    cfg: BridgeConfig,
    tracker: InputAudioTracker,
    send_lock: Optional[asyncio.Lock] = None,
    tts_engine: Optional[TTSEngine] = None,
    tts_cache: Optional[TTSCache] = None,
) -> None:
    """Receive events from OpenAI and either:
    - (standard) decode audio deltas into JitterBuffer
    - (custom voice) accumulate text deltas -> SentenceBuffer -> TTS -> JitterBuffer
    """
    use_custom_tts = tts_engine is not None
    contract = FsAudioContract(
        sample_rate=cfg.fs_out_sample_rate,
        channels=cfg.fs_channels,
        frame_ms=cfg.fs_frame_ms,
    )

    openai_out_rate = cfg.openai_output_sample_rate
    openai_out_channels = 1
    out_frame_bytes = frame_bytes(openai_out_rate, openai_out_channels, contract.frame_ms)

    jbuf = JitterBuffer(frame_bytes_=out_frame_bytes, frame_ms=float(contract.frame_ms))

    prebuffer_bytes = ceil_to_frame(
        frame_bytes(openai_out_rate, openai_out_channels, max(cfg.playout_prebuffer_ms, 0)),
        out_frame_bytes,
    )

    response_in_flight = False
    last_response_create_t = 0.0
    audio_chunks_received = 0
    audio_bytes_received = 0

    user_text_buf: list[str] = []
    ai_text_buf: list[str] = []

    item_max_buffer_ms = cfg.openai_item_max_buffer_ms
    openai_in_rate = cfg.openai_input_sample_rate
    item_max_bytes = frame_bytes(openai_in_rate, 1, item_max_buffer_ms) if item_max_buffer_ms > 0 else 0

    item_turn_in_flight = False
    last_item_turn_end_t = 0.0
    item_turn_min_interval_s = 0.25

    # ── Custom TTS pipeline state ──
    sentence_buffer: Optional[SentenceBuffer] = None
    tts_tasks: list[asyncio.Task] = []
    tts_voice_id = cfg.tts_voice_id or ""
    if use_custom_tts:
        sentence_buffer = SentenceBuffer(
            max_chars=cfg.tts_sentence_max_chars,
            min_chars=cfg.tts_sentence_min_chars,
        )
        logger.info(
            "Custom TTS active: engine=%s voice_id=%s cache=%s",
            tts_engine.name, tts_voice_id or "(default)",
            "enabled" if tts_cache else "disabled",
        )

    async def _synthesize_and_enqueue(text: str) -> None:
        """Synthesize a sentence via TTS and enqueue PCM into JitterBuffer."""
        if not text.strip():
            return

        # Check cache first
        if tts_cache is not None:
            cached = await tts_cache.get(text, tts_voice_id)
            if cached is not None:
                jbuf.enqueue_pcm(cached.pcm16)
                logger.debug("TTS cache hit: '%s' (%d bytes)", text[:40], len(cached.pcm16))
                return

        t0 = time.monotonic()
        pcm_parts: list[bytes] = []
        chunk_count = 0
        try:
            async for chunk in tts_engine.synthesize_stream(text, voice_id=tts_voice_id or None):
                pcm = ensure_even_bytes(chunk.pcm16)
                if pcm:
                    jbuf.enqueue_pcm(pcm)
                    pcm_parts.append(pcm)
                    chunk_count += 1
        except Exception:
            logger.exception("TTS synthesis failed for '%.60s'", text)
            return

        elapsed_ms = (time.monotonic() - t0) * 1000
        total_bytes = sum(len(p) for p in pcm_parts)
        logger.info(
            "TTS: '%s' → %d chunks, %d bytes, %.0fms",
            text[:50], chunk_count, total_bytes, elapsed_ms,
        )

        # Store in cache
        if tts_cache is not None and total_bytes > 0:
            full_pcm = b"".join(pcm_parts)
            await tts_cache.put(
                text, tts_voice_id, full_pcm,
                sample_rate=tts_engine.output_sample_rate,
                channels=tts_engine.output_channels,
                synthesis_ms=elapsed_ms,
            )

    def _schedule_tts(text: str) -> None:
        """Schedule TTS synthesis as a background task."""
        task = asyncio.create_task(_synthesize_and_enqueue(text))
        tts_tasks.append(task)
        # Clean up completed tasks
        tts_tasks[:] = [t for t in tts_tasks if not t.done()]

    async def _flush_sentence_buffer() -> None:
        """Flush remaining text in sentence buffer to TTS."""
        if sentence_buffer is not None:
            remaining = sentence_buffer.flush()
            if remaining:
                _schedule_tts(remaining)

    async def _cancel_tts_tasks() -> None:
        """Cancel all in-flight TTS synthesis (barge-in)."""
        for t in tts_tasks:
            if not t.done():
                t.cancel()
        for t in tts_tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        tts_tasks.clear()
        if sentence_buffer is not None:
            sentence_buffer.flush()  # discard pending text

    async def _send_item_audio_and_respond(reason: str) -> None:
        nonlocal response_in_flight, item_turn_in_flight, last_item_turn_end_t
        if response_in_flight:
            return

        now = time.monotonic()
        if item_turn_in_flight and (now - last_item_turn_end_t) < item_turn_min_interval_s:
            return
        item_turn_in_flight = True
        last_item_turn_end_t = now

        audio_buf = tracker.item_audio_buf
        if not isinstance(audio_buf, (bytearray, bytes)) or len(audio_buf) == 0:
            return

        if item_max_bytes > 0 and len(audio_buf) > item_max_bytes:
            drop = len(audio_buf) - item_max_bytes
            if isinstance(audio_buf, bytearray):
                del audio_buf[:drop]
            else:
                tracker.item_audio_buf = bytearray(audio_buf[drop:])

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
            if send_lock is not None:
                async with send_lock:
                    await openai_ws.send(json.dumps(item_evt))
            else:
                await openai_ws.send(json.dumps(item_evt))
        except Exception:
            logger.exception("Failed sending conversation.item.create")
            item_turn_in_flight = False
            return

        if isinstance(audio_buf, bytearray):
            del audio_buf[:len(local_audio)]
        else:
            tracker.item_audio_buf = bytearray()
        tracker.on_committed()

        if not _can_create_response():
            item_turn_in_flight = False
            return
        _mark_response_created()
        response_in_flight = True
        try:
            resp_modalities = ["text"] if use_custom_tts else ["audio", "text"]
            pkt2 = json.dumps({
                "type": "response.create",
                "response": {"modalities": resp_modalities},
            })
            if send_lock is not None:
                async with send_lock:
                    await openai_ws.send(pkt2)
            else:
                await openai_ws.send(pkt2)
        except Exception:
            logger.exception("Failed sending response.create")
            response_in_flight = False
            item_turn_in_flight = False

    def _can_create_response() -> bool:
        now = time.monotonic()
        return (now - last_response_create_t) * 1000.0 >= max(cfg.response_min_interval_ms, 0)

    def _mark_response_created() -> None:
        nonlocal last_response_create_t
        last_response_create_t = time.monotonic()

    async def _playout_loop() -> None:
        step_s = contract.frame_ms / 1000.0  # 0.020

        frames_sent = 0
        underruns = 0
        last_stats_t = time.monotonic()
        last_stats_sent = 0
        last_stats_underruns = 0

        silence_frame = b"\x00" * out_frame_bytes

        PREBUFFER_TIMEOUT_S = 5.0

        prebuffer_start = time.monotonic()
        if prebuffer_bytes > 0:
            while jbuf.buffered_bytes < prebuffer_bytes:
                if time.monotonic() - prebuffer_start > PREBUFFER_TIMEOUT_S:
                    logger.warning(
                        "Prebuffer timeout (%.1fs) — starting playout with %d/%d bytes",
                        PREBUFFER_TIMEOUT_S, jbuf.buffered_bytes, prebuffer_bytes,
                    )
                    break
                await asyncio.sleep(0.005)
        else:
            while jbuf.buffered_frames == 0:
                if time.monotonic() - prebuffer_start > PREBUFFER_TIMEOUT_S:
                    logger.warning("Prebuffer timeout — no audio received, starting anyway")
                    break
                await asyncio.sleep(0.005)

        next_t = time.monotonic() + step_s

        try:
            while True:
                now = time.monotonic()
                sleep_s = next_t - now
                if sleep_s > 0.0005:
                    await asyncio.sleep(sleep_s)

                now = time.monotonic()

                if now - next_t > step_s * 2:
                    next_t = now

                next_t += step_s

                frame = jbuf.dequeue()
                if frame is not None:
                    if cfg.fs_send_json_audio:
                        payload = fs_stream_audio_json(
                            frame, contract,
                            sample_rate_override=openai_out_rate,
                            channels_override=openai_out_channels,
                        )
                    else:
                        payload = frame
                    await upstream_ws.send(payload)
                    frames_sent += 1
                else:
                    underruns += 1
                    if cfg.fs_send_json_audio:
                        payload = fs_stream_audio_json(
                            silence_frame, contract,
                            sample_rate_override=openai_out_rate,
                            channels_override=openai_out_channels,
                        )
                    else:
                        payload = silence_frame
                    await upstream_ws.send(payload)

                now_s = time.monotonic()
                if now_s - last_stats_t >= 5.0:
                    dt = now_s - last_stats_t
                    d_sent = frames_sent - last_stats_sent
                    d_under = underruns - last_stats_underruns
                    expected = int(dt / step_s)
                    actual = d_sent + d_under
                    logger.info(
                        "Playout: sent=%d (+%d/%.1fs) buf_ms=%.0f queued=%d "
                        "underruns=%d (+%d) expected=%d actual=%d",
                        frames_sent, d_sent, dt,
                        jbuf.buffered_ms, jbuf.buffered_frames,
                        underruns, d_under, expected, actual,
                    )
                    last_stats_t = now_s
                    last_stats_sent = frames_sent
                    last_stats_underruns = underruns
        except asyncio.CancelledError:
            return

    playout_task = asyncio.create_task(_playout_loop())

    try:
        async for message in openai_ws:
            if isinstance(message, (bytes, bytearray)):
                continue

            evt = _safe_json_loads(str(message))
            if not evt:
                continue

            evt_type = evt.get("type")

            if evt_type in (
                "input_audio_transcription.delta",
                "input_audio_transcription",
                "conversation.item.input_audio_transcription.delta",
                "conversation.item.input_audio_transcription.completed",
            ):
                t = _extract_text(evt)
                if t:
                    if evt_type.endswith(".completed"):
                        user_text_buf.append(t)
                        logger.info("USER_TEXT: %s", "".join(user_text_buf).strip())
                        user_text_buf.clear()
                continue

            if evt_type in (
                "response.text.delta",
                "response.output_text.delta",
                "response.text",
                "response.output_text",
                "response.audio_transcript.delta",
                "response.audio_transcript.done",
            ):
                t = _extract_text(evt)
                if t:
                    if evt_type.endswith(".done"):
                        ai_text_buf.clear()
                        ai_text_buf.append(t)
                        logger.info("AI_TEXT: %s", t.strip())
                    else:
                        ai_text_buf.append(t)

                    # ── Custom TTS: feed text tokens into SentenceBuffer ──
                    if use_custom_tts and sentence_buffer is not None:
                        # Only process delta events (streaming tokens), not done/completed
                        is_delta = "delta" in evt_type
                        if is_delta:
                            sentences = sentence_buffer.push(t)
                            for sentence in sentences:
                                logger.debug("TTS sentence: '%s'", sentence[:60])
                                _schedule_tts(sentence)
                continue

            if evt_type in ("response.completed", "response.done"):
                ai_text_buf.clear()
                user_text_buf.clear()
                response_in_flight = False
                item_turn_in_flight = False

                # Flush any remaining text in sentence buffer to TTS
                if use_custom_tts:
                    await _flush_sentence_buffer()

                logger.info(
                    "Response done: chunks=%d bytes=%d buf_ms=%.0f",
                    audio_chunks_received, audio_bytes_received,
                    jbuf.buffered_ms,
                )
                audio_chunks_received = 0
                audio_bytes_received = 0
                continue

            if evt_type in (
                "response.audio.delta",
                "response.output_audio.delta",
                "response.audio",
                "response.output_audio",
                "response.audio_chunk",
            ):
                audio_b64 = (
                    evt.get("delta")
                    or evt.get("audio")
                    or evt.get("data")
                    or evt.get("chunk")
                    or (evt.get("payload") if isinstance(evt.get("payload"), str) else None)
                )
                if not audio_b64:
                    continue
                try:
                    pcm = base64.b64decode(audio_b64)
                except Exception:
                    continue

                pcm = ensure_even_bytes(pcm)
                audio_chunks_received += 1
                audio_bytes_received += len(pcm)

                if audio_chunks_received == 1:
                    logger.info("First audio chunk: bytes=%d", len(pcm))

                src_ch = 1
                if isinstance(evt.get("channels"), int):
                    src_ch = int(evt["channels"])
                if src_ch != 1:
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", DeprecationWarning)
                            import audioop
                        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
                    except ImportError:
                        try:
                            import audioop_lts as audioop  # type: ignore
                            pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
                        except ImportError:
                            pcm = ensure_even_bytes(pcm)

                jbuf.enqueue_pcm(pcm)
                continue

            if evt_type in (
                "input_audio_buffer.speech_stopped",
                "input_audio_buffer.speech_end",
                "input_audio_buffer.speech_end_detected",
                "input_audio_buffer.vad_stop",
            ):
                logger.info("Turn end (%s)", evt_type)
                if cfg.openai_input_mode == "item":
                    await _send_item_audio_and_respond(f"server_vad:{evt_type}")
                else:
                    tracker.on_committed()
                continue

            if evt_type in (
                "input_audio_buffer.speech_started",
                "input_audio_buffer.speech_start",
                "input_audio_buffer.speech_start_detected",
                "input_audio_buffer.vad_start",
            ):
                cleared = jbuf.clear()
                if cleared > 0:
                    cleared_ms = (cleared / out_frame_bytes) * contract.frame_ms if out_frame_bytes > 0 else 0
                    logger.info("Barge-in: cleared %d bytes (%.0f ms) from jitter buffer", cleared, cleared_ms)

                # Cancel in-flight TTS synthesis on barge-in
                if use_custom_tts:
                    await _cancel_tts_tasks()
                    logger.debug("Barge-in: cancelled TTS tasks")

                try:
                    clear_cmd = json.dumps({
                        "type": "streamAudio",
                        "data": {
                            "audioDataType": "raw",
                            "audioData": "",
                            "sampleRate": openai_out_rate,
                            "channels": openai_out_channels,
                            "clear": True,
                        },
                    })
                    await upstream_ws.send(clear_cmd)
                    logger.debug("Barge-in: sent clear command to C module")
                except Exception:
                    logger.debug("Failed to send clear command to C module")

                if response_in_flight:
                    try:
                        cancel_pkt = json.dumps({"type": "response.cancel"})
                        if send_lock is not None:
                            async with send_lock:
                                await openai_ws.send(cancel_pkt)
                        else:
                            await openai_ws.send(cancel_pkt)
                        logger.info("Barge-in: sent response.cancel")
                        response_in_flight = False
                    except Exception:
                        logger.debug("Failed to send response.cancel")
                continue

            if evt_type == "input_audio_buffer.committed":
                tracker.on_commit_acked()
                tracker.on_committed()
                continue

            if evt_type == "error":
                logger.error("OpenAI error: %s", json.dumps(evt, ensure_ascii=False))
                continue

            if evt_type in ("session.created", "session.updated"):
                logger.info("OpenAI: %s", evt_type)
                continue

            if evt_type in (
                "rate_limits.updated",
                "response.created",
                "response.output_item.added",
                "response.output_item.done",
                "response.content_part.added",
                "response.content_part.done",
                "conversation.item.created",
                "response.audio.done",
                "response.output_audio.done",
            ):
                continue

            if evt_type:
                logger.debug("OpenAI unhandled: %s", evt_type)

    finally:
        playout_task.cancel()
        try:
            await playout_task
        except asyncio.CancelledError:
            pass
        # Clean up TTS tasks
        if use_custom_tts:
            await _cancel_tts_tasks()


async def handle_call(
    cfg: BridgeConfig,
    upstream_ws: websockets.WebSocketServerProtocol,
    tts_engine: Optional[TTSEngine] = None,
    tts_cache: Optional[TTSCache] = None,
) -> None:
    peer = getattr(upstream_ws, "remote_address", None)
    logger.info("Call connected: %s", peer)

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

    try:
        await openai_ws.close()
    except Exception:
        pass
    try:
        await upstream_ws.close()
    except Exception:
        pass


async def run_server(
    cfg: BridgeConfig,
    tts_engine: Optional[TTSEngine] = None,
    tts_cache: Optional[TTSCache] = None,
) -> None:
    async def _handler(ws: websockets.WebSocketServerProtocol):
        try:
            await handle_call(cfg, ws, tts_engine=tts_engine, tts_cache=tts_cache)
        except websockets.ConnectionClosed:
            pass
        except Exception:
            logger.exception("Bridge error")

    logger.info("Listening on ws://%s:%d", cfg.host, cfg.port)
    if tts_engine is not None:
        logger.info("Custom voice active: %s", tts_engine.name)
    stop_event = asyncio.Event()

    import signal
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, OSError):
            pass  

    async with websockets.serve(_handler, cfg.host, cfg.port, max_size=16 * 1024 * 1024):
        logger.info("Server ready — press Ctrl+C to stop")
        await stop_event.wait()
        logger.info("Shutting down gracefully...")

    # Clean up TTS engine
    if tts_engine is not None:
        try:
            await tts_engine.close()
            logger.info("TTS engine closed")
        except Exception:
            logger.warning("TTS engine close failed", exc_info=True)
