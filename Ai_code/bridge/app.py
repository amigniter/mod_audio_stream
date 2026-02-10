from __future__ import annotations
import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional
import audioop
import websockets
from .audio import ceil_to_frame, ensure_even_bytes, frame_bytes, drop_oldest_frame_aligned
from .config import BridgeConfig
from .fs_payloads import FsAudioContract, fs_handshake_json, fs_stream_audio_json
from .openai_client import build_ssl_context, connect_openai_realtime

logger = logging.getLogger(__name__)


@dataclass
class InputAudioTracker:
    appended_since_commit_bytes: int = 0
    commits_sent: int = 0
    commits_acked: int = 0

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
    """Best-effort extraction of a text delta/final from common Realtime event shapes."""
    for k in ("delta", "text", "transcript"):
        v = evt.get(k)
        if isinstance(v, str) and v.strip():
            return v
    # Some events nest payloads
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

    # Resample FS audio (e.g. 8kHz) to OpenAI's expected rate (24kHz) if needed.
    openai_in_rate = int(getattr(cfg, "openai_input_sample_rate", cfg.fs_sample_rate))

    expected_frame_bytes = frame_bytes(cfg.fs_sample_rate, cfg.fs_channels, cfg.fs_frame_ms)

    inbuf = bytearray()
    need_input_resample = (cfg.fs_sample_rate != openai_in_rate)
    ratecv_state = None
    if not hasattr(tracker, "item_audio_buf"):
        setattr(tracker, "item_audio_buf", bytearray())

    async for message in upstream_ws:
        if isinstance(message, (bytes, bytearray)):
            if not started:
                started = True
                logger.info("FreeSWITCH: first PCM chunk received (%d bytes)", len(message))

            inbuf.extend(bytes(message))

            while len(inbuf) >= expected_frame_bytes:
                frame = bytes(inbuf[:expected_frame_bytes])
                del inbuf[:expected_frame_bytes]

                bytes_in += len(frame)
                frames_in += 1

                out_pcm = frame
                if cfg.fs_channels != 1:
                    try:
                        out_pcm = audioop.tomono(out_pcm, 2, 0.5, 0.5)
                    except Exception:
                        pass

                if need_input_resample:
                    try:
                        out_pcm, ratecv_state = audioop.ratecv(
                            out_pcm, 2, 1, 
                            cfg.fs_sample_rate, openai_in_rate,
                            ratecv_state,
                        )
                    except Exception:
                        logger.exception("Input resample %d->%d failed", cfg.fs_sample_rate, openai_in_rate)

                out_pcm = ensure_even_bytes(out_pcm)

                try:
                    pkt = json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(out_pcm).decode("ascii"),
                        }
                    )
                    if send_lock is not None:
                        async with send_lock:
                            await openai_ws.send(pkt)
                    else:
                        await openai_ws.send(pkt)
                except Exception:
                    logger.exception("Failed sending input_audio_buffer.append to OpenAI")
                    raise

                # Only mark appended after successful send
                tracker.on_appended(len(out_pcm))
                if getattr(cfg, "openai_input_mode", "buffer") == "item":
                    # Keep a copy so we can send a full utterance as a single conversation.item.create.
                    getattr(tracker, "item_audio_buf").extend(out_pcm)

                if frames_in == 1 or (frames_in % 50 == 0):
                    logger.debug(
                        "Appended to OpenAI: frames=%d appended_since_commit_bytes=%d",
                        frames_in,
                        tracker.appended_since_commit_bytes,
                    )

                if frames_in == 1 or (frames_in % 250 == 0):
                    # INFO telemetry every ~5s at 20ms frames.
                    logger.info(
                        "FS->OpenAI telemetry: frames=%d appended_since_commit_bytes=%d openai_rate=%d resample=%s",
                        frames_in,
                        tracker.appended_since_commit_bytes,
                        openai_in_rate,
                        need_input_resample,
                    )

            if frames_in and frames_in % 50 == 0:
                logger.debug("FreeSWITCH->OpenAI: frames=%d bytes=%d tail=%d", frames_in, bytes_in, len(inbuf))

            
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
) -> None:
    contract = FsAudioContract(
        sample_rate=int(cfg.fs_out_sample_rate),
        channels=int(cfg.fs_channels),
        frame_ms=int(cfg.fs_frame_ms),
    )

    openai_out_rate = int(getattr(cfg, "openai_output_sample_rate", contract.sample_rate))
    openai_out_channels = 1

    out_frame_bytes = frame_bytes(openai_out_rate, openai_out_channels, contract.frame_ms)

    buf = bytearray()

    silence_frame = b"\x00" * out_frame_bytes

    max_buf_ms = max(cfg.playout_max_buffer_ms, cfg.playout_prebuffer_ms, cfg.fs_frame_ms)
    max_buf_bytes = ceil_to_frame(
        frame_bytes(openai_out_rate, openai_out_channels, max_buf_ms),
        out_frame_bytes,
    )
    prebuffer_bytes = ceil_to_frame(
        frame_bytes(openai_out_rate, openai_out_channels, max(cfg.playout_prebuffer_ms, 0)),
        out_frame_bytes,
    )

    min_commit_ms = 100
    min_commit_bytes = frame_bytes(cfg.fs_sample_rate, 1, min_commit_ms)
    last_turn_evt_t = 0.0

    response_in_flight = False
    last_response_create_t = 0.0

    commit_pending = False
    commit_reason: Optional[str] = None
    commit_sent_t = 0.0
    response_ready = False
    response_ready_reason: Optional[str] = None

    user_text_buf: list[str] = []
    ai_text_buf: list[str] = []

    pending_turn_end: Optional[str] = None

    item_max_buffer_ms = int(getattr(cfg, "openai_item_max_buffer_ms", 20000))
    item_max_bytes = frame_bytes(cfg.fs_sample_rate, 1, item_max_buffer_ms)

    target_buf_ms = int(getattr(cfg, "playout_target_buffer_ms", max(0, cfg.playout_prebuffer_ms)))
    max_drain_frames = int(getattr(cfg, "playout_max_drain_frames", 8))
    if max_drain_frames < 8:
        max_drain_frames = 8
    if max_drain_frames > 20:
        max_drain_frames = 20

    target_buf_bytes = ceil_to_frame(
        frame_bytes(openai_out_rate, openai_out_channels, target_buf_ms),
        out_frame_bytes,
    )

    item_turn_in_flight = False
    last_item_turn_end_t = 0.0
    item_turn_min_interval_s = 0.25

    async def _send_item_audio_and_respond(reason: str) -> None:
        """Item-based audio input: send one utterance as a conversation item, then create a response."""
        nonlocal response_in_flight
        nonlocal item_turn_in_flight, last_item_turn_end_t
        # Avoid overlapping responses.
        if response_in_flight:
            return

        now = time.monotonic()
        if item_turn_in_flight and (now - last_item_turn_end_t) < item_turn_min_interval_s:
            logger.debug("Item mode: turn already in flight; ignoring extra turn-end (%s)", reason)
            return
        item_turn_in_flight = True
        last_item_turn_end_t = now

        audio_buf = getattr(tracker, "item_audio_buf", None)
        if not isinstance(audio_buf, (bytearray, bytes)) or len(audio_buf) == 0:
            logger.debug("Item mode: nothing to send for %s", reason)
            return

        if item_max_bytes > 0 and len(audio_buf) > item_max_bytes:
            # Drop oldest audio, keep the most recent tail. This is safer than unbounded growth.
            drop = len(audio_buf) - item_max_bytes
            if isinstance(audio_buf, bytearray):
                del audio_buf[:drop]
            else:
                audio_buf = audio_buf[drop:]
                setattr(tracker, "item_audio_buf", bytearray(audio_buf))
            logger.warning(
                "Item mode buffer capped: dropped_bytes=%d kept_bytes=%d",
                drop,
                len(getattr(tracker, "item_audio_buf")),
            )

        # Build item create payload.
        b64_audio = base64.b64encode(bytes(audio_buf)).decode("ascii")
        item_evt = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_audio", "audio": b64_audio},
                ],
            },
        }

        # Send the item atomically from a local copy. Only clear/ack the buffer after
        # the network send succeeds to avoid losing audio on transient errors.
        local_audio = bytes(audio_buf)
        if not local_audio:
            logger.debug("Item mode: nothing to send after copy (%s)", reason)
            return

        pkt = json.dumps(item_evt)
        try:
            logger.info("Item mode: sending conversation.item.create (%s)", reason)
            if send_lock is not None:
                async with send_lock:
                    await openai_ws.send(pkt)
            else:
                await openai_ws.send(pkt)
        except Exception:
            logger.exception("Failed sending conversation.item.create to OpenAI; keeping buffer")
            item_turn_in_flight = False
            return

        # Commit locally only after successful send
        if isinstance(audio_buf, bytearray):
            # clear the shared buffer
            del audio_buf[: len(local_audio)]
        else:
            # overwrite attribute if it was bytes
            setattr(tracker, "item_audio_buf", bytearray())
        tracker.on_committed()

        if not _can_create_response():
            # No response allowed yet (rate limit). Allow a new turn-end to retrigger later.
            item_turn_in_flight = False
            return
        response_in_flight = True
        logger.info("Creating response (%s)", reason)
        try:
            pkt2 = json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio", "text"],
                        "instructions": "Answer in English in 1-2 short sentences.",
                    },
                }
            )
            if send_lock is not None:
                async with send_lock:
                    await openai_ws.send(pkt2)
            else:
                await openai_ws.send(pkt2)
        except Exception:
            logger.exception("Failed sending response.create after item audio; allowing retry")
            response_in_flight = False
            item_turn_in_flight = False

    def _can_create_response() -> bool:
        nonlocal last_response_create_t
        now = time.monotonic()
        if (now - last_response_create_t) * 1000.0 < max(cfg.response_min_interval_ms, 0):
            return False
        last_response_create_t = now
        return True

    async def _maybe_create_response(reason: str) -> None:
        nonlocal response_in_flight
        # Absolute safety: never create a response unless OpenAI has ACKed a commit.
        nonlocal response_ready, response_ready_reason, commit_pending
        logger.debug(
            "_maybe_create_response(%s): response_ready=%s commit_pending=%s response_in_flight=%s",
            reason,
            response_ready,
            commit_pending,
            response_in_flight,
        )
        if commit_pending:
            logger.debug("Not creating response yet (%s): commit still pending", reason)
            return
        if not response_ready:
            logger.debug("Not creating response yet (%s): waiting for commit ACK", reason)
            return
        # Once we start a response, consume the readiness so we don't create duplicates.
        response_ready = False
        response_ready_reason = None
        if response_in_flight:
            return
        if not _can_create_response():
            return
        response_in_flight = True
        logger.info("Creating response (%s)", reason)
        try:
            pkt = json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio", "text"],
                        "instructions": "Answer in English in 1-2 short sentences.",
                    },
                }
            )
            if send_lock is not None:
                async with send_lock:
                    await openai_ws.send(pkt)
            else:
                await openai_ws.send(pkt)
        except Exception:
            logger.exception("Failed to send response.create; allowing retry")
            response_in_flight = False

    async def _maybe_commit(reason: str) -> None:
        nonlocal last_turn_evt_t, commit_pending, commit_reason, commit_sent_t
        now = time.monotonic()
        if (now - last_turn_evt_t) < 0.15:
            return
        last_turn_evt_t = now

        if commit_pending:
            return

        if tracker.appended_since_commit_bytes < min_commit_bytes:
            logger.debug(
                "Skipping commit (%s): appended=%d (<%d bytes)",
                reason,
                tracker.appended_since_commit_bytes,
                min_commit_bytes,
            )
            return

        commit_pending = True
        commit_reason = reason
        commit_sent_t = now
        logger.info(
            "Sending commit (%s): appended_since_commit_bytes=%d threshold=%d commits_sent=%d commits_acked=%d",
            reason,
            tracker.appended_since_commit_bytes,
            min_commit_bytes,
            tracker.commits_sent,
            tracker.commits_acked,
        )
        try:
            pkt = json.dumps({"type": "input_audio_buffer.commit"})
            if send_lock is not None:
                async with send_lock:
                    await openai_ws.send(pkt)
            else:
                await openai_ws.send(pkt)
        except Exception:
            logger.exception("Failed sending input_audio_buffer.commit to OpenAI; will retry")
            commit_pending = False
            commit_reason = None
            return

        # Only mark commit_sent after a successful send
        tracker.on_commit_sent()

    async def _maybe_commit_if_turn_pending() -> None:
        nonlocal pending_turn_end
        if pending_turn_end is None:
            return
        if commit_pending:
            return
        if tracker.appended_since_commit_bytes < min_commit_bytes:
            return

        reason = pending_turn_end
        pending_turn_end = None
        await _maybe_commit(reason)

    async def _playout_loop() -> None:
        step_s = contract.frame_ms / 1000.0
        next_t = time.monotonic() + step_s
        min_sleep = max(cfg.playout_sleep_granularity_ms, 0) / 1000.0

        frames_sent = 0
        underruns = 0
        last_stats_t = time.monotonic()
        last_stats_frames_sent = 0
        last_stats_underruns = 0
        last_stats_buf_len = 0

        if prebuffer_bytes > 0:
            while len(buf) < prebuffer_bytes:
                await asyncio.sleep(step_s)

        try:
            while True:
                now = time.monotonic()
                if now < next_t:
                    sleep_s = next_t - now
                    if min_sleep > 0 and sleep_s < min_sleep:
                        sleep_s = min_sleep
                    await asyncio.sleep(sleep_s)

                if time.monotonic() - next_t > step_s * 3:
                    next_t = time.monotonic() + step_s

                now2 = time.monotonic()
                lag_s = now2 - next_t
                if lag_s > max(cfg.playout_catchup_max_ms, 0) / 1000.0:
                    missed = int(lag_s / step_s)
                    if missed > 0:
                        next_t += missed * step_s
                        logger.debug("Playout catchup: missed_frames=%d lag_ms=%.1f", missed, lag_s * 1000.0)

                next_t += step_s

                # --- Adaptive drain ---
                # OpenAI delivers audio in large bursts (seconds of audio in ~100ms).
                # If we only drain 1 frame/tick, the buffer grows unbounded and the cap
                # in the receive path drops huge chunks (audible artifacts).
                # Instead: when buffer is above target, drain faster proportionally.
                # When buffer is at/below target, drain exactly 1 frame/tick (real-time).
                buf_len = len(buf)
                frames_to_send = 1
                if buf_len > target_buf_bytes and target_buf_bytes > 0:
                    # How many frames above target?
                    excess_frames = (buf_len - target_buf_bytes) // out_frame_bytes
                    # Scale drain: 1 base + proportional excess, capped by max_drain_frames
                    frames_to_send = min(max_drain_frames, 1 + excess_frames)
                elif buf_len > out_frame_bytes * 3 and target_buf_bytes == 0:
                    # No target set but buffer is growing; drain a bit faster
                    frames_to_send = min(max_drain_frames, buf_len // out_frame_bytes)

                for _ in range(frames_to_send):
                    if len(buf) >= out_frame_bytes:
                        frame = bytes(buf[:out_frame_bytes])
                        del buf[:out_frame_bytes]
                    else:
                        underruns += 1
                        continue

                    if cfg.fs_send_json_audio:
                        payload = fs_stream_audio_json(
                            frame,
                            contract,
                            sample_rate_override=openai_out_rate,
                            channels_override=openai_out_channels,
                        )
                    else:
                        payload = frame
                    await upstream_ws.send(payload)
                    frames_sent += 1

                now_stats = time.monotonic()
                if now_stats - last_stats_t >= 5.0:
                    dt = now_stats - last_stats_t
                    sent_delta = frames_sent - last_stats_frames_sent
                    underrun_delta = underruns - last_stats_underruns
                    buf_delta = len(buf) - last_stats_buf_len
                    logger.info(
                        "Playout stats: frames_sent=%d (+%d/%.1fs) buf_ms=%.1f (delta_bytes=%d) target_ms=%d underruns=%d (+%d) drops_capped_by_max=%s",
                        frames_sent,
                        sent_delta,
                        dt,
                        (len(buf) / out_frame_bytes) * contract.frame_ms if out_frame_bytes > 0 else 0.0,
                        buf_delta,
                        target_buf_ms,
                        underruns,
                        underrun_delta,
                        "yes" if max_buf_bytes > 0 else "no",
                    )
                    last_stats_t = now_stats
                    last_stats_frames_sent = frames_sent
                    last_stats_underruns = underruns
                    last_stats_buf_len = len(buf)
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
                    # Deltas can repeat/overlap; only print stable text at completion.
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
            ):
                t = _extract_text(evt)
                if t:
                    ai_text_buf.append(t)
                    logger.info("AI_TEXT: %s", "".join(ai_text_buf).strip())
                continue

            if evt_type in ("response.completed", "response.done"):
                # Clear text buffers at end of response
                if ai_text_buf:
                    logger.info("AI_TEXT_FINAL: %s", "".join(ai_text_buf).strip())
                ai_text_buf.clear()
                user_text_buf.clear()
                response_in_flight = False

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

                # OpenAI Realtime API outputs 24kHz mono PCM16.
                # Use config defaults if event omits sample_rate/channels.
                src_rate = int(getattr(cfg, "openai_output_sample_rate", cfg.fs_sample_rate))
                src_ch = 1
                if isinstance(evt.get("sample_rate"), int):
                    src_rate = int(evt["sample_rate"])
                if isinstance(evt.get("channels"), int):
                    src_ch = int(evt["channels"])

                # Do not resample here. We rely on `mod_audio_stream` (SpeexDSP) to resample on injection.
                # Ensure mono.
                if src_ch != 1:
                    try:
                        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
                        src_ch = 1
                    except Exception:
                        pcm = ensure_even_bytes(pcm)

                buf.extend(pcm)

                # NOTE: we no longer hard-cap (drop) in the receive path.
                # The playout loop drains aggressively when above target_buf_bytes,
                # which prevents the oscillating "burst→drop→underrun" cycle that
                # caused audible artifacts. The C++ mod's own inject_buffer has its
                # own overflow protection as a final safety net.
                if max_buf_bytes > 0 and len(buf) > max_buf_bytes * 2:
                    # Absolute safety: only if buffer somehow reaches 2x max (shouldn't
                    # happen with drain, but protects against pathological stalls).
                    overflow = len(buf) - max_buf_bytes
                    dropped = drop_oldest_frame_aligned(buf, overflow, out_frame_bytes)
                    logger.warning(
                        "Playout emergency cap: dropped_bytes=%d buf_ms_now=%.1f max_buf_ms=%d",
                        dropped,
                        (len(buf) / out_frame_bytes) * contract.frame_ms if out_frame_bytes > 0 else 0.0,
                        cfg.playout_max_buffer_ms,
                    )

            elif evt_type in (
                "input_audio_buffer.speech_stopped",
                "input_audio_buffer.speech_end",
                "input_audio_buffer.speech_end_detected",
                "input_audio_buffer.vad_stop",
            ):
                logger.info(
                    "Turn end (%s): appended_since_commit_bytes=%d threshold=%d",
                    evt_type,
                    tracker.appended_since_commit_bytes,
                    min_commit_bytes,
                )
                pending_turn_end = f"server_vad:{evt_type}"
                if tracker.appended_since_commit_bytes < min_commit_bytes:
                    logger.debug(
                        "Turn-end pending (not enough audio yet): appended=%d threshold=%d",
                        tracker.appended_since_commit_bytes,
                        min_commit_bytes,
                    )
                if getattr(cfg, "openai_input_mode", "buffer") == "item":
                    await _send_item_audio_and_respond(pending_turn_end)
                    pending_turn_end = None
                else:
                    await _maybe_commit_if_turn_pending()

            elif evt_type == "input_audio_buffer.committed":
                # Only now is it safe to create a response.
                tracker.on_commit_acked()
                tracker.on_committed()
                reason = commit_reason or "commit_ack"
                commit_pending = False
                commit_reason = None
                response_ready = True
                response_ready_reason = reason
                await _maybe_create_response(reason)

            # After any event, if we have a pending turn-end and enough audio has since arrived,
            # commit it immediately.
            await _maybe_commit_if_turn_pending()

            if evt_type in ("response.completed", "response.done"):
                response_in_flight = False
                item_turn_in_flight = False

            elif evt_type in ("error", "session.created", "session.updated"):
                if evt_type == "error":
                    # Log full error payload so we can debug model/key/validation issues.
                    # Do not log secrets; Realtime errors should not contain API keys.
                    logger.error("OpenAI error: %s", json.dumps(evt, ensure_ascii=False))
                else:
                    logger.info("OpenAI event: %s", evt_type)
                if evt_type == "error" and commit_pending:
                    err = evt.get("error")
                    code = err.get("code") if isinstance(err, dict) else None
                    if code in ("input_audio_buffer_commit_empty", "input_audio_buffer_commit_too_small"):
                        # A commit failed; allow future turns to try again.
                        commit_pending = False
                        commit_reason = None
                        # IMPORTANT: don't call tracker.on_committed() here.
                        # OpenAI is telling us it had 0ms, but locally we may have appended bytes.
                        # Keep the local counter so we can retry commit once we have enough data.
                        logger.warning(
                            "Commit rejected by OpenAI (%s). Keeping local appended_since_commit_bytes=%d; commits_sent=%d commits_acked=%d",
                            code,
                            tracker.appended_since_commit_bytes,
                            tracker.commits_sent,
                            tracker.commits_acked,
                        )

    finally:
        playout_task.cancel()
        try:
            await playout_task
        except asyncio.CancelledError:
            pass


async def handle_call(cfg: BridgeConfig, upstream_ws: websockets.WebSocketServerProtocol) -> None:
    peer = getattr(upstream_ws, "remote_address", None)
    logger.info("Call connected: %s", peer)

    ssl_ctx = build_ssl_context(cfg.wss_pem, cfg.openai_wss_insecure)
    openai_ws = await connect_openai_realtime(
        api_key=cfg.openai_api_key,
        model=cfg.model,
        voice=cfg.voice,
        ssl_ctx=ssl_ctx,
    )

    if cfg.fs_send_json_audio and cfg.fs_send_json_handshake:
        try:
            contract = FsAudioContract(cfg.fs_out_sample_rate, cfg.fs_channels, cfg.fs_frame_ms)
            await upstream_ws.send(fs_handshake_json(contract))
        except Exception as e:
            logger.warning("FreeSWITCH: failed to send handshake: %s", e)

    tracker = InputAudioTracker()
    # Single lock to serialize sends to the OpenAI websocket. This prevents
    # concurrent send races and keeps local bookkeeping accurate.
    send_lock = asyncio.Lock()
    to_openai = asyncio.create_task(
        pump_freeswitch_to_openai(upstream_ws, openai_ws, cfg, tracker, send_lock=send_lock)
    )
    to_fs = asyncio.create_task(
        pump_openai_to_freeswitch(openai_ws, upstream_ws, cfg, tracker, send_lock=send_lock)
    )

    done, pending = await asyncio.wait({to_openai, to_fs}, return_when=asyncio.FIRST_EXCEPTION)
    for t in pending:
        t.cancel()

    # Close OpenAI websocket cleanly.
    try:
        await openai_ws.close()
    except Exception:
        logger.debug("openai_ws.close() failed or already closed")

    # Also attempt to close the incoming upstream websocket (FreeSWITCH side).
    try:
        await upstream_ws.close()
    except Exception:
        logger.debug("upstream_ws.close() failed or already closed")


async def run_server(cfg: BridgeConfig) -> None:
    async def _handler(ws: websockets.WebSocketServerProtocol):
        try:
            await handle_call(cfg, ws)
        except websockets.ConnectionClosed:
            pass
        except Exception:
            logger.exception("Bridge error")

    logger.info("Listening for FreeSWITCH on ws://%s:%d", cfg.host, cfg.port)
    async with websockets.serve(_handler, cfg.host, cfg.port, max_size=None):
        await asyncio.Future()
