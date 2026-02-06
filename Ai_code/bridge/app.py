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
from .resample import PCM16Resampler

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


def _ratecv_pcm16_mono(pcm: bytes, src_rate: int, dst_rate: int, state: Any) -> tuple[bytes, Any]:
    if not pcm or src_rate == dst_rate:
        return pcm, state
    converted, new_state = audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, state)
    return ensure_even_bytes(converted), new_state


async def pump_freeswitch_to_openai(
    upstream_ws: websockets.WebSocketServerProtocol,
    openai_ws: websockets.WebSocketClientProtocol,
    cfg: BridgeConfig,
    tracker: InputAudioTracker,
) -> None:
    bytes_in = 0
    frames_in = 0
    started = False

    # OpenAI side input settings (we may resample FS audio before append)
    openai_in_rate = int(getattr(cfg, "openai_input_sample_rate", cfg.fs_sample_rate))
    openai_resample = bool(getattr(cfg, "openai_resample_input", False))
    to_openai_ratecv_state = None

    expected_frame_bytes = frame_bytes(cfg.fs_sample_rate, cfg.fs_channels, cfg.fs_frame_ms)

    inbuf = bytearray()

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

                # Normalize to mono and (optionally) resample before sending to OpenAI.
                out_pcm = frame
                if cfg.fs_channels != 1:
                    try:
                        out_pcm = audioop.tomono(out_pcm, 2, 0.5, 0.5)
                    except Exception:
                        pass

                if openai_resample and openai_in_rate != cfg.fs_sample_rate:
                    try:
                        out_pcm, to_openai_ratecv_state = _ratecv_pcm16_mono(
                            out_pcm,
                            cfg.fs_sample_rate,
                            openai_in_rate,
                            to_openai_ratecv_state,
                        )
                    except Exception:
                        out_pcm = ensure_even_bytes(out_pcm)
                else:
                    out_pcm = ensure_even_bytes(out_pcm)

                try:
                    await openai_ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(out_pcm).decode("ascii"),
                            }
                        )
                    )
                except Exception:
                    logger.exception("Failed sending input_audio_buffer.append to OpenAI")
                    raise

                tracker.on_appended(len(out_pcm))

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
                        openai_resample,
                    )

            if frames_in and frames_in % 50 == 0:
                logger.debug("FreeSWITCH->OpenAI: frames=%d bytes=%d tail=%d", frames_in, bytes_in, len(inbuf))

            # IMPORTANT:
            # Do NOT commit from this pump.
            # Commits must be synchronized with OpenAI's VAD turn-end events (received in the
            # OpenAI->FS pump), otherwise OpenAI can reject commits as empty (0.00ms) even while
            # audio is flowing.
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
) -> None:
    contract = FsAudioContract(
        sample_rate=int(cfg.fs_out_sample_rate),
        channels=int(cfg.fs_channels),
        frame_ms=int(cfg.fs_frame_ms),
    )

    out_frame_bytes = frame_bytes(contract.sample_rate, contract.channels, contract.frame_ms)

    buf = bytearray()

    max_buf_bytes = ceil_to_frame(
        frame_bytes(cfg.fs_sample_rate, cfg.fs_channels, max(cfg.playout_max_buffer_ms, cfg.fs_frame_ms)),
        out_frame_bytes,
    )
    prebuffer_bytes = ceil_to_frame(
        frame_bytes(cfg.fs_sample_rate, cfg.fs_channels, max(cfg.playout_prebuffer_ms, 0)),
        out_frame_bytes,
    )

    # Output path resamplers (stateful across deltas)
    openai_to_fs_resampler = None
    fs_to_out_resampler = None

    # OpenAI requires ~>=100ms buffered before commit.
    # IMPORTANT: threshold must be computed in the same format/rate we appended.
    openai_in_rate = int(getattr(cfg, "openai_input_sample_rate", cfg.fs_sample_rate))
    openai_resample = bool(getattr(cfg, "openai_resample_input", False))
    min_commit_ms = 100
    min_commit_bytes = frame_bytes(openai_in_rate if openai_resample else cfg.fs_sample_rate, 1, min_commit_ms)
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

    # If we get a turn-end but don't yet have enough audio appended, remember it
    # and commit as soon as we cross the threshold. This avoids commit_empty.
    pending_turn_end: Optional[str] = None

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
        await openai_ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["audio", "text"],
                        "instructions": "Answer in English in 1-2 short sentences.",
                    },
                }
            )
        )

    async def _maybe_commit(reason: str) -> None:
        nonlocal last_turn_evt_t, commit_pending, commit_reason, commit_sent_t
        now = time.monotonic()
        if (now - last_turn_evt_t) < 0.15:
            return
        last_turn_evt_t = now

        if commit_pending:
            return

        # IMPORTANT: do not commit if we haven't appended enough audio.
        # This prevents input_audio_buffer_commit_empty.
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
        tracker.on_commit_sent()
        logger.info(
            "Sending commit (%s): appended_since_commit_bytes=%d threshold=%d commits_sent=%d commits_acked=%d",
            reason,
            tracker.appended_since_commit_bytes,
            min_commit_bytes,
            tracker.commits_sent,
            tracker.commits_acked,
        )
        await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

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

                now2 = time.monotonic()
                lag_s = now2 - next_t
                if lag_s > max(cfg.playout_catchup_max_ms, 0) / 1000.0:
                    missed = int(lag_s / step_s)
                    if missed > 0:
                        next_t += missed * step_s
                        logger.debug("Playout catchup: missed_frames=%d lag_ms=%.1f", missed, lag_s * 1000.0)

                next_t += step_s

                if len(buf) >= out_frame_bytes:
                    frame = bytes(buf[:out_frame_bytes])
                    del buf[:out_frame_bytes]
                    payload = fs_stream_audio_json(frame, contract) if cfg.fs_send_json_audio else frame
                    await upstream_ws.send(payload)
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

            # ---- Text events (telemetry): user transcription + assistant text ----
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

                # Default OpenAI output is typically 16k mono PCM16.
                # Some events omit sample_rate/channels; use config defaults in that case.
                src_rate = int(getattr(cfg, "openai_output_sample_rate", cfg.fs_sample_rate))
                src_ch = 1
                if isinstance(evt.get("sample_rate"), int):
                    src_rate = int(evt["sample_rate"])
                if isinstance(evt.get("channels"), int):
                    src_ch = int(evt["channels"])

                # OpenAI -> FS (mono @ FS_SAMPLE_RATE)
                if openai_to_fs_resampler is None or openai_to_fs_resampler.src_rate != src_rate or openai_to_fs_resampler.src_channels != src_ch:
                    openai_to_fs_resampler = PCM16Resampler(
                        src_rate=src_rate,
                        dst_rate=cfg.fs_sample_rate,
                        src_channels=src_ch,
                        dst_channels=1,
                    )
                pcm = openai_to_fs_resampler.convert(pcm)

                # FS internal -> FS output rate (often same, but keep correct)
                if cfg.fs_out_sample_rate != cfg.fs_sample_rate:
                    if fs_to_out_resampler is None:
                        fs_to_out_resampler = PCM16Resampler(
                            src_rate=cfg.fs_sample_rate,
                            dst_rate=cfg.fs_out_sample_rate,
                            src_channels=1,
                            dst_channels=1,
                        )
                    pcm = fs_to_out_resampler.convert(pcm)

                buf.extend(pcm)

                if max_buf_bytes > 0 and len(buf) > max_buf_bytes:
                    overflow = len(buf) - max_buf_bytes
                    dropped = drop_oldest_frame_aligned(buf, overflow, out_frame_bytes)
                    logger.debug("Playout buffer capped: dropped_bytes=%d", dropped)

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
    to_openai = asyncio.create_task(pump_freeswitch_to_openai(upstream_ws, openai_ws, cfg, tracker))
    to_fs = asyncio.create_task(pump_openai_to_freeswitch(openai_ws, upstream_ws, cfg, tracker))

    done, pending = await asyncio.wait({to_openai, to_fs}, return_when=asyncio.FIRST_EXCEPTION)
    for t in pending:
        t.cancel()

    await openai_ws.close()


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
