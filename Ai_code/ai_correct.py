from __future__ import annotations

"""
Minimal, single-file OpenAI Realtime <-> FreeSWITCH bridge with fixes for
playout alignment. This version focuses on the playback-side bug that caused
silence on the FreeSWITCH caller: buffer caps were trimming arbitrary byte
counts, destroying 20 ms frame alignment expected by mod_audio_stream.

Key fixes:
- Drop buffered audio only in whole 20 ms frames (frame-aligned trimming).
- Keep modest prebuffer (configurable) to smooth bursty OpenAI audio.
- Defensive parsing and logging cleanups.
"""

import asyncio
import base64
import json
import os
import ssl
import time
from dataclasses import dataclass
from typing import Any, Optional

import audioop

import websockets


def _build_ssl_context() -> ssl.SSLContext:
    """Create an SSL context for outbound OpenAI WSS connections.

    Fixes macOS/Python setups where the default CA store isn't found, causing:
      [SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate

    Priority:
    1) Use an explicit CA bundle provided via WSS_PEM (path to PEM file)
    2) Fall back to certifi (if installed)
    3) Fall back to system defaults

    Dev escape hatch:
      OPENAI_WSS_INSECURE=1 disables verification (not recommended).
    """

    if os.getenv("OPENAI_WSS_INSECURE", "0") == "1":
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        log("WARNING: OPENAI_WSS_INSECURE=1 -> TLS certificate verification disabled")
        return ctx

    ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)

    # If user provides a PEM file (often containing one-or-more CA certs), use it.
    # Accept relative paths (e.g. ./wss.pem) and resolve to absolute.
    pem_path = os.getenv("WSS_PEM", "").strip()
    if pem_path and not os.path.isabs(pem_path):
        pem_path = os.path.abspath(pem_path)
    if pem_path:
        try:
            # Safety: WSS_PEM must be a CA bundle, NOT a private key.
            with open(pem_path, "rb") as f:
                head = f.read(4096)
            if b"PRIVATE KEY" in head:
                raise ValueError(
                    "WSS_PEM appears to contain a private key; provide a CA bundle PEM (root/intermediate certs) only"
                )

            ctx.load_verify_locations(cafile=pem_path)
            log(f"TLS: using CA bundle from WSS_PEM={pem_path}")
            return ctx
        except Exception as e:
            log(f"TLS: failed to load WSS_PEM={pem_path}: {e} (will try certifi/system)")

    # Fall back to certifi if available (common fix on macOS).
    try:
        import certifi  # type: ignore

        cafile = certifi.where()
        ctx.load_verify_locations(cafile=cafile)
        log(f"TLS: using certifi CA bundle: {cafile}")
    except Exception:
        # Keep system defaults.
        pass

    return ctx


@dataclass
class InputAudioTracker:
    """Shared state for input-audio commit gating.

    The FreeSWITCH->OpenAI pump appends audio; the OpenAI->FreeSWITCH pump
    receives VAD/transcription events and decides when to commit+respond.
    This shared counter prevents "commit_empty" errors and avoids creating
    responses when we haven't actually appended enough audio.
    """

    appended_since_commit_bytes: int = 0

    def on_appended(self, n: int) -> None:
        if n > 0:
            self.appended_since_commit_bytes += n

    def on_committed(self) -> None:
        self.appended_since_commit_bytes = 0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_throttled(key: str, msg: str, every_s: float = 2.0) -> None:
    """Log at most once every `every_s` seconds per key."""
    now = time.monotonic()
    last = getattr(log_throttled, "_last", None)
    if last is None:
        last = {}
        setattr(log_throttled, "_last", last)

    prev = last.get(key, 0.0)
    if now - prev >= every_s:
        last[key] = now
        log(msg)


@dataclass
class BridgeConfig:
    ivr_host: str
    ivr_port: int
    openai_api_key: str
    model: str
    voice: str
    force_commit_ms: int
    force_response_on_commit: bool
    # Playback pacing (OpenAI -> FreeSWITCH)
    fs_frame_ms: int
    fs_sample_rate: int
    fs_channels: int
    playout_prebuffer_ms: int
    playout_max_buffer_ms: int
    send_test_tone_on_connect: bool
    # Response control (server_vad with create_response=False)
    response_min_interval_ms: int
    # Playout tuning
    playout_catchup_max_ms: int
    playout_sleep_granularity_ms: int
    # FreeSWITCH outbound framing
    fs_send_json_audio: bool
    fs_send_json_handshake: bool
    # Optional downsample before sending to FreeSWITCH (e.g. 16000 -> 8000)
    fs_out_sample_rate: int


def _frame_bytes(sample_rate: int, channels: int, frame_ms: int) -> int:
    bytes_per_sample = 2  # PCM16
    samples = int(sample_rate * frame_ms / 1000)
    return samples * channels * bytes_per_sample


def _ratecv_pcm16_mono(
    pcm: bytes,
    src_rate: int,
    dst_rate: int,
    state: Any,
) -> tuple[bytes, Any]:
    """Stateful mono PCM16 resample helper.

    Returns: (converted_pcm, new_state)
    """
    if not pcm or src_rate == dst_rate:
        return pcm, state

    converted, new_state = audioop.ratecv(pcm, 2, 1, src_rate, dst_rate, state)
    # PCM16 should always be even bytes.
    if len(converted) % 2:
        converted = converted[:-1]
    return converted, new_state


def _ceil_to_frame(n: int, frame: int) -> int:
    if frame <= 0:
        return n
    return ((n + frame - 1) // frame) * frame


def _downmix_to_mono_pcm16(pcm: bytes, channels: int) -> bytes:
    """Downmix little-endian PCM16 with N channels to mono PCM16."""
    if channels <= 1:
        return pcm
    # audioop.tomono expects sample-width bytes
    return audioop.tomono(pcm, 2, 1.0 / channels, 1.0 / channels)


def _ensure_pcm16_mono_16k(
    pcm: bytes,
    src_rate: int,
    src_channels: int,
    dst_rate: int = 16000,
    dst_channels: int = 1,
) -> bytes:
    """Convert PCM16 audio to mono/16k.

    Notes:
    - Uses stdlib `audioop` (no extra deps).
    - Keeps internal resample state outside this helper; caller should provide
      continuous audio chunks (OK for OpenAI deltas).
    """
    if not pcm:
        return b""

    # Downmix first (cheaper than resampling multi-channel).
    if src_channels != dst_channels:
        if dst_channels != 1:
            raise ValueError("Only mono output supported")
        pcm = _downmix_to_mono_pcm16(pcm, src_channels)
        src_channels = 1

    if src_rate == dst_rate:
        return pcm

    # Stateless resample; for best quality use stateful conversion in caller.
    converted, _ = audioop.ratecv(pcm, 2, src_channels, src_rate, dst_rate, None)
    return converted


def _b64encode_audio(pcm_bytes: bytes) -> str:
    return base64.b64encode(pcm_bytes).decode("ascii")


def _fs_streamaudio_json(frame: bytes, sample_rate: int, channels: int) -> str:
    """Build JSON exactly as mod_audio_stream/audio_streamer_glue.cpp expects."""
    return _ws_text(
        json.dumps(
            {
                "type": "streamAudio",
                "data": {
                    "audioDataType": "raw",
                    "audioData": _b64encode_audio(frame),
                    "sampleRate": int(sample_rate),
                    "channels": int(channels),
                },
            }
        )
    )


def _ws_text(payload: str) -> str:
    """Marker helper for a websocket text frame payload."""
    return payload


def _safe_json_loads(s: str) -> Optional[dict[str, Any]]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _print_text_delta(text: str) -> None:
    print(text, end="", flush=True)


def _extract_any_text_delta(evt: dict[str, Any]) -> Optional[str]:
    if isinstance(evt.get("delta"), str) and evt.get("delta"):
        return evt["delta"]
    if isinstance(evt.get("text"), str) and evt.get("text"):
        return evt["text"]
    return None


def _event_type(evt: dict[str, Any]) -> str:
    t = evt.get("type")
    return t if isinstance(t, str) else ""


def _extract_any_text_list(evt: dict[str, Any]) -> list[str]:
    out: list[str] = []

    def walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            # Heuristic: skip obvious base64 blobs
            if len(x) > 300 and all(c.isalnum() or c in "+/=\n" for c in x[:50]):
                return
            out.append(x)
            return
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("audio", "data", "pcm", "delta") and isinstance(v, str) and len(v) > 500:
                    continue
                walk(v)
            return
        if isinstance(x, list):
            for item in x:
                walk(item)

    for key in ("delta", "text", "transcript", "content"):
        v = evt.get(key)
        if isinstance(v, str) and v:
            out.append(v)

    for key in ("response", "item", "output", "outputs"):
        if key in evt:
            walk(evt[key])

    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


async def openai_realtime_session(cfg: BridgeConfig) -> websockets.WebSocketClientProtocol:
    url = f"wss://api.openai.com/v1/realtime?model={cfg.model}"
    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    ssl_ctx = _build_ssl_context()

    async def _connect_with_kwargs(**kwargs):
        return await websockets.connect(url, max_size=None, ssl=ssl_ctx, **kwargs)

    ws = None
    for key in ("extra_headers", "headers", "additional_headers"):
        try:
            ws = await _connect_with_kwargs(**{key: headers})
            break
        except TypeError:
            pass
    if ws is None:
        for key in ("extra_headers", "headers", "additional_headers"):
            try:
                ws = await _connect_with_kwargs(**{key: list(headers.items())})
                break
            except TypeError:
                pass

    if ws is None:
        raise RuntimeError(
            "websockets package too old/incompatible to pass headers; upgrade websockets."
        )

    session_update = {
        "type": "session.update",
        "session": {
            "voice": cfg.voice,
            "instructions": (
                "You are a helpful AI IVR agent. ALWAYS respond in English only. "
                "Keep every response short (1-2 sentences). "
                "Use simple words. Ask at most one question at a time."
            ),
            # Realtime enforces a minimum temperature; keep it low-ish but valid.
            "temperature": 0.6,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
            # We'll create responses ourselves (avoids repeated/auto responses).
            "turn_detection": {"type": "server_vad", "create_response": False},
        },
    }

    await ws.send(json.dumps(session_update))

    greet = {
        "type": "response.create",
        "response": {
            "modalities": ["audio", "text"],
            "instructions": "In English: greet the caller in one short sentence and ask how you can help.",
        },
    }
    await ws.send(json.dumps(greet))

    return ws


async def pump_freeswitch_to_openai(
    upstream_ws: websockets.WebSocketServerProtocol,
    openai_ws: websockets.WebSocketClientProtocol,
    cfg: BridgeConfig,
    tracker: InputAudioTracker,
) -> None:
    last_commit = time.monotonic()
    bytes_in = 0
    frames_in = 0
    started = False

    # OpenAI rejects commits when the input buffer has < ~100ms of audio.
    # Use the shared tracker to account for audio appended in this coroutine.
    min_commit_ms = 100
    min_commit_bytes = _frame_bytes(cfg.fs_sample_rate, cfg.fs_channels, min_commit_ms)

    expected_frame_bytes = _frame_bytes(cfg.fs_sample_rate, cfg.fs_channels, cfg.fs_frame_ms)
    if expected_frame_bytes <= 0:
        raise RuntimeError("Invalid FreeSWITCH frame sizing")

    # mod_audio_stream is *supposed* to send exact frames, but in practice WS can
    # deliver coalesced or fragmented messages. Reassemble to guarantee contract.
    inbuf = bytearray()

    async for message in upstream_ws:
        if isinstance(message, (bytes, bytearray)):
            if not started:
                started = True
                log(f"FreeSWITCH: first PCM frame received ({len(message)} bytes)")

            inbuf.extend(bytes(message))

            # Emit exactly one or more full 20ms frames.
            while len(inbuf) >= expected_frame_bytes:
                frame = bytes(inbuf[:expected_frame_bytes])
                del inbuf[:expected_frame_bytes]

                bytes_in += len(frame)
                frames_in += 1
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": _b64encode_audio(frame),
                }
                await openai_ws.send(json.dumps(event))
                tracker.on_appended(len(frame))

            # If the upstream ever sends odd byte counts, we'll keep the tail
            # until the next message. Log occasionally for visibility.
            if len(inbuf) and (frames_in % 50 == 0):
                log_throttled(
                    "fs_in_tail",
                    f"FreeSWITCH inbound reassembly tail={len(inbuf)} bytes (waiting for full frame)",
                    every_s=2.0,
                )

            # Lightweight rate stats to confirm we're continuously receiving audio from FS
            if frames_in % 50 == 0:
                log_throttled(
                    "fs_in",
                    f"FreeSWITCH->OpenAI: frames={frames_in} bytes={bytes_in}",
                    every_s=2.0,
                )

            if cfg.force_commit_ms > 0:
                now = time.monotonic()
                if (now - last_commit) * 1000 >= cfg.force_commit_ms:
                    last_commit = now
                    if tracker.appended_since_commit_bytes >= min_commit_bytes:
                        await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                        # IMPORTANT: don't reset the tracker until the server ACKs the commit.
                    else:
                        log_throttled(
                            "skip_empty_commit",
                            (
                                "Skipping input_audio_buffer.commit: only "
                                f"{tracker.appended_since_commit_bytes} bytes appended since last commit "
                                f"(<{min_commit_bytes} bytes / {min_commit_ms}ms)"
                            ),
                            every_s=2.0,
                        )
                    # If server-side VAD auto-response is disabled, you may choose to create a
                    # response on commit. In production you'll normally prefer VAD end-of-turn.
                    if cfg.force_response_on_commit:
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
        else:
            text = str(message)
            parsed = _safe_json_loads(text)
            if parsed is not None:
                log(f"FreeSWITCH text event: {parsed}")
            else:
                log(f"FreeSWITCH text message: {text}")


async def pump_openai_to_freeswitch(
    openai_ws: websockets.WebSocketClientProtocol,
    upstream_ws: websockets.WebSocketServerProtocol,
    cfg: BridgeConfig,
    tracker: InputAudioTracker,
) -> None:
    audio_frames = 0
    # Terminal logging: keep USER and AI text separated and readable.
    # By default we print full lines when an utterance/response completes.
    user_text_buf: list[str] = []
    ai_text_buf: list[str] = []
    stream_text_deltas = os.getenv("STREAM_TEXT_DELTAS", "0") == "1"

    bytes_from_openai = 0
    bytes_sent_to_fs = 0
    frames_sent_to_fs = 0
    first_playout_sent = False

    # Target playout format for FreeSWITCH (can be downsampled vs internal processing rate).
    fs_out_rate = int(cfg.fs_out_sample_rate or cfg.fs_sample_rate)
    # Your mod_audio_stream injection path is safest with mono replacement.
    # Force mono when using JSON injection (WS->FS playback).
    if cfg.fs_send_json_audio and cfg.fs_channels != 1:
        log("FreeSWITCH outbound: forcing FS_CHANNELS=1 for JSON injection")
        cfg.fs_channels = 1

    frame_bytes = _frame_bytes(fs_out_rate, cfg.fs_channels, cfg.fs_frame_ms)
    if frame_bytes <= 0:
        raise RuntimeError("Invalid frame sizing for playout")

    if cfg.fs_frame_ms != 20 or cfg.fs_channels != 1:
        log(
            "WARNING: This bridge is tuned for PCM16/mono/16k/20ms. "
            f"Configured: {_fmt_audio(cfg.fs_sample_rate, cfg.fs_channels, cfg.fs_frame_ms)} "
            f"frame_bytes={frame_bytes}"
        )

    if fs_out_rate not in (8000, 16000):
        log(
            "WARNING: fs_out_sample_rate is unusual; mod_audio_stream typically expects 8000 or 16000. "
            f"Configured fs_out_sample_rate={fs_out_rate}"
        )

    buf = bytearray()

    def _fs_audio_payload(frame: bytes) -> bytes | str:
        """Build a FreeSWITCH payload for outbound playback.

        Required pipeline (when JSON mode enabled):
          PCM16 (raw audio frames)
            -> (optional) resample 16k -> 8k
            -> Base64 encode
            -> JSON wrapper (streamAudio)
            -> WebSocket -> mod_audio_stream

        Some mod_audio_stream builds still accept raw binary PCM frames; keep that as an option.
        """
        if not cfg.fs_send_json_audio:
            return frame
        return _fs_streamaudio_json(frame, sample_rate=fs_out_rate, channels=cfg.fs_channels)

    resample_state = None  # OpenAI delta rate -> cfg.fs_sample_rate
    fs_out_resample_state = None  # cfg.fs_sample_rate -> fs_out_rate
    max_buf_bytes = _ceil_to_frame(
        _frame_bytes(
            cfg.fs_sample_rate,
            cfg.fs_channels,
            max(cfg.playout_max_buffer_ms, cfg.fs_frame_ms),
        ),
        frame_bytes,
    )

    underruns = 0
    prebuffer_bytes = _ceil_to_frame(
        _frame_bytes(
            cfg.fs_sample_rate,
            cfg.fs_channels,
            max(cfg.playout_prebuffer_ms, 0),
        ),
        frame_bytes,
    )

    response_in_flight = False
    last_response_create_t = 0.0

    min_commit_ms = 100
    min_commit_bytes = _frame_bytes(cfg.fs_sample_rate, cfg.fs_channels, min_commit_ms)
    last_turn_evt_t = 0.0

    commit_pending = False
    commit_reason: Optional[str] = None

    debug_evt_types = os.getenv("DEBUG_OPENAI_EVENTS", "0") == "1"
    debug_text = os.getenv("DEBUG_TEXT_EVENTS", "0") == "1"

    def _can_create_response() -> bool:
        nonlocal last_response_create_t
        now = time.monotonic()
        if (now - last_response_create_t) * 1000.0 < max(cfg.response_min_interval_ms, 0):
            return False
        last_response_create_t = now
        return True

    async def _maybe_create_response(reason: str) -> None:
        nonlocal response_in_flight
        if response_in_flight:
            log_throttled(
                "resp_skip_inflight",
                f"Skipping response.create ({reason}): response already in flight",
                every_s=1.0,
            )
            return
        if not _can_create_response():
            log_throttled(
                "resp_skip_rate",
                f"Skipping response.create ({reason}): rate-limited",
                every_s=1.0,
            )
            return

        response_in_flight = True
        log(f"Creating response ({reason})")
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
        """Commit buffered input audio and request a response, but only if buffer isn't empty.

        OpenAI errors if we commit with <~100ms of audio in the input buffer.
        """
        nonlocal last_turn_evt_t, commit_pending, commit_reason
        now = time.monotonic()

        # De-bounce rapid repeated VAD events (some servers emit multiple stop signals).
        if (now - last_turn_evt_t) < 0.15:
            log_throttled(
                "turn_debounce",
                f"Ignoring duplicate turn event ({reason})",
                every_s=1.0,
            )
            return
        last_turn_evt_t = now

        if commit_pending:
            log_throttled(
                "skip_commit_pending",
                f"Skipping commit ({reason}): previous commit still pending",
                every_s=1.0,
            )
            return

        # If a response is currently in flight, don't start another turn/commit.
        # This prevents server errors like conversation_already_has_active_response.
        if response_in_flight:
            log_throttled(
                "skip_commit_resp_inflight",
                f"Skipping commit ({reason}): response already in flight",
                every_s=1.0,
            )
            return

        if tracker.appended_since_commit_bytes < min_commit_bytes:
            log_throttled(
                "skip_commit_empty",
                (
                    f"Skipping commit ({reason}): only "
                    f"{tracker.appended_since_commit_bytes} bytes appended since last commit "
                    f"(<{min_commit_bytes} bytes / {min_commit_ms}ms)"
                ),
                every_s=1.0,
            )
            return

        commit_pending = True
        commit_reason = reason
        await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        # Don't reset the tracker yet; only reset after commit ACK from server.

    async def _playout_loop() -> None:
        nonlocal underruns, first_playout_sent, bytes_sent_to_fs, frames_sent_to_fs
        # Deterministic pacing: schedule on a monotonic clock and correct drift.
        step_s = cfg.fs_frame_ms / 1000.0
        next_t = time.monotonic() + step_s

        # Avoid busy waiting and reduce timing jitter on macOS.
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

                # Drift correction / catchup: if we're too far behind, jump forward.
                now2 = time.monotonic()
                lag_s = now2 - next_t
                if lag_s > max(cfg.playout_catchup_max_ms, 0) / 1000.0:
                    missed = int(lag_s / step_s)
                    if missed > 0:
                        next_t += missed * step_s
                        log_throttled(
                            "playout_catchup",
                            f"Playout catch-up: lag_ms={lag_s*1000:.1f} missed_frames={missed}",
                            every_s=2.0,
                        )

                next_t += step_s

                if len(buf) >= frame_bytes:
                    frame = bytes(buf[:frame_bytes])
                    del buf[:frame_bytes]

                    if len(frame) != frame_bytes:
                        log(f"BUG: frame size {len(frame)} != expected {frame_bytes}; dropping")
                    else:
                        payload = _fs_audio_payload(frame)
                        await upstream_ws.send(payload)
                        if not first_playout_sent:
                            first_playout_sent = True
                            if isinstance(payload, str):
                                log(
                                    "OpenAI->FreeSWITCH: first playout frame sent (JSON/base64) "
                                    f"pcm_bytes={frame_bytes}"
                                )
                            else:
                                log(f"OpenAI->FreeSWITCH: first playout frame sent ({frame_bytes} bytes)")
                        frames_sent_to_fs += 1
                        bytes_sent_to_fs += frame_bytes
                else:
                    underruns += 1
                    if underruns == 1 or underruns % 200 == 0:
                        log(f"Playout underrun (buffer={len(buf)} bytes, count={underruns})")

                if frames_sent_to_fs and frames_sent_to_fs % 50 == 0:
                    log_throttled(
                        "fs_out",
                        f"OpenAI->FreeSWITCH: frames={frames_sent_to_fs} bytes={bytes_sent_to_fs} buffer={len(buf)}",
                        every_s=2.0,
                    )
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

            evt_type = _event_type(evt)
            if debug_evt_types and evt_type:
                log_throttled("openai_evt_type", f"OpenAI evt_type={evt_type}", every_s=0.5)

            # ---- Audio (OpenAI -> FreeSWITCH) ----
            # Realtime event names have varied over time. Treat these as synonyms.
            is_audio_evt = evt_type in (
                "response.audio.delta",
                "response.output_audio.delta",
                "response.audio",
                "response.output_audio",
                "response.audio_chunk",
                "response.output_audio_chunk",
            )

            if is_audio_evt:
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

                # Defensive: PCM16 must be even bytes.
                if len(pcm) % 2:
                    log_throttled(
                        "openai_odd_pcm",
                        f"OpenAI sent odd PCM byte count={len(pcm)}; trimming last byte to keep PCM16 alignment",
                        every_s=2.0,
                    )
                    pcm = pcm[:-1]

                # We treat OpenAI audio deltas as "src" and convert to our internal FS rate first.
                src_rate = cfg.fs_sample_rate
                src_ch = cfg.fs_channels

                if isinstance(evt.get("sample_rate"), int):
                    src_rate = int(evt["sample_rate"])
                if isinstance(evt.get("channels"), int):
                    src_ch = int(evt["channels"])

                if src_ch != 1:
                    pcm = _downmix_to_mono_pcm16(pcm, src_ch)
                    src_ch = 1

                if src_rate != cfg.fs_sample_rate:
                    pcm, resample_state = _ratecv_pcm16_mono(
                        pcm,
                        src_rate,
                        cfg.fs_sample_rate,
                        resample_state,
                    )

                # Optional downsample stage before playout (e.g., 16k -> 8k) per your required pipeline.
                if fs_out_rate != cfg.fs_sample_rate:
                    pcm, fs_out_resample_state = _ratecv_pcm16_mono(
                        pcm,
                        cfg.fs_sample_rate,
                        fs_out_rate,
                        fs_out_resample_state,
                    )

                buf.extend(pcm)
                bytes_from_openai += len(pcm)

                # Frame alignment proof: we allow a tail < frame_bytes, but we track it.
                tail = len(buf) % frame_bytes
                if tail and audio_frames % 50 == 0:
                    log_throttled(
                        "buf_tail",
                        f"Playout buffer tail (non-frame remainder)={tail} bytes (kept until full frame)",
                        every_s=2.0,
                    )

                if max_buf_bytes > 0 and len(buf) > max_buf_bytes:
                    # Drop only whole frames to keep alignment for mod_audio_stream
                    overflow = len(buf) - max_buf_bytes
                    frames_to_drop = (overflow + frame_bytes - 1) // frame_bytes
                    drop_bytes = frames_to_drop * frame_bytes
                    del buf[:drop_bytes]
                    log(
                        f"Playout buffer capped, dropped {drop_bytes} bytes "
                        f"({frames_to_drop} frames)"
                    )

                audio_frames += 1
                if audio_frames == 1:
                    log(f"OpenAI: first audio delta received (delta_bytes={len(pcm)})")
                elif audio_frames % 100 == 0:
                    log(
                        f"OpenAI: audio_deltas={audio_frames} bytes_from_openai={bytes_from_openai} last_delta_bytes={len(pcm)} buffer={len(buf)}"
                    )

            # ---- Turn detection (server VAD) ----
            elif evt_type in (
                "input_audio_buffer.speech_stopped",
                "input_audio_buffer.speech_end",
                "input_audio_buffer.speech_end_detected",
                "input_audio_buffer.vad_stop",
            ):
                # Commit at end-of-turn, but only if enough audio exists.
                await _maybe_commit(f"server_vad:{evt_type}")

            # ---- USER transcription (OpenAI -> terminal) ----
            elif evt_type in (
                "input_audio_transcription.delta",
                "input_audio_transcription.done",
                "input_audio_transcription.completed",
            ):
                delta = _extract_any_text_delta(evt)
                if delta:
                    user_text_buf.append(delta)
                    if stream_text_deltas:
                        if len(user_text_buf) == 1:
                            print("\nUSER: ", end="", flush=True)
                        _print_text_delta(delta)
                    if debug_text:
                        log_throttled("user_text", f"USER delta_chars={len(delta)}", every_s=1.0)

                if evt_type in ("input_audio_transcription.done", "input_audio_transcription.completed"):
                    if user_text_buf:
                        full_user = "".join(user_text_buf).strip()
                        user_text_buf.clear()
                        if full_user:
                            if stream_text_deltas:
                                # Close the streaming line.
                                print("", flush=True)
                            else:
                                print(f"\nUSER: {full_user}", flush=True)
                    # NOTE: We intentionally DO NOT auto-commit on transcription completion.
                    # Some servers emit transcription events without having committed audio in the
                    # input buffer, which can trigger commit_empty errors. We rely on server VAD
                    # end-of-turn events for commit+response.

            # ---- AI response text (OpenAI -> terminal) ----
            elif evt_type in (
                "response.output_text.delta",
                "response.text.delta",
                "response.output_text",
                "response.text",
                "response.output_text.done",
                "response.text.done",
            ) or (evt_type.startswith("response.") and "text" in evt_type):
                # Prefer direct delta/text fields when present.
                pieces: list[str] = []
                d = _extract_any_text_delta(evt)
                if d:
                    pieces = [d]
                else:
                    pieces = _extract_any_text_list(evt)

                if pieces:
                    joined = " ".join(pieces)
                    if joined:
                        ai_text_buf.append(joined)
                        if stream_text_deltas:
                            if len(ai_text_buf) == 1:
                                print("\nAI: ", end="", flush=True)
                            _print_text_delta(joined)
                    if debug_text:
                        log_throttled("ai_text", f"AI pieces={len(pieces)}", every_s=1.0)

                if evt_type.endswith(".done"):
                    if ai_text_buf:
                        full_ai = " ".join(ai_text_buf).strip()
                        ai_text_buf.clear()
                        if full_ai:
                            if stream_text_deltas:
                                print("", flush=True)
                            else:
                                print(f"\nAI: {full_ai}", flush=True)

            elif evt_type in (
                "input_audio_buffer.committed",
                "input_audio_buffer.commit",
            ):
                # Server acknowledged commit. Now it's safe to reset counters and create a response.
                if commit_pending:
                    tracker.on_committed()
                    reason = commit_reason or "commit_ack"
                    commit_pending = False
                    commit_reason = None
                    await _maybe_create_response(reason)
                else:
                    # Some servers might emit this even if we didn't track pending.
                    tracker.on_committed()

            elif evt_type in ("error", "session.created", "session.updated"):
                log(f"OpenAI event: {evt_type} -> {evt}")

                # If a commit failed, clear pending state so we can try again next turn.
                if evt_type == "error" and commit_pending:
                    err = evt.get("error")
                    code = err.get("code") if isinstance(err, dict) else None
                    if code in ("input_audio_buffer_commit_empty", "input_audio_buffer_commit_too_small"):
                        commit_pending = False
                        commit_reason = None

                # If the server rejects response creation because one is already active,
                # reflect that locally so we don't keep trying.
                if evt_type == "error":
                    err = evt.get("error")
                    code = err.get("code") if isinstance(err, dict) else None
                    if code == "conversation_already_has_active_response":
                        response_in_flight = True

            elif evt_type in ("response.created", "response.started", "response.done"):
                log(f"OpenAI event: {evt_type}")
                if evt_type in ("response.created", "response.started"):
                    response_in_flight = True
                if evt_type in ("response.done",):
                    response_in_flight = False

            elif evt_type in ("response.completed", "response.done"):
                # End-of-response marker: flush any buffered AI text.
                if ai_text_buf:
                    full_ai = " ".join(ai_text_buf).strip()
                    ai_text_buf.clear()
                    if full_ai:
                        if stream_text_deltas:
                            print("", flush=True)
                        else:
                            print(f"\nAI: {full_ai}", flush=True)
                response_in_flight = False

    finally:
        playout_task.cancel()



def _fmt_audio(sample_rate: int, channels: int, frame_ms: int) -> str:
    # Convenience for logs
    return f"pcm16/{channels}ch/{sample_rate}Hz/{frame_ms}ms"


def _pcm16_sine(
    sample_rate: int,
    freq_hz: float,
    duration_ms: int,
    amp: float = 0.2,
) -> bytes:
    """Generate a mono PCM16 sine wave."""
    import math

    n = int(sample_rate * (duration_ms / 1000.0))
    out = bytearray()
    for i in range(n):
        # [-1, 1]
        v = amp * math.sin(2.0 * math.pi * freq_hz * (i / sample_rate))
        s = int(max(-1.0, min(1.0, v)) * 32767)
        out += int(s).to_bytes(2, byteorder="little", signed=True)
    return bytes(out)


async def _send_test_tone(ws: websockets.WebSocketServerProtocol, cfg: BridgeConfig) -> None:
    """Send a short beep so we can prove the playback path works."""
    if not cfg.send_test_tone_on_connect:
        return

    fs_out_rate = int(cfg.fs_out_sample_rate or cfg.fs_sample_rate)
    if cfg.fs_send_json_audio and cfg.fs_channels != 1:
        log("Test tone: forcing FS_CHANNELS=1 for JSON injection")
        cfg.fs_channels = 1
    frame_bytes = _frame_bytes(fs_out_rate, cfg.fs_channels, cfg.fs_frame_ms)
    tone = _pcm16_sine(fs_out_rate, freq_hz=440.0, duration_ms=300)
    # pad to frame boundary
    if len(tone) % frame_bytes:
        tone += b"\x00" * (frame_bytes - (len(tone) % frame_bytes))

    log("Sending test tone (beep) to FreeSWITCH...")
    for off in range(0, len(tone), frame_bytes):
        frame = tone[off : off + frame_bytes]
        if cfg.fs_send_json_audio:
            await ws.send(_fs_streamaudio_json(frame, sample_rate=fs_out_rate, channels=cfg.fs_channels))
        else:
            await ws.send(frame)
        await asyncio.sleep(cfg.fs_frame_ms / 1000)
    log("Test tone sent")


async def handle_call(cfg: BridgeConfig, upstream_ws: websockets.WebSocketServerProtocol) -> None:
    peer = getattr(upstream_ws, "remote_address", None)
    log("====================================")
    log(f"Call connected: {peer}")
    log("Connecting to OpenAI Realtime...")

    openai_ws = await openai_realtime_session(cfg)
    log("OpenAI Realtime connected")
    log("====================================")

    if cfg.fs_send_json_audio and cfg.fs_send_json_handshake:
        try:
            await upstream_ws.send(
                _ws_text(
                    json.dumps(
                        {
                            "type": "start",
                            "format": "pcm16",
                            "rate": cfg.fs_sample_rate,
                            "channels": cfg.fs_channels,
                            "frame_ms": cfg.fs_frame_ms,
                        }
                    )
                )
            )
            log("FreeSWITCH: sent JSON handshake")
        except Exception as e:
            log(f"FreeSWITCH: failed to send JSON handshake: {e}")

    # Optional sanity check: if you don't hear this beep, the issue is on the FS side
    # (wrong UUID, stream not mixed, mod_audio_stream not playing, etc.).
    await _send_test_tone(upstream_ws, cfg)

    tracker = InputAudioTracker()
    to_openai = asyncio.create_task(pump_freeswitch_to_openai(upstream_ws, openai_ws, cfg, tracker))
    to_fs = asyncio.create_task(pump_openai_to_freeswitch(openai_ws, upstream_ws, cfg, tracker))

    done, pending = await asyncio.wait({to_openai, to_fs}, return_when=asyncio.FIRST_EXCEPTION)

    for t in pending:
        t.cancel()

    await openai_ws.close()


async def main() -> None:
    cfg = BridgeConfig(
        ivr_host=os.getenv("IVR_HOST", "0.0.0.0"),
        ivr_port=int(os.getenv("IVR_PORT", "8765")),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview"),
        voice=os.getenv("OPENAI_REALTIME_VOICE", "alloy"),
        force_commit_ms=int(os.getenv("OPENAI_FORCE_COMMIT_MS", "0")),
        force_response_on_commit=os.getenv("OPENAI_FORCE_RESPONSE_ON_COMMIT", "0") == "1",
        fs_frame_ms=int(os.getenv("FS_FRAME_MS", "20")),
        fs_sample_rate=int(os.getenv("FS_SAMPLE_RATE", "16000")),
        fs_channels=int(os.getenv("FS_CHANNELS", "1")),
        fs_out_sample_rate=int(os.getenv("FS_OUT_SAMPLE_RATE", "0"))
        or int(os.getenv("FS_SAMPLE_RATE", "16000")),
        playout_prebuffer_ms=int(os.getenv("PLAYOUT_PREBUFFER_MS", "300")),
        playout_max_buffer_ms=int(os.getenv("PLAYOUT_MAX_BUFFER_MS", "8000")),
        send_test_tone_on_connect=os.getenv("SEND_TEST_TONE", "0") == "1",
        response_min_interval_ms=int(os.getenv("RESPONSE_MIN_INTERVAL_MS", "400")),
        playout_catchup_max_ms=int(os.getenv("PLAYOUT_CATCHUP_MAX_MS", "120")),
        playout_sleep_granularity_ms=int(os.getenv("PLAYOUT_SLEEP_GRANULARITY_MS", "2")),
        fs_send_json_audio=os.getenv("FS_SEND_JSON_AUDIO", "0") == "1",
        fs_send_json_handshake=os.getenv(
            "FS_SEND_JSON_HANDSHAKE",
            "1" if os.getenv("FS_SEND_JSON_AUDIO", "0") == "1" else "0",
        )
        == "1",
    )

    if not cfg.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    log("====================================")
    log("OpenAI Realtime <-> FreeSWITCH bridge")
    log(f"Listening for FreeSWITCH on ws://{cfg.ivr_host}:{cfg.ivr_port}")
    log(f"OpenAI model: {cfg.model} | voice: {cfg.voice}")
    log(
        "Audio contract (FS ws): "
        f"{_fmt_audio(cfg.fs_sample_rate, cfg.fs_channels, cfg.fs_frame_ms)} "
        f"=> frame_bytes={_frame_bytes(cfg.fs_sample_rate, cfg.fs_channels, cfg.fs_frame_ms)}"
    )
    if cfg.fs_out_sample_rate != cfg.fs_sample_rate:
        log(
            "FreeSWITCH outbound audio: "
            f"pcm16/{cfg.fs_channels}ch/{cfg.fs_out_sample_rate}Hz/{cfg.fs_frame_ms}ms "
            f"=> frame_bytes={_frame_bytes(cfg.fs_out_sample_rate, cfg.fs_channels, cfg.fs_frame_ms)}"
        )
    log(
        "Playout: "
        f"prebuffer_ms={cfg.playout_prebuffer_ms} max_buffer_ms={cfg.playout_max_buffer_ms} "
        f"catchup_max_ms={cfg.playout_catchup_max_ms} sleep_granularity_ms={cfg.playout_sleep_granularity_ms}"
    )
    log(
        "FreeSWITCH outbound framing: "
        f"{'json+base64' if cfg.fs_send_json_audio else 'raw_binary_pcm'}"
    )
    log(
        "Response control: "
        f"force_commit_ms={cfg.force_commit_ms} force_response_on_commit={cfg.force_response_on_commit} "
        f"min_interval_ms={cfg.response_min_interval_ms}"
    )
    log("====================================")

    async def _handler(ws: websockets.WebSocketServerProtocol):
        try:
            await handle_call(cfg, ws)
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            log(f"Bridge error: {e}")

    async with websockets.serve(_handler, cfg.ivr_host, cfg.ivr_port, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Shutting down (Ctrl+C)")

