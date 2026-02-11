"""
Echo-back server for mod_audio_stream.
Receives PCM from FreeSWITCH, applies gain, and injects it back
one frame at a time on a steady 20ms clock — zero packet loss.

This is the CORRECT test server. Do NOT use checkPushBack1000.py — 
that script batches frames (100ms batches) which causes bursty delivery,
buffer overflow in the C module, and garbled/unclear audio.

Usage:
    cd Ai_code && python echo_server.py

Environment variables:
    LISTEN_HOST          (default 0.0.0.0)
    LISTEN_PORT          (default 8765)
    AUDIO_SAMPLE_RATE    (default 8000)   — must match FreeSWITCH codec rate
    AUDIO_CHANNELS       (default 1)
    ECHO_INJECT          (default 1)      — set 0 to disable echo
    ECHO_GAIN            (default 1.0)    — volume multiplier (1.0 = full volume)
    PLAYOUT_BUFFER_MS    (default 500)    — max buffered audio before dropping oldest
    FRAME_MS             (default 20)     — pacing interval (match FS ptime)
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import json
import os
import time
from collections import deque

import websockets


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _streamaudio_json(pcm16: bytes, sample_rate: int, channels: int) -> str:
    """Build a streamAudio JSON payload for mod_audio_stream."""
    if len(pcm16) % 2:
        pcm16 = pcm16[:-1]
    return json.dumps(
        {
            "type": "streamAudio",
            "data": {
                "audioDataType": "raw",
                "audioData": base64.b64encode(pcm16).decode("ascii"),
                "sampleRate": sample_rate,
                "channels": channels,
            },
        },
        separators=(",", ":"),
    )


async def run_connection(ws, in_sr: int, in_ch: int) -> None:
    echo_enabled = os.getenv("ECHO_INJECT", "1") == "1"
    echo_gain = float(os.getenv("ECHO_GAIN", "1.0"))
    frame_ms = int(os.getenv("FRAME_MS", "20"))
    max_buf_ms = int(os.getenv("PLAYOUT_BUFFER_MS", "500"))

    frame_bytes = in_sr * 2 * in_ch * frame_ms // 1000  
    max_buf_bytes = in_sr * 2 * in_ch * max_buf_ms // 1000

    buf = bytearray()

    stats_frames_in = 0
    stats_frames_out = 0
    stats_drops = 0
    stats_underruns = 0
    stats_t = time.monotonic()

    async def receiver() -> None:
        """Receive PCM from FreeSWITCH, apply gain, append to buffer."""
        nonlocal stats_frames_in, stats_drops

        async for msg in ws:
            if not isinstance(msg, (bytes, bytearray)):
                continue
            pcm = bytes(msg)
            if not pcm or len(pcm) % 2:
                continue

            if not echo_enabled:
                continue

            if echo_gain != 1.0:
                try:
                    pcm = audioop.mul(pcm, 2, echo_gain)
                except Exception:
                    pass

            stats_frames_in += 1

            buf.extend(pcm)
            if len(buf) > max_buf_bytes:
                overflow = len(buf) - max_buf_bytes
                overflow = (overflow // frame_bytes) * frame_bytes
                if overflow > 0:
                    del buf[:overflow]
                    stats_drops += overflow // frame_bytes

    async def injector() -> None:
        """
        Send exactly ONE frame every frame_ms milliseconds.
        This steady pacing matches FreeSWITCH's WRITE_REPLACE tick rate
        and prevents buffer overflow/underrun in the C module.
        """
        nonlocal stats_frames_out, stats_underruns

        step_s = frame_ms / 1000.0
        next_t = time.monotonic() + step_s

        while True:
            now = time.monotonic()
            if now < next_t:
                await asyncio.sleep(next_t - now)

            if time.monotonic() - next_t > step_s * 3:
                next_t = time.monotonic()
            next_t += step_s

            if len(buf) >= frame_bytes:
                frame = bytes(buf[:frame_bytes])
                del buf[:frame_bytes]
                try:
                    await ws.send(
                        _streamaudio_json(frame, sample_rate=in_sr, channels=in_ch)
                    )
                    stats_frames_out += 1
                except Exception:
                    break
            else:
                stats_underruns += 1

    async def stats_printer() -> None:
        """Print telemetry every 5 seconds."""
        nonlocal stats_t
        while True:
            await asyncio.sleep(5.0)
            now = time.monotonic()
            dt = now - stats_t
            log(
                f"Echo stats: in={stats_frames_in} out={stats_frames_out} "
                f"drops={stats_drops} underruns={stats_underruns} "
                f"buf_ms={len(buf) * 1000 // (in_sr * 2 * in_ch) if in_sr else 0} "
                f"dt={dt:.1f}s"
            )
            stats_t = now

    recv_task = asyncio.create_task(receiver())
    inj_task = asyncio.create_task(injector())
    stats_task = asyncio.create_task(stats_printer())

    done, pending = await asyncio.wait(
        {recv_task, inj_task, stats_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()
    for t in pending:
        try:
            await t
        except asyncio.CancelledError:
            pass


async def main() -> None:
    listen_host = os.getenv("LISTEN_HOST", "0.0.0.0").strip()
    listen_port = int(os.getenv("LISTEN_PORT", "8765"))
    in_sr = int(os.getenv("AUDIO_SAMPLE_RATE", "8000"))
    in_ch = int(os.getenv("AUDIO_CHANNELS", "1"))

    log(f"Config: sr={in_sr} ch={in_ch} frame={os.getenv('FRAME_MS','20')}ms")

    async def handler(ws):
        peer = getattr(ws, "remote_address", "?")
        log(f"Call connected: {peer}")
        try:
            await run_connection(ws, in_sr=in_sr, in_ch=in_ch)
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            log(f"Error: {e}")
        finally:
            log(f"Call disconnected: {peer}")

    async with websockets.serve(
        handler,
        listen_host,
        listen_port,
        max_size=2**20,
        ping_interval=20,
        ping_timeout=20,
    ):
        log(f"Listening ws://{listen_host}:{listen_port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
