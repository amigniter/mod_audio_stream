# ...existing code...
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import math

import audioop
import websockets

try:
    from vosk import Model, KaldiRecognizer  # type: ignore
except Exception:
    Model = None
    KaldiRecognizer = None


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _frame_bytes(sample_rate: int, channels: int, frame_ms: int) -> int:
    return int(sample_rate * (frame_ms / 1000.0)) * channels * 2


def _streamaudio_json(pcm16: bytes, sample_rate: int, channels: int) -> str:
    """Build JSON the module expects for AI->FS injection (pushback).

    Matches audio_streamer_glue.cpp: type=streamAudio, data.audioDataType=raw,
    data.audioData=base64(PCM16LE).
    """
    return json.dumps(
        {
            "type": "streamAudio",
            "data": {
                "audioDataType": "raw",
                "audioData": base64.b64encode(pcm16).decode("ascii"),
                "sampleRate": int(sample_rate),
                "channels": int(channels),
            },
        }
    )


def _is_pushback_ack(message: bytes | str) -> tuple[bool, str]:
    """Detect pushback/inject ACKs from mod_audio_stream.

    Your C++ code rewrites the JSON and adds the field `bytes` (decoded.size()),
    then sends it via EVENT_PLAY. We ONLY treat messages containing `bytes` as an
    injection ACK so we don't accidentally classify other JSON as ACKs.
    """
    if isinstance(message, (bytes, bytearray)):
        return (False, "")
    try:
        obj = json.loads(message)
    except Exception:
        return (False, "")
    if not isinstance(obj, dict):
        return (False, "")

    # Two common cases:
    # 1) Sender receives rewritten jsonData only: {audioDataType, sampleRate, channels, bytes, ...}
    # 2) Full wrapper echoed back: {type:'streamAudio', data:{..., bytes}}
    if "bytes" in obj:
        return (True, message)
    if obj.get("type") == "streamAudio" and isinstance(obj.get("data"), dict):
        data = obj["data"]
        if "bytes" in data:
            return (True, message)
    return (False, "")


def _pushback_ack_summary(ack_text: str) -> str:
    """Pretty-print ACK fields so logs stay readable."""
    try:
        obj = json.loads(ack_text)
    except Exception:
        return ack_text

    if isinstance(obj, dict) and obj.get("type") == "streamAudio" and isinstance(obj.get("data"), dict):
        obj = obj["data"]

    if not isinstance(obj, dict):
        return ack_text

    # Only summarize if it looks like an actual ACK
    if "bytes" not in obj:
        return ack_text

    b = obj.get("bytes")
    sr = obj.get("sampleRate")
    ch = obj.get("channels")
    adt = obj.get("audioDataType")

    def _fmt(v) -> str:
        if isinstance(v, bool) or v is None:
            return "?"
        if isinstance(v, (int, float)):
            return str(int(v))
        if isinstance(v, str):
            return v
        return "?"

    return f"bytes={_fmt(b)} sr={_fmt(sr)} ch={_fmt(ch)} type={_fmt(adt)}"


def _pcm16_sine(sample_rate: int, freq_hz: float, duration_ms: int, amp: float = 0.2) -> bytes:
    """Generate mono PCM16LE sine wave."""
    n = int(sample_rate * (duration_ms / 1000.0))
    out = bytearray()
    for i in range(n):
        v = amp * math.sin(2.0 * math.pi * freq_hz * (i / sample_rate))
        s = int(max(-1.0, min(1.0, v)) * 32767)
        out += int(s).to_bytes(2, byteorder="little", signed=True)
    return bytes(out)


async def _inject_test_tone(ws, sample_rate: int, channels: int) -> None:
    """Send a short beep into the call via pushback/inject.

    Enable with: INJECT_TEST_TONE=1
    """
    if os.getenv("INJECT_TEST_TONE", "0") != "1":
        return

    # Optional override so injection can be tested at 8000Hz even if you are
    # sniffing/transcribing at a different rate.
    inject_sr = int(os.getenv("INJECT_SAMPLE_RATE", str(sample_rate)))
    inject_ch = int(os.getenv("INJECT_CHANNELS", str(channels)))

    frame_ms = int(os.getenv("INJECT_FRAME_MS", os.getenv("FRAME_MS", "20")))
    freq = float(os.getenv("INJECT_TONE_HZ", "440"))
    dur_ms = int(os.getenv("INJECT_TONE_MS", "400"))
    amp = float(os.getenv("INJECT_TONE_AMP", "0.2"))

    if inject_ch != 1:
        log("INJECT_TEST_TONE currently supports only mono (INJECT_CHANNELS=1)")
        return

    fb = _frame_bytes(inject_sr, inject_ch, frame_ms)
    if fb <= 0:
        return

    tone = _pcm16_sine(inject_sr, freq_hz=freq, duration_ms=dur_ms, amp=amp)
    if len(tone) % fb:
        tone += b"\x00" * (fb - (len(tone) % fb))

    log(
        "PUSHBACK: sending test tone "
        f"freq={freq:.0f}Hz dur_ms={dur_ms} frame_ms={frame_ms} "
        f"sampleRate={inject_sr} channels={inject_ch} frame_bytes={fb}"
    )

    total_frames = 0
    total_bytes = 0
    for off in range(0, len(tone), fb):
        frame = tone[off : off + fb]
        await ws.send(_streamaudio_json(frame, sample_rate=inject_sr, channels=inject_ch))
        total_frames += 1
        total_bytes += len(frame)
        await asyncio.sleep(frame_ms / 1000.0)

    log(f"PUSHBACK: test tone sent frames={total_frames} bytes={total_bytes}")


def _extract_pcm_from_ws_message(message: bytes | str) -> bytes:
    if isinstance(message, (bytes, bytearray)):
        return bytes(message)

    try:
        evt = json.loads(message)
    except Exception:
        return b""
    if not isinstance(evt, dict) or evt.get("type") != "streamAudio":
        return b""
    data = evt.get("data")
    if not isinstance(data, dict):
        return b""
    audio_b64 = data.get("audioData")
    if not isinstance(audio_b64, str) or not audio_b64:
        return b""
    try:
        return base64.b64decode(audio_b64)
    except Exception:
        return b""


def _init_vosk(sample_rate: int):
    model_path = os.getenv("VOSK_MODEL_PATH", "").strip()
    if not model_path:
        raise RuntimeError("Set VOSK_MODEL_PATH to your Vosk model directory")

    if Model is None or KaldiRecognizer is None:
        raise RuntimeError("vosk not installed. Run: pip install vosk")

    if not os.path.isdir(model_path):
        raise RuntimeError(f"VOSK_MODEL_PATH not found: {model_path}")

    model = Model(model_path)
    # NOTE: Many Vosk models are trained for a fixed sample rate (commonly 16000).
    # If we pass a mismatched rate (e.g. 8000) Vosk can error out.
    rec = KaldiRecognizer(model, float(sample_rate))
    try:
        rec.SetWords(False)
    except Exception:
        pass
    return rec


def _vosk_text(rec) -> str:
    """Extract 'text' from Vosk JSON result."""
    try:
        obj = json.loads(rec.FinalResult())
    except Exception:
        return ""
    txt = obj.get("text")
    return txt.strip() if isinstance(txt, str) else ""


async def _sniff_stream(ws, sample_rate: int, channels: int) -> None:
    if channels != 1:
        raise RuntimeError("This Vosk sniffer expects AUDIO_CHANNELS=1 (mono)")

    frame_ms = int(os.getenv("FRAME_MS", "20"))
    silence_end_ms = int(os.getenv("SILENCE_END_MS", "600"))
    min_utt_ms = int(os.getenv("MIN_UTTERANCE_MS", "300"))
    threshold = int(os.getenv("VAD_THRESHOLD", "500"))

    bytes_per_frame = int(sample_rate * (frame_ms / 1000.0)) * channels * 2
    if bytes_per_frame <= 0:
        raise RuntimeError("Invalid frame sizing")

    # Vosk model rate: default to 16000 (matches vosk-model-small-en-us-0.15).
    # If you use an 8k model, set VOSK_SAMPLE_RATE=8000.
    vosk_sr = int(os.getenv("VOSK_SAMPLE_RATE", "16000"))
    rec = _init_vosk(vosk_sr)
    if sample_rate != vosk_sr:
        log(f"VOSK: resampling input {sample_rate}Hz -> {vosk_sr}Hz")
    ratecv_state = None

    buf = bytearray()
    utt = bytearray()
    in_speech = False
    silence_ms = 0

    def _flush_utterance() -> None:
        nonlocal utt, in_speech, silence_ms
        if not utt:
            in_speech = False
            silence_ms = 0
            return

        duration_ms = (len(utt) / (sample_rate * channels * 2)) * 1000.0
        in_speech = False
        silence_ms = 0

        if duration_ms < float(min_utt_ms):
            utt = bytearray()
            rec.Reset()
            return

        pcm = bytes(utt)
        if sample_rate != vosk_sr:
            try:
                pcm, ratecv_state = audioop.ratecv(pcm, 2, channels, sample_rate, vosk_sr, ratecv_state)
            except Exception:
                # If resampling fails, try without resampling (may still fail on mismatch).
                pass

        rec.AcceptWaveform(pcm)
        text = _vosk_text(rec)
        rec.Reset()
        utt = bytearray()

        if text:
            print(f"\nUSER: {text}", flush=True)
        else:
            log(f"SPEECH: {duration_ms:.0f}ms (no text)")

    async for msg in ws:
        is_ack, ack_text = _is_pushback_ack(msg)
        if is_ack:
            log(f"PUSHBACK_ACK: {_pushback_ack_summary(ack_text)}")
            continue

        pcm = _extract_pcm_from_ws_message(msg)
        if not pcm:
            continue

        # keep PCM16 aligned
        if len(pcm) % 2:
            pcm = pcm[:-1]

        buf.extend(pcm)

        while len(buf) >= bytes_per_frame:
            frame = bytes(buf[:bytes_per_frame])
            del buf[:bytes_per_frame]

            try:
                avg = abs(int(audioop.avg(frame, 2)))
            except Exception:
                avg = 0

            if avg >= threshold:
                in_speech = True
                silence_ms = 0
                utt.extend(frame)
            else:
                if in_speech:
                    utt.extend(frame)
                    silence_ms += frame_ms
                    if silence_ms >= silence_end_ms:
                        _flush_utterance()

    _flush_utterance()


async def main() -> None:
    mode = os.getenv("MODE", "server").strip().lower()
    listen_host = os.getenv("LISTEN_HOST", "0.0.0.0").strip()
    listen_port = int(os.getenv("LISTEN_PORT", "8765"))
    ws_url = os.getenv("MOD_AUDIO_STREAM_WS", "ws://127.0.0.1:8765").strip()
    sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "8000"))
    channels = int(os.getenv("AUDIO_CHANNELS", "1"))

    log("====================================")
    log("mod_audio_stream -> Vosk transcript (local-only, console only)")
    log("Optional PUSHBACK/inject: set INJECT_TEST_TONE=1 to send a beep into the call")
    log(f"Mode: {mode}")
    if mode == "client":
        log(f"Connecting: {ws_url}")
    else:
        log(f"Listening: ws://{listen_host}:{listen_port}")
    log(f"Audio expected: pcm16/{channels}ch/{sample_rate}Hz")
    log(f"VOSK_MODEL_PATH: {os.getenv('VOSK_MODEL_PATH','').strip()}")
    log("====================================")

    if mode == "client":
        async with websockets.connect(ws_url, max_size=None) as ws:
            # Inject on connect (optional), then continue sniffing/transcribing.
            await _inject_test_tone(ws, sample_rate=sample_rate, channels=channels)
            await _sniff_stream(ws, sample_rate, channels)
        return

    if mode != "server":
        raise SystemExit("MODE must be 'server' or 'client'")

    async def handler(ws):
        peer = getattr(ws, "remote_address", None)
        log(f"Client connected: {peer}")
        try:
            # Inject on connect (optional), then continue sniffing/transcribing.
            await _inject_test_tone(ws, sample_rate=sample_rate, channels=channels)
            await _sniff_stream(ws, sample_rate, channels)
        except websockets.ConnectionClosed:
            pass
        finally:
            log(f"Client disconnected: {peer}")

    async with websockets.serve(handler, listen_host, listen_port, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Shutting down (Ctrl+C)")
# ...existing code...