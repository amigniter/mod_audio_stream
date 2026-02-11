"""
OpenAI Realtime API WebSocket client.

Establishes the connection and configures the session with optimal settings
for voice-to-voice over telephone (matching ChatGPT Voice quality).
"""
from __future__ import annotations

import json
import logging
import ssl
from typing import Optional

import websockets

logger = logging.getLogger(__name__)


def build_ssl_context(wss_pem: str, insecure: bool) -> ssl.SSLContext:
    """Create SSL context for OpenAI WebSocket connection."""
    if insecure:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        logger.warning("TLS verification DISABLED (insecure mode)")
        return ctx

    ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)

    if wss_pem:
        try:
            with open(wss_pem, "rb") as f:
                head = f.read(4096)
            if b"PRIVATE KEY" in head:
                raise ValueError("WSS_PEM looks like a private key; provide CA bundle")
            ctx.load_verify_locations(cafile=wss_pem)
            logger.info("TLS: custom CA from %s", wss_pem)
            return ctx
        except Exception as e:
            logger.warning("TLS: failed to load %s: %s (trying certifi)", wss_pem, e)

    try:
        import certifi
        ctx.load_verify_locations(cafile=certifi.where())
    except Exception:
        pass  

    return ctx


async def connect_openai_realtime(
    *,
    api_key: str,
    model: str,
    voice: str,
    ssl_ctx: ssl.SSLContext,
    vad_threshold: float = 0.5,
    vad_prefix_padding_ms: int = 300,
    vad_silence_duration_ms: int = 500,
    temperature: float = 0.8,
    system_instructions: str = "",
    text_only_mode: bool = False,
) -> websockets.WebSocketClientProtocol:
    """Connect to OpenAI Realtime API and configure session.

    When text_only_mode=True, the session uses modalities=["text"] so
    OpenAI returns only text responses (no audio). This is ~3-5x faster
    and allows using a custom TTS engine for voice synthesis.
    """

    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    # websockets >=12 uses `additional_headers`; older versions use
    # `extra_headers`.  Try the modern kwarg first, fall back once.
    ws: Optional[websockets.WebSocketClientProtocol] = None
    for kwarg_name in ("additional_headers", "extra_headers"):
        try:
            ws = await websockets.connect(
                url, max_size=None, ssl=ssl_ctx,
                **{kwarg_name: headers},
            )
            break
        except TypeError:
            continue

    if ws is None:
        raise RuntimeError(
            "Cannot pass headers to websockets — upgrade: pip install 'websockets>=12'"
        )

    default_instructions = (
        "You are a warm, friendly voice assistant on a phone call. "
        "Speak exactly like a real human would — with natural rhythm, "
        "pauses, emphasis, and emotion. Vary your tone and pacing. "
        "Use contractions (I'm, you're, let's, can't) and casual filler "
        "words (well, so, actually, hmm) when appropriate. "
        "Keep responses concise: 1 to 3 sentences unless asked for detail. "
        "Never sound like you're reading from a script. "
        "If the user's audio is unclear, say something like 'Sorry, I didn't quite catch that — could you say that again?' "
        "Match the user's energy — if they're excited, be excited back. "
        "If they're calm, be calm and reassuring."
    )

    if text_only_mode:
        modalities = ["text"]
        logger.info("OpenAI session: TEXT-ONLY mode (custom TTS will handle audio)")
    else:
        modalities = ["audio", "text"]

    session_config = {
        "input_audio_format": "pcm16",

        "voice": voice,
        "modalities": modalities,

        "turn_detection": {
            "type": "server_vad",
            "create_response": True,
            "threshold": vad_threshold,
            "prefix_padding_ms": vad_prefix_padding_ms,
            "silence_duration_ms": vad_silence_duration_ms,
        },

        "input_audio_transcription": {
            "model": "gpt-4o-mini-transcribe",
            "language": "en",
        },

        "temperature": temperature,

        "instructions": system_instructions or default_instructions,
    }

    if not text_only_mode:
        session_config["output_audio_format"] = "pcm16"

    session_update = {
        "type": "session.update",
        "session": session_config,
    }

    await ws.send(json.dumps(session_update))
    logger.info(
        "OpenAI session: voice=%s vad=%.2f silence=%dms prefix=%dms temp=%.1f",
        voice, vad_threshold, vad_silence_duration_ms,
        vad_prefix_padding_ms, temperature,
    )

    return ws
