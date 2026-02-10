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
        pass  # Fall back to system CAs

    return ctx


async def connect_openai_realtime(
    *,
    api_key: str,
    model: str,
    voice: str,
    ssl_ctx: ssl.SSLContext,
    # ── VAD tuning (controls when AI thinks user stopped talking) ──
    vad_threshold: float = 0.5,
    vad_prefix_padding_ms: int = 300,
    vad_silence_duration_ms: int = 500,
    # ── AI behavior ──
    temperature: float = 0.8,
    system_instructions: str = "",
) -> websockets.WebSocketClientProtocol:
    """Connect to OpenAI Realtime API and configure session for phone-quality voice."""

    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    # Try different header parameter names for websockets compatibility
    ws: Optional[websockets.WebSocketClientProtocol] = None
    for key in ("extra_headers", "headers", "additional_headers"):
        try:
            ws = await websockets.connect(
                url, max_size=None, ssl=ssl_ctx,
                **{key: headers},
            )
            break
        except TypeError:
            pass

    if ws is None:
        for key in ("extra_headers", "headers", "additional_headers"):
            try:
                ws = await websockets.connect(
                    url, max_size=None, ssl=ssl_ctx,
                    **{key: list(headers.items())},
                )
                break
            except TypeError:
                pass

    if ws is None:
        raise RuntimeError("Cannot pass headers to websockets — upgrade: pip install websockets>=12")

    # ── Configure session for maximum voice quality ──
    #
    # This is THE key to matching ChatGPT Voice quality.
    # ChatGPT has all these tuned internally. We set them explicitly.
    #
    default_instructions = (
        "You are a helpful, friendly voice assistant on a phone call. "
        "Keep responses concise and conversational (1 to 3 sentences). "
        "Speak naturally as if talking to a friend. "
        "If the user's audio is unclear, ask them to repeat politely."
    )

    session_update = {
        "type": "session.update",
        "session": {
            # ── Audio format: 24kHz PCM16 = maximum quality ──
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",

            # ── Voice and modalities ──
            "voice": voice,
            "modalities": ["audio", "text"],

            # ── Turn detection (VAD) — critical for conversation flow ──
            #
            # threshold: 0.0-1.0. Higher = needs louder speech to trigger.
            #   0.5 = good for phone lines with background noise
            #   0.3 = more sensitive (quiet speakers)
            #   0.7 = less sensitive (noisy environments)
            #
            # prefix_padding_ms: audio before speech-start to include.
            #   300ms captures beginning of words before VAD triggered.
            #
            # silence_duration_ms: silence before ending turn.
            #   500ms = natural conversation pace
            #   300ms = faster, more responsive
            #   800ms = allows mid-sentence pauses
            #
            "turn_detection": {
                "type": "server_vad",
                "create_response": True,
                "threshold": vad_threshold,
                "prefix_padding_ms": vad_prefix_padding_ms,
                "silence_duration_ms": vad_silence_duration_ms,
            },

            # ── Transcription (for logging/debugging) ──
            "input_audio_transcription": {
                "model": "gpt-4o-mini-transcribe",
                "language": "en",
            },

            # ── Response behavior ──
            "temperature": temperature,

            # ── System instructions ──
            "instructions": system_instructions or default_instructions,
        },
    }

    await ws.send(json.dumps(session_update))
    logger.info(
        "OpenAI session: voice=%s vad=%.2f silence=%dms prefix=%dms temp=%.1f",
        voice, vad_threshold, vad_silence_duration_ms,
        vad_prefix_padding_ms, temperature,
    )

    return ws
