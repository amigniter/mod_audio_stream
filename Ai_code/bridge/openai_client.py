from __future__ import annotations

import logging
import ssl
import websockets

logger = logging.getLogger(__name__)


def build_ssl_context(wss_pem: str, insecure: bool) -> ssl.SSLContext:
    """Create SSL context for outbound OpenAI WSS.

    - If insecure: disables verification (dev only)
    - Else: uses WSS_PEM if provided, else falls back to certifi/system
    """
    if insecure:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        logger.warning("OPENAI_WSS_INSECURE=1 -> TLS verification disabled")
        return ctx

    ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)

    if wss_pem:
        try:
            with open(wss_pem, "rb") as f:
                head = f.read(4096)
            if b"PRIVATE KEY" in head:
                raise ValueError("WSS_PEM looks like a private key; provide CA bundle PEM")

            ctx.load_verify_locations(cafile=wss_pem)
            logger.info("TLS: using CA bundle from WSS_PEM=%s", wss_pem)
            return ctx
        except Exception as e:
            logger.warning("TLS: failed to load WSS_PEM=%s: %s (will try certifi/system)", wss_pem, e)

    try:
        import certifi

        cafile = certifi.where()
        ctx.load_verify_locations(cafile=cafile)
        logger.info("TLS: using certifi CA bundle: %s", cafile)
    except Exception:
        pass

    return ctx


async def connect_openai_realtime(
    *,
    api_key: str,
    model: str,
    voice: str,
    ssl_ctx: ssl.SSLContext,
):
    url = f"wss://api.openai.com/v1/realtime?model={model}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

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
        raise RuntimeError("websockets package too old/incompatible to pass headers; upgrade websockets")

    session_update = {
        "type": "session.update",
        "session": {
            "voice": voice,
            "instructions": (
                "You are a helpful AI IVR agent. ALWAYS respond in English only. "
                "Keep every response short (1-2 sentences). "
                "Use simple words. Ask at most one question at a time."
            ),
            "temperature": 0.6,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "gpt-4o-mini-transcribe", "language": "en"},
            "turn_detection": {"type": "server_vad", "create_response": False},
        },
    }

    await ws.send(__import__("json").dumps(session_update))

    return ws
