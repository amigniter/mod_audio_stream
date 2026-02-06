from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"{key} must be an int, got {v!r}") from e


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class BridgeConfig:
    host: str
    port: int
    openai_api_key: str
    model: str
    voice: str
    fs_frame_ms: int
    fs_sample_rate: int
    fs_channels: int
    fs_out_sample_rate: int
    playout_prebuffer_ms: int
    playout_max_buffer_ms: int
    playout_catchup_max_ms: int
    playout_sleep_granularity_ms: int
    force_commit_ms: int
    force_response_on_commit: bool
    response_min_interval_ms: int
    fs_send_json_audio: bool
    fs_send_json_handshake: bool
    openai_input_sample_rate: int
    openai_resample_input: bool
    wss_pem: str
    openai_wss_insecure: bool
    send_test_tone_on_connect: bool


def load_config(env_file: str | None = None) -> BridgeConfig:
    """Load config from .env + environment.

    Precedence: real environment wins over .env values.
    """
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)

    host = os.getenv("HOST", os.getenv("IVR_HOST", "0.0.0.0")).strip()
    port = _env_int("PORT", _env_int("IVR_PORT", 8765))

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")

    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview").strip()
    voice = os.getenv("OPENAI_REALTIME_VOICE", "alloy").strip()

    fs_frame_ms = _env_int("FS_FRAME_MS", 20)
    fs_sample_rate = _env_int("FS_SAMPLE_RATE", 16000)
    fs_channels = _env_int("FS_CHANNELS", 1)

    fs_out_sample_rate = _env_int("FS_OUT_SAMPLE_RATE", fs_sample_rate)

    playout_prebuffer_ms = _env_int("PLAYOUT_PREBUFFER_MS", 60)
    playout_max_buffer_ms = _env_int("PLAYOUT_MAX_BUFFER_MS", 1200)
    playout_catchup_max_ms = _env_int("PLAYOUT_CATCHUP_MAX_MS", 120)
    playout_sleep_granularity_ms = _env_int("PLAYOUT_SLEEP_GRANULARITY_MS", 2)

    force_commit_ms = _env_int("OPENAI_FORCE_COMMIT_MS", 0)
    force_response_on_commit = _env_bool("OPENAI_FORCE_RESPONSE_ON_COMMIT", False)
    response_min_interval_ms = _env_int("RESPONSE_MIN_INTERVAL_MS", 400)

    fs_send_json_audio = _env_bool("FS_SEND_JSON_AUDIO", False)
    fs_send_json_handshake = _env_bool(
        "FS_SEND_JSON_HANDSHAKE",
        True if fs_send_json_audio else False,
    )

    # OpenAI input side: many realtime pipelines behave best at 16k mono PCM16.
    openai_input_sample_rate = _env_int("OPENAI_INPUT_SAMPLE_RATE", 16000)
    openai_resample_input = _env_bool("OPENAI_RESAMPLE_INPUT", True)

    wss_pem = os.getenv("WSS_PEM", "").strip().strip('"').strip("'")
    if wss_pem and not os.path.isabs(wss_pem):
        wss_pem = str(Path(wss_pem).expanduser().resolve())

    openai_wss_insecure = _env_bool("OPENAI_WSS_INSECURE", False)

    send_test_tone_on_connect = _env_bool("SEND_TEST_TONE", False)

    return BridgeConfig(
        host=host,
        port=port,
        openai_api_key=openai_api_key,
        model=model,
        voice=voice,
        fs_frame_ms=fs_frame_ms,
        fs_sample_rate=fs_sample_rate,
        fs_channels=fs_channels,
        fs_out_sample_rate=fs_out_sample_rate,
        playout_prebuffer_ms=playout_prebuffer_ms,
        playout_max_buffer_ms=playout_max_buffer_ms,
        playout_catchup_max_ms=playout_catchup_max_ms,
        playout_sleep_granularity_ms=playout_sleep_granularity_ms,
        force_commit_ms=force_commit_ms,
        force_response_on_commit=force_response_on_commit,
        response_min_interval_ms=response_min_interval_ms,
        fs_send_json_audio=fs_send_json_audio,
        fs_send_json_handshake=fs_send_json_handshake,
        openai_input_sample_rate=openai_input_sample_rate,
        openai_resample_input=openai_resample_input,
        wss_pem=wss_pem,
        openai_wss_insecure=openai_wss_insecure,
        send_test_tone_on_connect=send_test_tone_on_connect,
    )
