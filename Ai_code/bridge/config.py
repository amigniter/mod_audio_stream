from __future__ import annotations
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"{key} must be an int, got {v!r}") from e


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except Exception as e:
        raise ValueError(f"{key} must be a float, got {v!r}") from e


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
    force_commit_ms: int
    force_response_on_commit: bool
    response_min_interval_ms: int
    fs_send_json_audio: bool
    fs_send_json_handshake: bool
    openai_input_sample_rate: int
    openai_resample_input: bool
    openai_output_sample_rate: int
    openai_input_mode: str
    openai_item_max_buffer_ms: int
    wss_pem: str
    openai_wss_insecure: bool
    vad_threshold: float
    vad_prefix_padding_ms: int
    vad_silence_duration_ms: int
    temperature: float
    system_instructions: str

    tts_provider: str = "none"
    tts_api_key: str = ""
    tts_voice_id: str = ""
    tts_model: str = ""
    tts_fallback_provider: str = "openai"
    tts_fallback_api_key: str = ""
    tts_fallback_voice_id: str = ""
    tts_fallback_model: str = ""
    tts_selfhosted_url: str = ""
    tts_sentence_max_chars: int = 80
    tts_sentence_min_chars: int = 10
    tts_cache_enabled: bool = True
    tts_cache_max_entries: int = 500
    tts_cache_ttl_s: float = 3600.0
    health_port: int = 8766
    max_concurrent_calls: int = 100


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
    fs_sample_rate = _env_int("FS_SAMPLE_RATE", 8000)
    fs_channels = _env_int("FS_CHANNELS", 1)
    fs_out_sample_rate = _env_int("FS_OUT_SAMPLE_RATE", 24000)

    playout_prebuffer_ms = _env_int("PLAYOUT_PREBUFFER_MS", 100)

    force_commit_ms = _env_int("OPENAI_FORCE_COMMIT_MS", 0)
    force_response_on_commit = _env_bool("OPENAI_FORCE_RESPONSE_ON_COMMIT", False)
    response_min_interval_ms = _env_int("RESPONSE_MIN_INTERVAL_MS", 200)

    fs_send_json_audio = _env_bool("FS_SEND_JSON_AUDIO", False)
    fs_send_json_handshake = _env_bool(
        "FS_SEND_JSON_HANDSHAKE",
        True if fs_send_json_audio else False,
    )

    openai_input_sample_rate = _env_int("OPENAI_INPUT_SAMPLE_RATE", 24000)
    openai_resample_input = _env_bool("OPENAI_RESAMPLE_INPUT", True)

    openai_output_sample_rate = _env_int("OPENAI_OUTPUT_SAMPLE_RATE", 24000)

    openai_input_mode = os.getenv("OPENAI_INPUT_MODE", "buffer").strip().lower() or "buffer"
    if openai_input_mode not in ("buffer", "item"):
        raise ValueError("OPENAI_INPUT_MODE must be 'buffer' or 'item'")

    openai_item_max_buffer_ms = _env_int("OPENAI_ITEM_MAX_BUFFER_MS", 20000)

    wss_pem = os.getenv("WSS_PEM", "").strip().strip('"').strip("'")
    if wss_pem and not os.path.isabs(wss_pem):
        wss_pem = str(Path(wss_pem).expanduser().resolve())

    openai_wss_insecure = _env_bool("OPENAI_WSS_INSECURE", False)

    vad_threshold = _env_float("VAD_THRESHOLD", 0.5)
    vad_prefix_padding_ms = _env_int("VAD_PREFIX_PADDING_MS", 300)
    vad_silence_duration_ms = _env_int("VAD_SILENCE_DURATION_MS", 300)

    temperature = _env_float("OPENAI_TEMPERATURE", 0.6)
    system_instructions = os.getenv("OPENAI_SYSTEM_INSTRUCTIONS", "").strip()

    # ── Custom TTS settings ──
    tts_provider = os.getenv("TTS_PROVIDER", "none").strip().lower()
    tts_api_key = os.getenv("TTS_API_KEY", "").strip()
    tts_voice_id = os.getenv("TTS_VOICE_ID", "").strip()
    tts_model = os.getenv("TTS_MODEL", "").strip()
    tts_fallback_provider = os.getenv("TTS_FALLBACK_PROVIDER", "openai").strip().lower()
    tts_fallback_api_key = os.getenv("TTS_FALLBACK_API_KEY", "").strip()
    tts_fallback_voice_id = os.getenv("TTS_FALLBACK_VOICE_ID", "").strip()
    tts_fallback_model = os.getenv("TTS_FALLBACK_MODEL", "").strip()
    tts_selfhosted_url = os.getenv("TTS_SELFHOSTED_URL", "").strip()
    tts_sentence_max_chars = _env_int("TTS_SENTENCE_MAX_CHARS", 80)
    tts_sentence_min_chars = _env_int("TTS_SENTENCE_MIN_CHARS", 10)
    tts_cache_enabled = _env_bool("TTS_CACHE_ENABLED", True)
    tts_cache_max_entries = _env_int("TTS_CACHE_MAX_ENTRIES", 500)
    tts_cache_ttl_s = _env_float("TTS_CACHE_TTL_S", 3600.0)
    health_port = _env_int("HEALTH_PORT", 8766)
    max_concurrent_calls = _env_int("MAX_CONCURRENT_CALLS", 100)

    use_custom_tts = tts_provider != "none"
    if use_custom_tts:
        logger.info(
            "Custom TTS enabled: provider=%s voice_id=%s model=%s fallback=%s",
            tts_provider, tts_voice_id or "(default)", tts_model or "(default)", tts_fallback_provider,
        )
        if not tts_api_key and tts_provider not in ("selfhosted", "openai"):
            logger.warning("TTS_PROVIDER=%s but TTS_API_KEY is not set", tts_provider)

    if fs_sample_rate != openai_input_sample_rate and not openai_resample_input:
        logger.warning(
            "FS_SAMPLE_RATE=%d != OPENAI_INPUT_SAMPLE_RATE=%d but OPENAI_RESAMPLE_INPUT is off. "
            "Audio sent to OpenAI will be at the wrong rate. Set OPENAI_RESAMPLE_INPUT=1.",
            fs_sample_rate,
            openai_input_sample_rate,
        )
    if fs_out_sample_rate == fs_sample_rate and openai_output_sample_rate != fs_sample_rate:
        logger.warning(
            "FS_OUT_SAMPLE_RATE=%d equals FS_SAMPLE_RATE=%d but OpenAI outputs %d Hz. "
            "The C module resampler will need to handle %d→%d conversion. "
            "If the C module expects 24 kHz input, set FS_OUT_SAMPLE_RATE=24000.",
            fs_out_sample_rate,
            fs_sample_rate,
            openai_output_sample_rate,
            openai_output_sample_rate,
            fs_sample_rate,
        )
    if fs_sample_rate not in (8000, 16000, 24000, 32000, 44100, 48000):
        logger.warning("Unusual FS_SAMPLE_RATE=%d — verify this matches your FreeSWITCH codec.", fs_sample_rate)

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
        force_commit_ms=force_commit_ms,
        force_response_on_commit=force_response_on_commit,
        response_min_interval_ms=response_min_interval_ms,
        fs_send_json_audio=fs_send_json_audio,
        fs_send_json_handshake=fs_send_json_handshake,
        openai_input_sample_rate=openai_input_sample_rate,
        openai_resample_input=openai_resample_input,
        openai_output_sample_rate=openai_output_sample_rate,
        openai_input_mode=openai_input_mode,
        openai_item_max_buffer_ms=openai_item_max_buffer_ms,
        wss_pem=wss_pem,
        openai_wss_insecure=openai_wss_insecure,
        vad_threshold=vad_threshold,
        vad_prefix_padding_ms=vad_prefix_padding_ms,
        vad_silence_duration_ms=vad_silence_duration_ms,
        temperature=temperature,
        system_instructions=system_instructions,
        tts_provider=tts_provider,
        tts_api_key=tts_api_key,
        tts_voice_id=tts_voice_id,
        tts_model=tts_model,
        tts_fallback_provider=tts_fallback_provider,
        tts_fallback_api_key=tts_fallback_api_key,
        tts_fallback_voice_id=tts_fallback_voice_id,
        tts_fallback_model=tts_fallback_model,
        tts_selfhosted_url=tts_selfhosted_url,
        tts_sentence_max_chars=tts_sentence_max_chars,
        tts_sentence_min_chars=tts_sentence_min_chars,
        tts_cache_enabled=tts_cache_enabled,
        tts_cache_max_entries=tts_cache_max_entries,
        tts_cache_ttl_s=tts_cache_ttl_s,
        health_port=health_port,
        max_concurrent_calls=max_concurrent_calls,
    )
