from __future__ import annotations

import asyncio
import hashlib
import logging

from bridge.config import load_config
from bridge.logging_utils import setup_logging
from bridge.app import run_server


logger = logging.getLogger(__name__)


def _mask_secret(s: str, prefix: int = 8, suffix: int = 6) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if len(s) <= prefix + suffix:
        return "*" * len(s)
    return f"{s[:prefix]}...{s[-suffix:]}"


def _sha256_prefix(s: str, n: int = 12) -> str:
    if not s:
        return ""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def _log_config(cfg) -> None:
    api_key = getattr(cfg, "openai_api_key", "")
    logging.getLogger("bridge").info(
        "Config: host=%s port=%s model=%s voice=%s fs_sample_rate=%s fs_out_sample_rate=%s fs_channels=%s fs_frame_ms=%s "
        "force_commit_ms=%s(vad_driven) response_min_interval_ms=%s fs_send_json_audio=%s wss_pem=%s openai_wss_insecure=%s "
        "OPENAI_API_KEY_MASKED=%s OPENAI_API_KEY_SHA256_12=%s",
        getattr(cfg, "host", ""),
        getattr(cfg, "port", ""),
        getattr(cfg, "model", ""),
        getattr(cfg, "voice", ""),
        getattr(cfg, "fs_sample_rate", ""),
        getattr(cfg, "fs_out_sample_rate", ""),
        getattr(cfg, "fs_channels", ""),
        getattr(cfg, "fs_frame_ms", ""),
        getattr(cfg, "force_commit_ms", ""),
        getattr(cfg, "response_min_interval_ms", ""),
        getattr(cfg, "fs_send_json_audio", ""),
        getattr(cfg, "wss_pem", ""),
        getattr(cfg, "openai_wss_insecure", ""),
        _mask_secret(api_key),
        _sha256_prefix(api_key),
    )


def main() -> None:
    setup_logging()
    cfg = load_config()
    # Use bridge logger namespace so it's visible alongside the rest of the bridge logs.
    logging.getLogger("bridge").info("=== mod-audio-stream bridge starting (config dump below, secrets masked) ===")
    _log_config(cfg)
    asyncio.run(run_server(cfg))


if __name__ == "__main__":
    main()
