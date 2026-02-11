"""Production scaling helpers for the bridge."""
from .health import (
    call_started,
    call_ended,
    set_tts_engine,
    set_max_concurrent,
    get_active_calls,
    get_max_concurrent,
    start_health_server,
)
from .metrics import CallMetrics

__all__ = [
    "call_started",
    "call_ended",
    "set_tts_engine",
    "set_max_concurrent",
    "get_active_calls",
    "get_max_concurrent",
    "start_health_server",
    "CallMetrics",
]
