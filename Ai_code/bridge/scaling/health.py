"""
Health check HTTP endpoint for load balancers and Kubernetes probes.

Runs alongside the WebSocket server on a separate port.
Reports bridge health, active calls, TTS engine status.

Usage in Kubernetes:
  livenessProbe:
    httpGet:
      path: /healthz
      port: 8766
  readinessProbe:
    httpGet:
      path: /readyz
      port: 8766
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_active_calls: int = 0
_total_calls: int = 0
_start_time: float = 0.0
_tts_engine = None
_max_concurrent_calls: int = 100


def get_active_calls() -> int:
    """Return current active call count (safe to call from other modules)."""
    return _active_calls


def get_max_concurrent() -> int:
    """Return configured max concurrent call limit."""
    return _max_concurrent_calls


def set_tts_engine(engine) -> None:
    global _tts_engine
    _tts_engine = engine


def set_max_concurrent(n: int) -> None:
    global _max_concurrent_calls
    _max_concurrent_calls = n


def call_started() -> None:
    global _active_calls, _total_calls
    _active_calls += 1
    _total_calls += 1


def call_ended() -> None:
    global _active_calls
    _active_calls = max(0, _active_calls - 1)


async def _handle_health(reader, writer):
    """Handle HTTP health check requests."""
    try:
        data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
        request_line = data.decode("utf-8", errors="replace").split("\r\n")[0]
        path = request_line.split(" ")[1] if " " in request_line else "/"
    except Exception:
        writer.close()
        return

    if path == "/healthz":
        body = json.dumps({"status": "ok", "uptime_s": time.monotonic() - _start_time})
        status = "200 OK"

    elif path == "/readyz":
        ready = _active_calls < _max_concurrent_calls
        tts_ok = True
        if _tts_engine:
            try:
                tts_ok = await asyncio.wait_for(_tts_engine.health_check(), timeout=3.0)
            except Exception:
                tts_ok = False

        is_ready = ready and tts_ok
        status = "200 OK" if is_ready else "503 Service Unavailable"
        body = json.dumps({
            "ready": is_ready,
            "active_calls": _active_calls,
            "max_calls": _max_concurrent_calls,
            "tts_healthy": tts_ok,
        })

    elif path == "/metrics":
        body = json.dumps({
            "active_calls": _active_calls,
            "total_calls": _total_calls,
            "uptime_s": round(time.monotonic() - _start_time, 1),
            "max_concurrent_calls": _max_concurrent_calls,
        })
        status = "200 OK"

    else:
        body = json.dumps({"error": "not found"})
        status = "404 Not Found"

    response = (
        f"HTTP/1.1 {status}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
        f"{body}"
    )
    writer.write(response.encode("utf-8"))
    await writer.drain()
    writer.close()


async def start_health_server(port: int = 8766) -> asyncio.AbstractServer:
    """Start health check HTTP server.

    Separate from the WebSocket server so K8s probes
    don't interfere with call handling.
    """
    global _start_time
    _start_time = time.monotonic()

    server = await asyncio.start_server(_handle_health, "0.0.0.0", port)
    logger.info("Health check server listening on :%d", port)
    return server
