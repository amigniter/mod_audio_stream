from __future__ import annotations

import logging
import os

def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper().strip()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Optional: crank only bridge logs to DEBUG without flooding dependencies.
    # Usage: BRIDGE_DEBUG=1 python3 main.py
    if os.getenv("BRIDGE_DEBUG", "0").strip().lower() in ("1", "true", "yes", "y", "on"):
        logging.getLogger("bridge").setLevel(logging.DEBUG)
