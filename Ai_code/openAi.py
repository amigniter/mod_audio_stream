"""Compatibility wrapper.

This repo originally shipped a single-file bridge here.
The productionized implementation now lives in:

    - bridge/ (package)
    - main.py (entrypoint)

Run:
    python main.py

This file remains so existing docs/scripts don't break.
"""

from __future__ import annotations

from main import main

if __name__ == "__main__":
    main()