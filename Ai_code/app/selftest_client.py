from __future__ import annotations

import asyncio
import base64
import json
import os

import websockets


def _pcm_silence_16k_20ms() -> bytes:
    # 16000 samples/sec * 0.02 sec = 320 samples; 2 bytes/sample
    return b"\x00" * (320 * 2)


async def main() -> None:
    url = os.getenv("SELFTEST_URL", "ws://127.0.0.1:8765")
    json_mode = os.getenv("SELFTEST_JSON", "0") == "1"

    async with websockets.connect(url, max_size=None) as ws:
        # send a few frames like FreeSWITCH would
        for _ in range(5):
            frame = _pcm_silence_16k_20ms()
            if json_mode:
                await ws.send(
                    json.dumps(
                        {
                            "type": "streamAudio",
                            "data": {
                                "audioDataType": "raw",
                                "sampleRate": 16000,
                                "audioData": base64.b64encode(frame).decode("ascii"),
                            },
                        }
                    )
                )
            else:
                await ws.send(frame)
            await asyncio.sleep(0.02)

        # read one message back (may be audio JSON or raw)
        msg = await ws.recv()
        if isinstance(msg, bytes):
            print(f"got binary from server: {len(msg)} bytes")
        else:
            try:
                evt = json.loads(msg)
            except Exception:
                print(f"got text from server: {msg[:200]}")
                return
            print(f"got json from server: keys={list(evt.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
