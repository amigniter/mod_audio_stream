# Ai_code (FreeSWITCH ↔ OpenAI Realtime bridge)

This folder contains a Python WebSocket server that `mod_audio_stream` can connect to.

- FreeSWITCH → Python: raw PCM16 frames (binary) are received and forwarded to OpenAI Realtime as `input_audio_buffer.append`.
- OpenAI → Python → FreeSWITCH: OpenAI PCM16 audio deltas are buffered, aligned to 20ms frames, and sent back to FreeSWITCH.

## Files

- `ai_check.py` – current single-file bridge (kept for convenience)
- `wss.pem` – optional CA bundle for outbound WSS verification (must NOT contain a private key)
- `requirements.txt` – Python deps
- `.env.example` – environment variables template

## Message compatibility notes

`mod_audio_stream` can accept audio to play in two common ways:

1) **JSON play message** (recommended; matches repo `README.md`):

```json
{
  "type": "streamAudio",
  "data": {
    "audioDataType": "raw",
    "sampleRate": 16000,
    "audioData": "<base64 PCM16>"
  }
}
```

2) **Raw binary PCM16 frames** (some builds / configs).

This bridge can send either. Control it with `FS_SEND_JSON_AUDIO=1|0`.

## Run

Install deps, then run `ai_check.py`.

If you use the included `wss.pem` as a CA bundle, set:

- `WSS_PEM=./wss.pem`

You must set:

- `OPENAI_API_KEY`

