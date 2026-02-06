
# OpenAI Realtime ↔ FreeSWITCH Bridge (for `mod_audio_stream`)

This folder contains a small WebSocket server that sits between:

- **FreeSWITCH** using your custom `mod_audio_stream` module
- **OpenAI Realtime** (audio in/out over WSS)

It is tuned for the **20ms** PCM16 framing expectations in your `audio_streamer_glue.cpp`.

## Why this exists

Your custom `mod_audio_stream`:

- streams **binary PCM16** from FreeSWITCH → WebSocket (capture), and
- accepts **JSON text frames** from WebSocket → FreeSWITCH for *pushback / injection* using a strict schema.

This bridge:

1) receives the FreeSWITCH PCM stream
2) forwards it to OpenAI Realtime (`input_audio_buffer.append`)
3) receives OpenAI audio deltas
4) converts/resamples to match FreeSWITCH output settings
5) sends injection frames back to FreeSWITCH **frame-aligned**

## File structure

```
Ai_code/
	main.py                 # entrypoint
	openAi.py               # compatibility wrapper (old name)
	pyproject.toml          # deps (websockets, python-dotenv)
	.env                    # local config (DO NOT COMMIT secrets)
	.env.example            # template
	bridge/
		app.py                # core pumps + playout loop
		config.py             # .env loader/validation
		fs_payloads.py        # builds JSON schema expected by mod_audio_stream
		audio.py              # framing helpers (20ms alignment)
		openai_client.py      # OpenAI WSS connect + session.update
		logging_utils.py      # logging setup
	tests/
		test_fs_payloads.py
		test_audio_alignment.py
```

## The exact JSON schema your mod expects

Your `audio_streamer_glue.cpp::processMessage()` accepts:

```json
{
	"type": "streamAudio",
	"data": {
		"audioDataType": "raw",
		"audioData": "<base64 PCM16>",
		"sampleRate": 16000,
		"channels": 1
	}
}
```

Notes:

- `audioDataType` **must** be `raw`
- `sampleRate` is **required** (integer)
- `channels` may be `1` or `2`
- audio is PCM16LE in `audioData` (base64)
- the mod trims/aligns audio to **20ms** boundaries; this bridge tries to preserve the same contract

## Configuration (.env)

All settings are loaded dynamically from `.env` (and environment variables override `.env`).

Required:

- `OPENAI_API_KEY`

Common:

- `HOST` / `PORT`  
	Where the bridge listens. FreeSWITCH connects to: `ws://HOST:PORT`

- `WSS_PEM`  
	Path to a CA bundle for OpenAI WSS TLS verification (helpful on macOS).

Audio:

- `FS_SAMPLE_RATE` (typically `16000` or `8000`)
- `FS_CHANNELS` (typically `1`)
- `FS_FRAME_MS` (recommended `20`)
- `FS_OUT_SAMPLE_RATE` (optional; defaults to `FS_SAMPLE_RATE`)

Protocol:

- `FS_SEND_JSON_AUDIO=1` to send the JSON injection schema above
- `FS_SEND_JSON_HANDSHAKE=1` (optional debug “start” message)

See `.env.example` for a complete list.

## How to run

Create and edit your local `.env`:

- copy `.env.example` → `.env`
- set `OPENAI_API_KEY`

Then run the bridge (from inside `Ai_code/`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python main.py
```

It will log:

- the WS listen address for FreeSWITCH
- OpenAI connection events

## FreeSWITCH usage (example)

Your module provides the API (from `mod_audio_stream.c`):

```
uuid_audio_stream <uuid> start <ws-uri> [mono|mixed|stereo] [8000|16000] [metadata]
```

Example (bridge on same host, port 8765):

- `ws://127.0.0.1:8765`

Then start streaming for a call UUID:

```
uuid_audio_stream <uuid> start ws://127.0.0.1:8765 mono 16000
```

### Troubleshooting

- If you don’t hear injected audio, confirm:
	- `FS_SEND_JSON_AUDIO=1`
	- `FS_OUT_SAMPLE_RATE` matches what the FreeSWITCH leg actually expects (usually `16000` or `8000`)
	- your `mod_audio_stream` session is in a mode where WRITE_REPLACE is active

- If OpenAI TLS fails on macOS:
	- set `WSS_PEM=./wss.pem` (a CA bundle)

- If logs are too chatty:
	- `LOG_LEVEL=INFO` (or `WARNING`)

## Security notes

- Never commit real API keys. This repo ignores `Ai_code/.env` via `.gitignore`.
- Your `mod_audio_stream` supports a JSON `file` field for injection; consider disabling/restricting it in production to avoid arbitrary file reads.

