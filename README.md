# mod_audio_stream

**Real-time bidirectional audio streaming module for FreeSWITCH** — capture live call audio over WebSocket and inject AI-generated voice back into the call.

Built for production AI IVR systems: OpenAI Realtime API integration, custom voice (TTS) support, barge-in, high-quality Speex resampling, and architecture for 500+ concurrent calls.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Version](https://img.shields.io/badge/version-1.2.0-green)
![FreeSWITCH](https://img.shields.io/badge/FreeSWITCH-1.10+-orange)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

---

## Table of Contents

- [What This Does](#what-this-does)
- [Architecture](#architecture)
- [Components](#components)
- [Quick Start](#quick-start)
- [Building the C Module](#building-the-c-module)
- [Running the Python Bridge](#running-the-python-bridge)
- [FreeSWITCH Dialplan](#freeswitch-dialplan)
- [Configuration Reference](#configuration-reference)
- [Custom Voice (TTS)](#custom-voice-tts)
- [Scaling to 500+ Calls](#scaling-to-500-calls)
- [Audio Pipeline Details](#audio-pipeline-details)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What This Does

When a caller dials your IVR number (e.g., 1001 → 1234):

1. **FreeSWITCH** answers the call and captures the caller's 8kHz audio
2. **mod_audio_stream** (this C module) streams that audio over WebSocket to the Python bridge
3. **The bridge** resamples 8kHz→24kHz and forwards to OpenAI Realtime API
4. **OpenAI** processes speech, generates an AI response (voice or text)
5. **The bridge** receives AI audio (or synthesizes it via custom TTS), buffers it in a jitter buffer, and sends 20ms PCM frames back
6. **mod_audio_stream** receives the frames, resamples 24kHz→8kHz via Speex, and injects audio into the call
7. **The caller hears the AI voice** in real-time

The entire round-trip takes **300–500ms** from end-of-speech to first AI audio.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FreeSWITCH Process                              │
│                                                                         │
│  Phone ←→ RTP ←→ [8kHz Codec]                                          │
│                       │                                                 │
│               ┌───────┴────────┐                                        │
│               │  media_bug     │                                        │
│               │  (SMBF_READ +  │                                        │
│               │  WRITE_REPLACE)│                                        │
│               └───────┬────────┘                                        │
│                       │                                                 │
│          ┌────────────┼────────────┐                                    │
│          │ READ path  │ WRITE path │                                    │
│          │ (capture)  │ (inject)   │                                    │
│          ▼            │            ▼                                    │
│   [Speex 8→24k]      │    [inject_buffer]                              │
│          │            │         ▲                                       │
│          ▼            │         │                                       │
│   ┌──────────────┐   │   [Speex 24→8k, quality=7]                     │
│   │AudioStreamer  │───┘         │                                       │
│   │(WebSocket)    │◄────────────┘                                       │
│   └──────┬────────┘                                                     │
│          │ ws://                                                         │
└──────────┼──────────────────────────────────────────────────────────────┘
           │
    ┌──────┴──────┐
    │   Python    │
    │   Bridge    │
    │  (app.py)   │
    └──────┬──────┘
           │ wss://
    ┌──────┴──────┐         ┌──────────────┐
    │   OpenAI    │         │  Custom TTS  │
    │  Realtime   │────────▶│  (optional)  │
    │    API      │  text   │  ElevenLabs  │
    └─────────────┘         │  Cartesia    │
                            │  Self-hosted │
                            └──────────────┘
```

---

## Components

### C/C++ Module (`mod_audio_stream`)

| File | Purpose |
|------|---------|
| `mod_audio_stream.c` | FreeSWITCH module entry — registers API, media bug callbacks |
| `mod_audio_stream.h` | Data structures — `private_t`, config, events |
| `audio_streamer_glue.cpp` | Core engine — WebSocket client, `processMessage()`, Speex resample, inject buffer |
| `audio_streamer_glue.h` | C-linkage function declarations |
| `base64.cpp` / `base64.h` | Base64 encode/decode for audio payloads |
| `libs/libwsc/` | Embedded WebSocket client library |

### Python Bridge (`Ai_code/`)

| File | Purpose |
|------|---------|
| `main.py` | Entry point — loads config, starts server |
| `bridge/app.py` | Core engine — `JitterBuffer`, playout loop, bidirectional pumps |
| `bridge/config.py` | Environment/`.env` config loader with validation |
| `bridge/openai_client.py` | OpenAI Realtime WebSocket connection + session config |
| `bridge/resample.py` | Multi-backend resampler (soxr → samplerate → audioop) |
| `bridge/fs_payloads.py` | Builds JSON matching `processMessage()` schema |
| `bridge/audio.py` | PCM frame math, base64, alignment helpers |
| `bridge/tts/` | **Custom TTS subsystem** — pluggable voice engines |
| `bridge/scaling/` | Health checks, metrics for production deployment |

---

## Quick Start

### Prerequisites

- **FreeSWITCH** 1.10+ (with dev headers)
- **CMake** 3.18+
- **libspeexdsp** (Speex resampler)
- **OpenSSL** + **zlib**
- **Python** 3.10+ (for the bridge)
- **OpenAI API key** with Realtime API access

### 1. Build & Install the C Module

```bash

sudo apt-get install -y libfreeswitch-dev libssl-dev zlib1g-dev libspeexdsp-dev cmake

git clone https://github.com/Rahulcse79/mod_audio_stream.git
cd mod_audio_stream

git submodule init && git submodule update
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install
```

Or use the convenience script:

```bash
sudo bash ./build-mod-audio-stream.sh
```

### 2. Start the Python Bridge

```bash
cd Ai_code
python -m venv .venv
source .venv/bin/activate

pip install -e ".[hq]"

cp .env.example .env
python main.py
```

### 3. Connect FreeSWITCH

In your FreeSWITCH dialplan or via `fs_cli`:

```xml
<extension name="ai_ivr">
  <condition field="destination_number" expression="^1234$">
    <action application="answer"/>
    <action application="set" data="STREAM_INJECT_BUFFER_MS=5000"/>
    <action application="set" data="FS_SEND_JSON_AUDIO=true"/>
    <action application="uuid_audio_stream" data="${uuid} start ws://127.0.0.1:8765 mono 8000"/>
    <action application="park"/>
  </condition>
</extension>
```

Or via CLI:

```
uuid_audio_stream <call-uuid> start ws://127.0.0.1:8765 mono 8000
```

---

## Building the C Module

### Build Options

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_LOCAL=ON \                          # Use /usr/local/freeswitch
      -DFREESWITCH_SRC_ROOT=/path/to/freeswitch \ 
      ..
```

| CMake Variable | Description |
|---------------|-------------|
| `ENABLE_FREESWITCH_PKGCONFIG` | Use pkg-config to find FreeSWITCH (default: ON) |
| `FREESWITCH_SRC_ROOT` | Path to FreeSWITCH source tree |
| `FREESWITCH_INCLUDE_DIR` | Path to FreeSWITCH headers |
| `FREESWITCH_LIBRARY` | Path to libfreeswitch.so |
| `FS_MOD_DIR` | Where to install the .so module |
| `ENABLE_LOCAL` | Set PKG_CONFIG_PATH for `/usr/local/freeswitch` |

### Debian Package

```bash
cd build
cpack -G DEB
sudo dpkg -i mod-audio-stream_*.deb
```

---

## Running the Python Bridge

### Installation

```bash
cd Ai_code
pip install -e .         
pip install -e ".[hq]"   
pip install -e ".[alt]"  
```

### Configuration (`.env`)

Create `Ai_code/.env`:

```bash
OPENAI_API_KEY=sk-...
HOST=0.0.0.0
PORT=8765
FS_SAMPLE_RATE=8000          
FS_CHANNELS=1                
FS_FRAME_MS=20               
FS_OUT_SAMPLE_RATE=24000      
OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview
OPENAI_REALTIME_VOICE=alloy   
OPENAI_TEMPERATURE=0.6
OPENAI_RESAMPLE_INPUT=1       
VAD_THRESHOLD=0.5             
VAD_SILENCE_DURATION_MS=300   
VAD_PREFIX_PADDING_MS=300
PLAYOUT_PREBUFFER_MS=100      
FS_SEND_JSON_AUDIO=1          
FS_SEND_JSON_HANDSHAKE=1
```

---

## Configuration Reference

### FreeSWITCH Channel Variables

Set these on the channel before calling `uuid_audio_stream start`:

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAM_FRAME_MS` | `20` | Frame duration (5–60 ms) |
| `STREAM_INJECT_BUFFER_MS` | `5000` | Injection ring buffer size in ms |
| `STREAM_INJECT_MIN_BUFFER_MS` | `0` | Prebuffer before starting playback |
| `STREAM_INJECT_LOG_EVERY_MS` | `1000` | Stats log interval |
| `STREAM_BUFFER_SIZE` | `20` | Capture batch size (must be multiple of 20) |
| `STREAM_MAX_QUEUE_MS` | `0` | Max capture queue before dropping (0 = unlimited) |
| `STREAM_ALLOW_FILE_INJECTION` | `false` | Allow `"file"` field in JSON |
| `STREAM_MAX_AUDIO_BASE64_LEN` | `4MB` | Max base64 audio per message |
| `STREAM_RECONNECT_MAX` | `0` | Max WebSocket reconnect attempts |
| `STREAM_MESSAGE_DEFLATE` | `false` | Enable WebSocket compression |
| `STREAM_SUPPRESS_LOG` | `false` | Reduce logging verbosity |
| `STREAM_HEART_BEAT` | `0` | WebSocket ping interval (seconds) |
| `STREAM_DEBUG_JSON` | `false` | Log full JSON messages |
| `STREAM_EXTRA_HEADERS` | | JSON object of extra WS headers |
| `STREAM_TLS_CA_FILE` | | TLS CA bundle path |
| `STREAM_TLS_KEY_FILE` | | TLS client key |
| `STREAM_TLS_CERT_FILE` | | TLS client cert |
| `STREAM_TLS_DISABLE_HOSTNAME_VALIDATION` | `false` | Disable hostname check |

### Bridge Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | **(required)** | OpenAI API key |
| `HOST` / `PORT` | `0.0.0.0` / `8765` | Bridge listen address |
| `OPENAI_REALTIME_MODEL` | `gpt-4o-realtime-preview` | OpenAI model |
| `OPENAI_REALTIME_VOICE` | `alloy` | OpenAI built-in voice |
| `FS_SAMPLE_RATE` | `8000` | FreeSWITCH PCM rate |
| `FS_CHANNELS` | `1` | Channel count |
| `FS_FRAME_MS` | `20` | Frame duration |
| `FS_OUT_SAMPLE_RATE` | `24000` | Output rate to C module |
| `OPENAI_INPUT_SAMPLE_RATE` | `24000` | Rate sent to OpenAI |
| `OPENAI_OUTPUT_SAMPLE_RATE` | `24000` | Rate from OpenAI |
| `OPENAI_RESAMPLE_INPUT` | `1` | Enable input resampling |
| `PLAYOUT_PREBUFFER_MS` | `100` | Prebuffer before playout |
| `VAD_THRESHOLD` | `0.5` | VAD sensitivity |
| `VAD_SILENCE_DURATION_MS` | `300` | Silence = end of turn |
| `VAD_PREFIX_PADDING_MS` | `300` | Pre-speech padding |
| `OPENAI_TEMPERATURE` | `0.6` | Response randomness |
| `OPENAI_SYSTEM_INSTRUCTIONS` | *(built-in)* | Custom system prompt |
| `OPENAI_INPUT_MODE` | `buffer` | `buffer` or `item` |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## Custom Voice (TTS)

Replace OpenAI's built-in voice with your own cloned/custom voice.

### How It Works

1. OpenAI Realtime is switched to **text-only mode** (`modalities: ["text"]`)
2. OpenAI still handles: speech detection (VAD), transcription, AI reasoning
3. Text responses stream to a **custom TTS engine** (your voice)
4. TTS audio chunks flow into the JitterBuffer → C module → caller

**Result:** The caller hears YOUR voice, not OpenAI's.

### Supported TTS Providers

| Provider | Latency | Voice Clone | Config |
|----------|---------|-------------|--------|
| **ElevenLabs** | ~200ms | Yes (30s sample) | `TTS_PROVIDER=elevenlabs` |
| **Cartesia Sonic** | ~100ms | Yes | `TTS_PROVIDER=cartesia` |
| **Self-hosted** (XTTS, Fish Speech) | ~150ms | Yes | `TTS_PROVIDER=selfhosted` |
| **OpenAI TTS-1** | ~300ms | No | `TTS_PROVIDER=openai` |

### Configuration

Add to your `.env`:

```bash

TTS_PROVIDER=elevenlabs         
TTS_API_KEY=your-elevenlabs-key
TTS_VOICE_ID=your-cloned-voice-id
TTS_MODEL=eleven_turbo_v2_5
TTS_FALLBACK_PROVIDER=openai   
TTS_SENTENCE_MAX_CHARS=80      
TTS_SENTENCE_MIN_CHARS=10      
TTS_CACHE_ENABLED=1
TTS_CACHE_MAX_ENTRIES=500
TTS_CACHE_TTL_S=3600
```

### TTS Architecture

```
LLM text stream ──▶ SentenceBuffer ──▶ TTSCache ──▶ TTS Engine ──▶ JitterBuffer
                     (accumulates       (check      (synthesize     (frame-aligned
                      to sentence        cache       audio via       playout at
                      boundaries)        first)      API/GPU)        real-time rate)
```

See [`Ai_code/ARCHITECTURE_CUSTOM_VOICE.md`](Ai_code/ARCHITECTURE_CUSTOM_VOICE.md) for the full design document.

---

## Scaling to 500+ Calls

### Horizontal Architecture

```
              SIP Proxy (Kamailio)
              ┌───┬───┬───┐
              ▼   ▼   ▼   ▼
         FS-1  FS-2  FS-3  FS-N     ← 170 calls each
              │   │   │   │
              ▼   ▼   ▼   ▼
         Bridge Pool (K8s)           ← 50 async calls each
              │   │   │   │
              ▼   ▼   ▼   ▼
         TTS Service Pool            ← GPU pods or managed API
```

### Sizing Guide

| Component | Per Instance | For 500 Calls |
|-----------|-------------|---------------|
| FreeSWITCH | ~170 calls | 3 instances |
| Bridge (Python asyncio) | ~50 calls | 10 instances |
| TTS (ElevenLabs) | API rate limit | Managed |
| TTS (self-hosted GPU) | ~50 calls/A10G | 10 GPUs |
| OpenAI Realtime | ~100 calls/key | 5 API keys |

### Health Checks

The bridge includes a built-in health server for Kubernetes probes:

```bash
HEALTH_PORT=8766              
MAX_CONCURRENT_CALLS=100      
```

Endpoints:
- `GET /healthz` — Liveness probe
- `GET /readyz` — Readiness probe (checks TTS + call capacity)
- `GET /metrics` — Call statistics

---

## Audio Pipeline Details

### Capture Path (Caller → AI)

```
FreeSWITCH RTP (8kHz PCM16 mono)
    │
    ├──▶ media_bug READ callback
    │       │
    │       ▼
    │    Speex resample 8kHz → 24kHz (quality 7)
    │       │
    │       ▼
    │    WebSocket binary frame ──────▶ Python Bridge
    │                                      │
    │                                      ▼
    │                                   soxr resample 8→24kHz
    │                                      │
    │                                      ▼
    │                                   OpenAI input_audio_buffer.append
    │                                      │
    │                                      ▼
    │                                   OpenAI VAD → transcribe → LLM
```

### Injection Path (AI → Caller)

```
OpenAI audio delta (24kHz PCM16 base64)    OR    Custom TTS (24kHz PCM16)
    │                                                  │
    ▼                                                  ▼
Python Bridge: base64 decode                   TTS streaming chunks
    │                                                  │
    └──────────────────┬───────────────────────────────┘
                       │
                       ▼
               JitterBuffer (unbounded, NEVER drops)
                       │
                       ▼ (20ms tick)
               Playout Loop (1 frame per 20ms, silence fill on underrun)
                       │
                       ▼
               JSON text frame: {"type":"streamAudio","data":{"audioData":"<b64>","sampleRate":24000}}
                       │
                       ▼
               C module processMessage()
                       │
                       ▼
               base64 decode → Speex resample 24→8kHz (quality 7) → inject_buffer
                       │
                       ▼
               media_bug WRITE_REPLACE callback → frame→data replacement
                       │
                       ▼
               Caller hears AI voice
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Speex quality 7** (not default 3) | Quality 3 causes audible aliasing on 24→8kHz. Quality 7 = broadcast-grade sinc filter. Single biggest audio quality improvement. |
| **Unbounded JitterBuffer** | OpenAI sends 3-5× faster than real-time. Bounded buffers drop sentence beginnings → crackling. |
| **Exactly 1 frame per 20ms tick** | Prevents C module buffer overflow. No bursts, no multi-drain. |
| **Silence fill on underrun** | Prevents audio pops/clicks from empty buffer gaps. |
| **Clock starts after prebuffer** | Prevents stale-clock burst catch-up on first audio. |
| **Triple barge-in clear** | Python JitterBuffer + C module inject_buffer + OpenAI response.cancel. |
| **soxr block mode** | soxr streaming mode buffers small frames and returns empty. Block mode always works. |

---

## API Reference

### FreeSWITCH API

```
uuid_audio_stream <uuid> start <ws-uri> [mono|mixed|stereo] [8000|16000|24000|32000|48000] [metadata]
uuid_audio_stream <uuid> stop [text]
uuid_audio_stream <uuid> pause
uuid_audio_stream <uuid> resume
uuid_audio_stream <uuid> send_text <text>
```

### FreeSWITCH Events

| Event | Fired When |
|-------|------------|
| `mod_audio_stream::connect` | WebSocket connected |
| `mod_audio_stream::disconnect` | WebSocket disconnected |
| `mod_audio_stream::error` | Connection error |
| `mod_audio_stream::json` | Non-audio JSON message received |
| `mod_audio_stream::play` | Audio successfully queued for injection |

### JSON Injection Schema

The C module's `processMessage()` accepts:

```json
{
  "type": "streamAudio",
  "data": {
    "audioDataType": "raw",
    "audioData": "<base64 PCM16LE>",
    "sampleRate": 24000,
    "channels": 1
  }
}
```

Barge-in clear command:

```json
{
  "type": "streamAudio",
  "data": {
    "audioDataType": "raw",
    "audioData": "",
    "sampleRate": 24000,
    "channels": 1,
    "clear": true
  }
}
```

---

## Troubleshooting

### No AI audio heard by caller

1. Verify `FS_SEND_JSON_AUDIO=1` in `.env`
2. Check `FS_OUT_SAMPLE_RATE` matches your FreeSWITCH codec (usually `8000` or `24000`)
3. Confirm the media bug has `SMBF_WRITE_REPLACE` (check FreeSWITCH logs for `stream_data_init`)
4. Look for `PUSHBACK queued` logs in FreeSWITCH — this confirms audio is being injected
5. Look for `PUSHBACK consume` logs — this confirms the caller leg is reading from inject_buffer

### Crackling / choppy audio

1. Ensure `PLAYOUT_PREBUFFER_MS=100` (or higher)
2. Check Speex resampler quality is 7 (default in `INJECT_RESAMPLE_QUALITY`)
3. Verify `soxr` is installed for the Python bridge: `pip install soxr numpy`
4. Check playout loop stats for underruns: `Playout: ... underruns=N`

### OpenAI TLS failure (macOS)

```bash
WSS_PEM=./wss.pem   
pip install certifi 
```

### High latency

1. Lower `VAD_SILENCE_DURATION_MS` to `200`–`300` (faster turn detection)
2. Lower `PLAYOUT_PREBUFFER_MS` to `60`–`80` (less buffering)
3. Use `TTS_PROVIDER=cartesia` (fastest managed TTS at ~100ms)
4. Enable `TTS_CACHE_ENABLED=1` (0ms for common phrases)

### Bridge not receiving audio

1. Check FreeSWITCH can reach the bridge: `curl -i ws://127.0.0.1:8765`
2. Look for `FreeSWITCH: first PCM chunk` log in bridge output
3. Verify firewall allows the WebSocket port

---

## Project Structure

```
mod_audio_stream/
├── mod_audio_stream.c          # FreeSWITCH module (API, media bug)
├── mod_audio_stream.h          # Data structures, events, config
├── audio_streamer_glue.cpp     # Core: WebSocket, processMessage, inject, resample
├── audio_streamer_glue.h       # C-linkage declarations
├── base64.cpp / base64.h       # Base64 codec
├── CMakeLists.txt              # Build system
├── build-mod-audio-stream.sh   # One-liner build script
│
├── Ai_code/                    # Python AI bridge
│   ├── main.py                 # Entry point
│   ├── pyproject.toml          # Dependencies
│   ├── bridge/
│   │   ├── app.py              # JitterBuffer, playout loop, pumps
│   │   ├── config.py           # .env config loader
│   │   ├── openai_client.py    # OpenAI Realtime WSS connection
│   │   ├── resample.py         # Multi-backend resampler
│   │   ├── fs_payloads.py      # JSON schema builder
│   │   ├── audio.py            # PCM frame utilities
│   │   ├── logging_utils.py    # Logging setup
│   │   ├── tts/                # Custom voice subsystem
│   │   │   ├── base.py         # Abstract TTS interface
│   │   │   ├── factory.py      # Engine creation + failover
│   │   │   ├── elevenlabs.py   # ElevenLabs streaming
│   │   │   ├── cartesia.py     # Cartesia Sonic streaming
│   │   │   ├── selfhosted.py   # Self-hosted GPU TTS
│   │   │   ├── openai_tts.py   # OpenAI TTS-1 fallback
│   │   │   ├── cache.py        # Phrase caching (LRU)
│   │   │   └── sentence_buffer.py  # Sentence-boundary chunker
│   │   └── scaling/
│   │       ├── health.py       # K8s health endpoints
│   │       └── metrics.py      # Per-call metrics
│   └── tests/
│       ├── test_audio_alignment.py
│       ├── test_fs_payloads.py
│       └── test_resample.py
│
├── libs/libwsc/                # Embedded WebSocket client (C++)
├── cmake/                      # CMake modules
├── debian/                     # Debian packaging
└── build/                      # Build output
```

---

## License

MIT License — see [LICENSE](LICENSE).

Original mod_audio_stream by [amigniter](https://github.com/amigniter/mod_audio_stream).
Custom voice architecture, AI bridge, and production scaling by [Rahulcse79](https://github.com/Rahulcse79).
