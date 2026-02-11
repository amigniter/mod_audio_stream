# Custom Voice AI IVR — Production Architecture

## Date: February 2026 | Target: 500+ Concurrent Calls | Latency: <400ms E2E

---

## 1. Problem Statement

**Current state:** Caller → FreeSWITCH → mod_audio_stream → Python bridge → OpenAI Realtime API.
OpenAI returns **both** the AI text response AND the synthesized voice (alloy/echo/etc).
The voice is OpenAI's built-in voice — **you cannot change it**.

**Goal:** Replace OpenAI's voice with YOUR custom voice while keeping:
- Real-time latency (IVR-grade, <400ms first-byte)
- 500+ concurrent calls
- Natural, consistent voice quality

---

## 2. Architecture: Text-Mode OpenAI + Streaming Custom TTS

### Key Insight
OpenAI Realtime API supports `modalities: ["text"]` — it returns **only text** (no audio).
This is 3-5x faster than audio mode because OpenAI doesn't need to synthesize voice.
We then pipe that text through our own streaming TTS engine.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        PRODUCTION CALL FLOW                                  │
│                                                                              │
│  Phone ──RTP──▶ FreeSWITCH ──mod_audio_stream──▶ Python Bridge (per-call)   │
│                                                        │                     │
│                                                        │ 8kHz PCM            │
│                                                        ▼                     │
│                                              ┌─────────────────┐             │
│                                              │  ASR / STT      │             │
│                                              │  (Whisper/       │             │
│                                              │   Deepgram/      │             │
│                                              │   OpenAI VAD)    │             │
│                                              └────────┬────────┘             │
│                                                       │ text                 │
│                                                       ▼                      │
│                                              ┌─────────────────┐             │
│                                              │  LLM Engine     │             │
│                                              │  OpenAI GPT-4o  │             │
│                                              │  text-only mode │             │
│                                              │  (streaming)    │             │
│                                              └────────┬────────┘             │
│                                                       │ text chunks          │
│                                                       ▼                      │
│                                              ┌─────────────────┐             │
│                                              │  Custom TTS     │             │
│                                              │  (Your Voice)   │             │
│                                              │  Streaming mode │             │
│                                              └────────┬────────┘             │
│                                                       │ PCM audio chunks     │
│                                                       ▼                      │
│                                              ┌─────────────────┐             │
│                                              │  JitterBuffer   │             │
│                                              │  + Playout Loop │             │
│                                              └────────┬────────┘             │
│                                                       │ 20ms frames          │
│                                                       ▼                      │
│  Phone ◀──RTP──◀ FreeSWITCH ◀──inject_buffer──◀ C Module (Speex resample)   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Why This Architecture Wins

| Aspect | OpenAI Voice (current) | Text + Custom TTS (proposed) |
|--------|----------------------|-------------------------------|
| Voice control | None — fixed voices | Full — your cloned voice |
| LLM latency | ~300ms (generates audio) | ~100ms (text only, 3x faster) |
| TTS latency | N/A (bundled) | ~150ms first chunk (streaming) |
| Total first-byte | ~400-600ms | ~300-450ms |
| Cost per minute | $0.06 (audio mode) | $0.01 (text) + $0.01 (TTS) |
| Scalability | OpenAI rate limits | You control TTS scaling |

---

## 3. Component Design

### 3A. ASR (Automatic Speech Recognition)

**Two strategies:**

| Strategy | Latency | Quality | Cost | For 500 calls |
|----------|---------|---------|------|---------------|
| **Keep OpenAI VAD** | 0ms extra | Excellent | Free (bundled) | Already works |
| **Dedicated Whisper** | +50-100ms | Excellent | Self-hosted | Full control |
| **Deepgram Streaming** | +80ms | Very good | $0.0043/min | Easiest to scale |

**Recommendation:** Keep OpenAI's server_vad in text-only mode. It still
detects speech start/stop and sends `input_audio_buffer.speech_started` events.
This means your existing barge-in logic works unchanged.

### 3B. LLM (Language Model)

Switch OpenAI Realtime session to **text-only output**:

```python
"session": {
    "modalities": ["text"],          # ← KEY CHANGE: no "audio"
    "input_audio_format": "pcm16",   # Still accept audio input
    "turn_detection": { ... },        # VAD still works
    "input_audio_transcription": { "model": "gpt-4o-mini-transcribe" },
}
```

OpenAI still listens to audio (VAD + transcription) but responds with **text only**.
Text tokens stream ~3-5x faster than audio generation.

### 3C. Custom TTS Engine (The Core Change)

**Tier 1 — Managed Services (fastest to deploy):**

| Service | Latency | Voice Clone | Quality | Price |
|---------|---------|-------------|---------|-------|
| **ElevenLabs** | ~200ms | Yes (30s sample) | Excellent | $0.018/1K chars |
| **PlayHT** | ~150ms | Yes | Very good | $0.015/1K chars |
| **Cartesia Sonic** | ~100ms | Yes | Excellent | $0.01/1K chars |
| **Azure Neural TTS** | ~150ms | Yes (custom neural) | Excellent | $0.016/1K chars |

**Tier 2 — Self-Hosted (full control, lowest per-call cost at scale):**

| Engine | Latency | Voice Clone | GPU Needed | Quality |
|--------|---------|-------------|------------|---------|
| **Coqui XTTS v2** | ~200ms | Yes (6s sample) | 1× A10G per ~50 calls | Good |
| **Fish Speech** | ~150ms | Yes | 1× A10G per ~30 calls | Very good |
| **StyleTTS2** | ~100ms | Yes (fine-tune) | 1× A10G per ~80 calls | Excellent |
| **VITS2** | ~50ms | Fine-tune only | 1× T4 per ~100 calls | Good |

**Recommendation for 500+ calls:**
- **Start:** ElevenLabs or Cartesia (managed, instant voice clone, streaming API)
- **Scale:** Self-host XTTS v2 or Fish Speech on GPU cluster (cost drops 10x)

### 3D. Streaming TTS Protocol

The TTS must support **streaming** — send text chunks, get audio chunks back
incrementally. This is critical for latency.

```
LLM stream:  "Hello," → "how" → "can" → "I" → "help" → "you" → "today?"
                |
                ▼ (sentence boundary or ~50 chars)
TTS stream:  [PCM chunk 1: "Hello, how can I"] ──→ JitterBuffer
             [PCM chunk 2: "help you today?"]  ──→ JitterBuffer
                                                      │
                                                      ▼
                                              20ms frame playout
```

**Sentence-boundary chunking** is the key latency optimization:
- Buffer LLM text tokens until sentence end (. ? ! , or ~60 chars)
- Send that sentence to TTS as one request
- TTS returns streaming audio chunks
- Audio goes directly into JitterBuffer

This gives you ~200ms first-audio while the LLM is still generating the rest.

---

## 4. Scaling to 500+ Concurrent Calls

### 4A. Horizontal Architecture

```
                    ┌─────────────┐
                    │  SIP Proxy  │
                    │  (Kamailio) │
                    └──────┬──────┘
                           │ SIP
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │ FreeSWITCH-1 │ │ FreeSWITCH-2 │ │ FreeSWITCH-N │
     │ (170 calls)  │ │ (170 calls)  │ │ (170 calls)  │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │ ws://          │ ws://          │ ws://
            ▼                ▼                ▼
     ┌─────────────────────────────────────────────┐
     │          Bridge Pool (K8s / Docker)          │
     │                                              │
     │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
     │  │ Bridge-1 │  │ Bridge-2 │  │ Bridge-N │  │
     │  │ (50 calls│  │ (50 calls│  │ (50 calls│  │
     │  │  async)  │  │  async)  │  │  async)  │  │
     │  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
     │       │              │              │        │
     └───────┼──────────────┼──────────────┼────────┘
             │              │              │
             ▼              ▼              ▼
     ┌─────────────────────────────────────────────┐
     │           TTS Service Pool                   │
     │                                              │
     │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
     │  │ TTS GPU-1│  │ TTS GPU-2│  │ TTS GPU-N│  │
     │  │ A10G     │  │ A10G     │  │ A10G     │  │
     │  │ ~50 conc │  │ ~50 conc │  │ ~50 conc │  │
     │  └──────────┘  └──────────┘  └──────────┘  │
     │                                              │
     │  OR: ElevenLabs/Cartesia API (managed)       │
     └─────────────────────────────────────────────┘
```

### 4B. Sizing Table

| Component | Per Instance | For 500 Calls | Notes |
|-----------|-------------|---------------|-------|
| FreeSWITCH | ~170 calls | 3 instances | CPU-bound, 4 cores each |
| Bridge (Python) | ~50 async calls | 10 instances | Asyncio, 2 cores each |
| TTS (managed) | API rate limit | Pay per char | Simplest |
| TTS (self-hosted) | ~50 calls/GPU | 10× A10G GPUs | Lowest cost at scale |
| OpenAI API | ~100 calls/key | 5 API keys | Text mode = lower rate |
| Redis (session) | 10K+ sessions | 1 instance | For state/caching |

### 4C. Stateless Design

Each bridge instance is **stateless** — all per-call state lives in the WebSocket
connection and the asyncio tasks. No shared database during a call.

Between calls:
- Redis for voice profile cache (which voice ID for which caller)
- Redis for TTS audio cache (cache common phrases)
- PostgreSQL for call logs/analytics (async write)

---

## 5. Low-Latency Design Strategies

### 5A. Sentence-Boundary Text Buffering

Don't send every LLM token to TTS individually (would be choppy).
Don't wait for the full response (would be slow).
**Buffer to sentence boundaries** — optimal chunk size for natural speech.

```python
# Accumulate tokens until sentence boundary
buffer = ""
for token in llm_stream:
    buffer += token
    if _is_sentence_boundary(buffer):
        tts_stream.send(buffer)  # Non-blocking
        buffer = ""
# Flush remainder
if buffer:
    tts_stream.send(buffer)
```

### 5B. Parallel Pipeline (overlap LLM + TTS)

```
Time ──────────────────────────────────────────────────▶

LLM:    [generating sentence 1] [generating sentence 2] [sentence 3]
                     │                    │
TTS:                 ▼                    ▼
         [synthesizing sent 1]  [synthesizing sent 2]
                │                    │
Audio:          ▼                    ▼
         [playing sent 1]─────[playing sent 2]────────

Total latency = LLM first sentence + TTS first chunk ≈ 200ms + 150ms = 350ms
```

### 5C. Phrase Caching

Cache TTS output for common IVR phrases:
- "Thank you for calling"
- "Please hold while I transfer you"
- "Is there anything else I can help with?"
- "Let me look that up for you"

Cache hit = 0ms TTS latency. Reduces GPU load by ~30%.

### 5D. TTS Warm-up

Pre-load the voice model on bridge startup. First call should not pay
model-loading cost. Keep TTS connections warm with periodic health checks.

---

## 6. File Structure (New + Modified)

```
Ai_code/
├── bridge/
│   ├── __init__.py
│   ├── app.py                   ← MODIFIED: orchestrates text-mode + TTS
│   ├── audio.py                 ← unchanged
│   ├── config.py                ← MODIFIED: add TTS config fields
│   ├── fs_payloads.py           ← unchanged
│   ├── logging_utils.py         ← unchanged
│   ├── openai_client.py         ← MODIFIED: text-only session mode
│   ├── resample.py              ← unchanged
│   │
│   ├── tts/                     ← NEW: Custom TTS subsystem
│   │   ├── __init__.py
│   │   ├── base.py              ← Abstract TTS interface
│   │   ├── elevenlabs.py        ← ElevenLabs streaming TTS
│   │   ├── cartesia.py          ← Cartesia Sonic streaming TTS
│   │   ├── selfhosted.py        ← Self-hosted TTS (XTTS/Fish)
│   │   ├── openai_tts.py        ← OpenAI TTS-1 as fallback
│   │   ├── cache.py             ← Phrase caching layer
│   │   └── sentence_buffer.py   ← Sentence-boundary text chunker
│   │
│   └── scaling/                 ← NEW: Production scaling helpers
│       ├── __init__.py
│       ├── health.py            ← Health check endpoints
│       └── metrics.py           ← Prometheus metrics
│
├── main.py                      ← MODIFIED: TTS config loading
└── pyproject.toml               ← MODIFIED: new dependencies
```

---

## 7. Trade-offs & Recommendations

### Voice Quality vs Latency

| Setting | First-byte Latency | Voice Quality |
|---------|-------------------|---------------|
| Small TTS chunks (20 chars) | ~100ms | Choppy, unnatural rhythm |
| Sentence chunks (60-120 chars) | ~250ms | Natural, smooth |
| Full response | ~1-3s | Perfect but too slow |
| **Recommended: sentence boundary** | **~200ms** | **Natural** |

### Managed vs Self-Hosted TTS

| Aspect | Managed (ElevenLabs) | Self-Hosted (XTTS) |
|--------|---------------------|---------------------|
| Setup time | 1 hour | 2-3 weeks |
| Per-minute cost | ~$0.018/1K chars | ~$0.002/1K chars |
| 500-call monthly cost | ~$15K | ~$3K (GPU infra) |
| Voice quality | Excellent | Good-Excellent |
| Latency control | Limited | Full |
| Failover | Provider SLA | You manage |

**Recommendation:** Start with ElevenLabs/Cartesia for validation.
Migrate to self-hosted once you hit 200+ sustained concurrent calls.

### Failover Strategy

```
Primary TTS ──fail──▶ Secondary TTS ──fail──▶ OpenAI TTS-1 fallback
(ElevenLabs)          (Cartesia)               (always available)
```

Each bridge instance tries Primary → Secondary → Fallback with <100ms timeout.
The caller never hears silence — worst case they hear a different voice for one response.
