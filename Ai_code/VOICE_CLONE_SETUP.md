# Custom Voice Setup Guide

## Make the AI IVR Sound Exactly Like YOU

This guide shows how to clone your voice from `ww.wav` and use it as the AI's voice in phone calls. The caller will hear **your exact voice** speaking the AI's responses.

---

## How It Works

```
                   OpenAI Realtime API
                   ┌─────────────────┐
Caller speaks ──►  │ Speech → Text   │  ──► AI reasons ──► Text response
                   │ (VAD + ASR)     │           │
                   └─────────────────┘           │
                                                 ▼
                                          "Hello, how can
                                           I help you?"
                                                 │
                                                 ▼
                                     ┌───────────────────────┐
                                     │ YOUR Cloned Voice TTS │
                                     │ (ElevenLabs/Cartesia) │
                                     └───────────┬───────────┘
                                                 │
                                                 ▼
                                     Audio in YOUR voice ──► Caller hears YOU
```

OpenAI handles the brain (speech recognition, AI reasoning, text generation).
The cloned voice handles the mouth (text-to-speech in YOUR voice).

---

## Step 1: Clone Your Voice on ElevenLabs

### 1a. Create ElevenLabs Account

1. Go to [elevenlabs.io](https://elevenlabs.io) and sign up
2. Free tier gives you 10,000 characters/month (enough for testing)
3. Creator plan ($22/month) recommended for production

### 1b. Upload Your Voice Sample

1. Go to [Voice Lab](https://elevenlabs.io/app/voice-lab)
2. Click **"Add Generative or Cloned Voice"**
3. Select **"Instant Voice Clone"**
4. Upload your `ww.wav` file (9.6 seconds, mono, 48kHz — perfect for cloning)
5. Name it (e.g., "My IVR Voice")
6. Accept the terms → Click **"Add Voice"**

### 1c. Get Your Voice ID

1. In Voice Lab, find your new voice
2. Click the **Settings** icon (⚙️) on the voice card
3. Copy the **Voice ID** (looks like `pNInz6obpgDQGcFmaJgB`)

### 1d. Get Your API Key

1. Go to [Settings → API Keys](https://elevenlabs.io/app/settings/api-keys)
2. Click **"Create API Key"**
3. Copy the key (starts with `sk_...`)

---

## Step 2: Configure the Bridge

Edit your `Ai_code/.env` file:

```bash
# ── Your Custom Voice ──
TTS_PROVIDER=elevenlabs
TTS_API_KEY=sk_your_elevenlabs_api_key_here
TTS_VOICE_ID=pNInz6obpgDQGcFmaJgB    # ← paste your Voice ID here
TTS_MODEL=eleven_turbo_v2_5            # fastest model (~200ms latency)

# ── Failover (if ElevenLabs is down, use OpenAI voice) ──
TTS_FALLBACK_PROVIDER=openai

# ── Caching (0ms for common phrases like "Hello", "Thank you") ──
TTS_CACHE_ENABLED=1
```

---

## Step 3: Install Dependencies

```bash
cd Ai_code
pip install -e ".[hq]"   # includes soxr + aiohttp
```

---

## Step 4: Start the Bridge

```bash
cd Ai_code
python3 main.py
```

You should see:

```
INFO: Custom TTS enabled: provider=elevenlabs voice_id=pNInz6obpgDQGcFmaJgB model=eleven_turbo_v2_5
INFO: Initializing custom TTS: provider=elevenlabs
INFO: TTS engine ready: elevenlabs/eleven_turbo_v2_5
INFO: TTS cache: preloaded 8/8 common phrases
INFO: Custom voice pipeline ready: OpenAI text-only → elevenlabs/eleven_turbo_v2_5 → caller hears YOUR voice
INFO: Listening on ws://0.0.0.0:8765
INFO: OpenAI session: TEXT-ONLY mode (custom TTS will handle audio)
```

---

## Step 5: Test It

Make a call through FreeSWITCH to your IVR number. You should hear the AI responding **in your voice**.

```
fs_cli> uuid_audio_stream <uuid> start ws://127.0.0.1:8765 mono 8000
```

---

## Voice Quality Tips

### For Best Voice Match (100% like you)

| Setting | Value | Why |
|---------|-------|-----|
| `TTS_MODEL` | `eleven_multilingual_v2` | Highest quality (slightly slower) |
| Stability | 0.5 (default) | Higher = more consistent, lower = more expressive |
| Similarity Boost | 0.75 (default) | Higher = closer to your voice sample |
| ElevenLabs plan | Creator+ | Access to Professional Voice Clone (30min+ sample) |

### For Lowest Latency (IVR production)

| Setting | Value | Why |
|---------|-------|-----|
| `TTS_MODEL` | `eleven_turbo_v2_5` | Fastest model (~150-200ms TTFB) |
| `TTS_CACHE_ENABLED` | `1` | 0ms for common phrases |
| `VAD_SILENCE_DURATION_MS` | `200` | Faster turn detection |
| `PLAYOUT_PREBUFFER_MS` | `80` | Less buffering delay |

### Better Voice Clone Quality

Your `ww.wav` is 9.6 seconds. For even better cloning:

1. **Record 1-3 minutes** of you speaking naturally (various tones)
2. **Professional Voice Clone** (ElevenLabs Creator+ plan): upload 30+ minutes for a fine-tuned model that sounds indistinguishable from you
3. Use a quiet room, consistent mic distance
4. Speak in the same tone/energy you want the IVR to use

---

## Alternative Providers

### Cartesia Sonic (Fastest, ~100ms)

```bash
TTS_PROVIDER=cartesia
TTS_API_KEY=your-cartesia-api-key
TTS_VOICE_ID=your-cartesia-voice-id
TTS_MODEL=sonic-2
```

Clone voice at [play.cartesia.ai](https://play.cartesia.ai)

### Self-Hosted (XTTS / Fish Speech — Free, Private)

```bash
TTS_PROVIDER=selfhosted
TTS_SELFHOSTED_URL=http://your-gpu-server:8080
TTS_VOICE_ID=default
```

Run XTTS on your own GPU for zero API cost and full privacy.

### OpenAI TTS-1 (No voice clone, generic voices)

```bash
TTS_PROVIDER=openai
TTS_VOICE_ID=alloy   # or: echo, fable, onyx, nova, shimmer
```

Uses OpenAI's built-in voices (not your voice).

---

## Troubleshooting

### "All TTS engines failed"

- Check `TTS_API_KEY` is set correctly
- Check `TTS_VOICE_ID` exists in your ElevenLabs account
- Check internet connectivity to ElevenLabs API

### Voice doesn't sound like me

- Use `eleven_multilingual_v2` model (higher quality)
- Upload a longer/cleaner voice sample
- Consider Professional Voice Clone (30+ minutes of audio)
- Increase `similarity_boost` in the engine settings

### High latency (slow response)

- Use `eleven_turbo_v2_5` model
- Enable phrase caching (`TTS_CACHE_ENABLED=1`)
- Lower `VAD_SILENCE_DURATION_MS` to 200
- Check network latency to ElevenLabs

### No audio at all

- Check bridge logs for "TTS engine ready" message
- Verify `FS_SEND_JSON_AUDIO=1` in `.env`
- Verify `TTS_PROVIDER` is not `none`
- Check for errors in bridge log output
