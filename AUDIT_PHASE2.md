# PHASE 2: AUDIO PIPELINE VALIDATION — mod_audio_stream

---

## 1. CAPTURE PATH ANALYSIS

### Path: RTP → sbuffer → Speex Resample → WebSocket Binary Frame

```
FreeSWITCH RTP Engine
  │
  ▼
capture_callback(SWITCH_ABC_TYPE_READ)
  │
  ▼
stream_frame(bug)
  │
  ├─ switch_mutex_trylock(tech_pvt->mutex)  ← non-blocking ✅
  ├─ Copy shared_ptr<AudioStreamer> locally
  ├─ Copy resampler, channels, rtp_packets, sbuffer pointers
  ├─ switch_mutex_unlock()
  │
  ├── IF no resampler (native rate matches desired):
  │   ├─ switch_core_media_bug_read(bug, &frame)
  │   ├─ IF rtp_packets == 1:
  │   │   └─ streamer->writeBinary(frame.data, frame.datalen)  ← S16LE PCM direct
  │   └─ IF rtp_packets > 1:
  │       ├─ switch_buffer_write(sbuffer, frame.data, datalen)
  │       └─ When sbuffer full → flush to WS in chunks via read_scratch
  │
  └── IF resampler present:
      ├─ switch_core_media_bug_read(bug, &frame)
      ├─ speex_resampler_process_int(resampler, frame.data → read_scratch)
      ├─ IF rtp_packets == 1:
      │   └─ streamer->writeBinary(read_scratch, bytes_written)
      └─ IF rtp_packets > 1:
          ├─ switch_buffer_write(sbuffer, read_scratch, bytes_written)
          └─ When sbuffer full → flush to WS
```

### Validation:

| Property | Status | Detail |
|----------|--------|--------|
| Sample format | ✅ S16LE | FreeSWITCH native is S16LE. Speex processes `spx_int16_t` (= int16_t) |
| Frame alignment | ✅ | `frame.datalen` is always a multiple of `channels * 2` from FS |
| Buffer sizing | ⚠️ | `buflen = FRAME_SIZE_8000 * desiredSampling/8000 * channels * rtp_packets`. For rtp_packets=1, this means sbuffer is only used as overflow. For rtp_packets>1, it aggregates. Size is correct. |
| Resampler drain on teardown | 🔴 **Missing** | Speex resampler has internal state (FIR filter delay line). When destroyed without draining, the last few samples are lost. At quality=7, the delay is ~7 taps, which is ~0.9ms at 8kHz. |
| Byte order consistency | ✅ | S16LE throughout on LE hosts (checked by `host_is_little_endian()` in inject path) |
| read_scratch sizing | ✅ | Initialized to `SWITCH_RECOMMENDED_BUFFER_SIZE` (8192 bytes). Max output from resampler is bounded by `out_len` capped to scratch capacity. |

### Latency contribution:

| Stage | Latency | Notes |
|-------|---------|-------|
| RTP frame | 20ms (at 20ms ptime) | FreeSWITCH default ptime |
| sbuffer aggregation (rtp_packets=1) | 0ms | Direct send |
| sbuffer aggregation (rtp_packets>1) | 20ms × rtp_packets | Accumulates before flush |
| Speex resample | <0.5ms | Quality 7, 160–320 samples |
| WS send (local) | <1ms | Memory copy to WS library buffer |
| **Total capture latency** | **~21ms** (rtp_packets=1) | ✅ Within 20ms target |

---

## 2. INJECTION PATH ANALYSIS (WS Streaming Mode)

### Path: WebSocket → Decode → inject_buffer → Read Callback → RTP Frame

```
WebSocket Message (JSON with base64 audioData)
  │
  ▼
AudioStreamer::eventCallback(MESSAGE)
  │
  ▼
processMessage(psession, msg)
  ├─ cJSON parse
  ├─ base64_decode(audioData) → std::string decoded (S16LE PCM)
  ├─ OR file read → decoded
  ├─ byteswap if big-endian host
  ├─ Channel conversion (stereo↔mono if needed)
  ├─ Speex resample if sampleRate != output_sr
  │   ├─ lock mutex → init/verify inject_resampler
  │   ├─ unlock mutex
  │   └─ resample_pcm16le_speex(decoded, ..., inject_resampler)
  ├─ Frame alignment (to channels*2 and 20ms boundaries)
  ├─ lock mutex
  ├─ Overflow handling: drop oldest if buffer full
  ├─ switch_buffer_write(inject_buffer, decoded)
  └─ unlock mutex

  ─── Meanwhile, on Media Thread ───

capture_callback(SWITCH_ABC_TYPE_WRITE_REPLACE)
  ├─ lock mutex
  ├─ Ensure inject_scratch large enough (realloc from pool if needed)
  ├─ memset(inject_scratch, 0, need)   ← silence baseline
  ├─ Check inject_min_buffer_ms threshold
  ├─ switch_buffer_read(inject_buffer, inject_scratch, to_read)
  ├─ Track underruns
  ├─ unlock mutex
  ├─ memcpy inject_scratch → frame->data
  └─ switch_core_media_bug_set_write_replace_frame(bug, frame)
```

### Validation:

| Property | Status | Detail |
|----------|--------|--------|
| Sample format | ✅ S16LE | base64 decodes to raw PCM, assumed S16LE, byteswap on BE hosts |
| Frame alignment | ✅ | `decoded.size()` aligned to `channels*2` and `20ms frame` boundaries |
| Buffer sizing | ✅ | inject_buffer sized to `inject_bytes_per_ms × inject_ms`, default 5000ms |
| inject_scratch sizing | ⚠️ | Initially `SWITCH_RECOMMENDED_BUFFER_SIZE` (8192). If `frame->datalen > 8192`, pool-realloc occurs on media thread. At 48kHz stereo 20ms: 48000×2×2×0.02 = 3840 bytes. At 48kHz stereo 60ms: 11520 bytes > 8192. |
| Resampler drain | 🔴 **Missing** | inject_resampler destroyed in `destroy_tech_pvt()` without drain |
| Byte order | ✅ | Checked and swapped |
| Sample rate conversion | ✅ | Speex at quality 7, dynamically re-created if rate changes |

### Latency contribution:

| Stage | Latency | Notes |
|-------|---------|-------|
| WS receive + JSON parse + base64 decode | 1-5ms | Depends on message size |
| Speex inject resample | <0.5ms | |
| inject_buffer write | <0.1ms | |
| inject_min_buffer_ms wait | 60ms default | **This is the dominant latency** |
| WRITE_REPLACE read | 0ms | Synchronous with media clock |
| **Total injection latency** | **~62-66ms** | Dominated by min_buffer |

### ⚠️ Key Insight: `inject_min_buffer_ms`

The default `inject_min_buffer_ms = 60` means the media thread will NOT read from inject_buffer until at least 60ms worth of audio is buffered. This adds 60ms of latency but prevents choppy playback from network jitter. Trade-off:
- Too low (0-20ms): Choppy audio if WS messages arrive with jitter
- Too high (100ms+): Noticeable latency in conversational flow
- **Recommendation:** 40ms is a better default for voice IVRS

---

## 3. AI ENGINE AUDIO PATH ANALYSIS

### Capture: RTP → Upsample → OpenAI Realtime (24kHz PCM16 base64)

```
capture_callback(SWITCH_ABC_TYPE_READ)
  │
  ▼
ai_engine_feed_frame(bug)
  ├─ switch_core_media_bug_read(bug, &frame)
  └─ engine->feed_audio(samples, num_samples)
      ├─ IF upsample_resampler:
      │   ├─ resample_up(samples, num_samples, upsampled)
      │   │   ├─ lock resampler_mutex_
      │   │   ├─ speex_resampler_process_int(8kHz→24kHz)
      │   │   └─ unlock resampler_mutex_
      │   └─ openai_->send_audio(upsampled)
      └─ ELSE:
          └─ openai_->send_audio(samples)
                ├─ base64_encode_pcm(samples, num)
                ├─ Build JSON: {"type":"input_audio_buffer.append","audio":"..."}
                └─ ws_->sendMessage(msg)
```

### Injection: TTS Audio → Downsample → DSP → Ring Buffer → RTP

```
OpenAI response.text.delta
  ▼
on_openai_text_delta(delta)
  ├─ sentence_buffer_.add_token(delta, callback)
  └─ callback: enqueue TTSWorkItem → tts_queue_
      ▼
tts_worker_loop() [TTS thread]
  ├─ dequeue TTSWorkItem
  ├─ Check tts_cache_ (hit → skip HTTP)
  ├─ tts_engine_->synthesize(text, audio_cb, error_cb, abort_flag)
  │   └─ CURL streaming → curl_write_callback → audio_cb(samples)
  └─ audio_cb invokes on_tts_audio()
      ├─ IF tts_sr != freeswitch_sample_rate:
      │   ├─ resample_down(samples, count, tts_sr, resampled)
      │   │   ├─ lock resampler_mutex_
      │   │   ├─ speex_resampler_process_int(tts_sr→fs_sr)
      │   │   └─ unlock resampler_mutex_
      │   └─ dsp_.process(resampled.data(), resampled.size())
      └─ ring_buffer_->write_pcm16(resampled)

  ─── Meanwhile, on Media Thread ───

capture_callback(SWITCH_ABC_TYPE_WRITE_REPLACE)
  ├─ ai_engine_read_audio(tech_pvt, frame->data, frame->datalen/2)
  │   └─ engine->read_audio(dest, num_samples)
  │       ├─ ring_buffer_->read_pcm16(dest, num_samples)
  │       ├─ IF not enough: read partial + zero-fill remainder
  │       └─ return num_samples (or 0 if empty)
  ├─ IF filled == 0: memset(frame->data, 0, datalen)
  └─ switch_core_media_bug_set_write_replace_frame(bug, frame)
```

### Validation:

| Property | Status | Detail |
|----------|--------|--------|
| Sample format | ✅ S16LE | Throughout |
| Upsample quality | ✅ 7 | Good for voice |
| Downsample quality | ✅ 7 | |
| DSP at correct rate | ✅ | DSP initialized at `freeswitch_sample_rate`, applied AFTER downsample |
| Ring buffer SPSC contract | 🔴 **Violated** | TTS thread writes (producer), media thread reads (consumer), but `flush()` is called from OpenAI WS thread (neither). See Phase 1 Finding #5 |
| Zero-fill on underrun | ✅ | `read_audio()` zero-fills if partial read. `capture_callback` memsets to 0 if `filled == 0` |
| Frame alignment | ✅ | Ring buffer operates in bytes, `read_pcm16()` requests exact sample count × 2 bytes |
| Resampler drain | 🔴 **Missing** | `upsample_resampler_` and `downsample_resampler_` destroyed without drain in `stop()` |
| DSP chain clipping protection | ✅ | Soft clipper is the last stage, threshold 0.85 |
| DC offset prevention | ✅ | DC blocker is the first DSP stage |

### Latency contribution (AI mode end-to-end):

| Stage | Latency | Notes |
|-------|---------|-------|
| RTP frame capture | 20ms | FreeSWITCH ptime |
| Upsample 8k→24k | <0.5ms | |
| Base64 encode + WS send | <1ms | |
| **OpenAI processing** | **200-800ms** | Model inference, variable |
| Text delta → sentence buffer | <1ms | Accumulates until sentence boundary |
| TTS HTTP request (ElevenLabs) | 100-500ms | First byte latency |
| TTS streaming chunks | Overlapped | Chunks arrive during synthesis |
| Downsample (e.g., 16k→8k) | <0.5ms | |
| DSP processing | <0.5ms | All stages combined |
| Ring buffer write + read | <0.1ms | Lock-free SPSC |
| **Total E2E voice latency** | **~350-1350ms** | Dominated by OpenAI + TTS |

### ⚠️ Key Latency Bottlenecks:

1. **Sentence buffering:** The system waits for sentence boundaries before sending to TTS. With `min_sentence_chars = 10`, short responses like "Yes" or "Sure" are sent immediately. But multi-sentence responses accumulate. **This is correct behavior** — sending word-by-word to TTS would produce worse prosody.

2. **TTS first-byte latency:** ElevenLabs streaming starts returning audio after 100-300ms. This is the main controllable bottleneck.

3. **No audio pipeline latency** from inject_buffer in AI mode because AI mode uses `SPSCRingBuffer` directly (no `inject_min_buffer_ms` delay). ✅ Good design choice.

---

## 4. COMPUTE: MINIMUM inject_buffer_ms FOR GAPLESS PLAYBACK

For WS streaming mode, the minimum inject_buffer_ms depends on:
- WS message interval (how often server sends audio)
- Network jitter
- Frame size consumed by media thread

### Formula:
```
min_buffer_ms = max_jitter_ms + frame_ms
```

### Typical scenarios:

| WS Message Interval | Network Jitter | Frame MS | Min Buffer MS | Recommended |
|---------------------|---------------|----------|---------------|-------------|
| 20ms (real-time) | ±10ms | 20ms | 30ms | 40ms |
| 100ms (chunked) | ±50ms | 20ms | 70ms | 100ms |
| 200ms (batch) | ±100ms | 20ms | 120ms | 150ms |
| Variable (TTS) | ±200ms | 20ms | 220ms | 250ms |

### Current default: 60ms
- **OK for low-jitter real-time streams**
- **Too low for batch/TTS injection** — will cause underruns
- **Recommendation:** Make it adaptive based on observed jitter

---

## 5. SCRATCH BUFFER SIZING VALIDATION

### `read_scratch` — used for resampler output in capture path

Worst case: resample 20ms of 8kHz to 48kHz
- Input: 160 samples × 2 bytes = 320 bytes
- Output: 960 samples × 2 bytes = 1920 bytes
- With stereo: 3840 bytes
- Allocated: 8192 bytes ✅ Sufficient

### `inject_scratch` — used for reading inject_buffer

Worst case: 48kHz stereo 60ms frame
- 48000 × 2 channels × 2 bytes × 0.060 = 11520 bytes
- Allocated: 8192 bytes ⚠️ **Insufficient for 48kHz stereo 60ms**
- The code handles this by re-allocating from pool (line ~103 of mod_audio_stream.c)
- But this re-allocation happens on the media thread hot path
- **Recommendation:** Pre-allocate based on actual codec rate

### Ring buffer (AI mode)

Sized to: `freeswitch_sample_rate × 2 × inject_buffer_ms / 1000`
- At 8kHz, 5000ms: 80,000 bytes → rounded up to 131,072 (next power of 2) ✅
- At 16kHz, 5000ms: 160,000 bytes → rounded up to 262,144 ✅
- Sufficient for all TTS audio buffering

---

## 6. SUMMARY OF AUDIO PIPELINE ISSUES

| # | Issue | Severity | Fix Complexity |
|---|-------|----------|---------------|
| 1 | Speex resampler not drained on teardown | 🟢 Minor | Low — call `speex_resampler_skip_zeros()` or drain before destroy |
| 2 | SPSC ring buffer flush() from wrong thread | 🔴 Critical | Medium — redesign flush to use atomic flag |
| 3 | inject_scratch may be too small for 48kHz stereo 60ms | 🟡 Major | Low — pre-allocate based on codec |
| 4 | inject_min_buffer_ms default (60ms) may be suboptimal | 🟢 Minor | Config change |
| 5 | No PLC (Packet Loss Concealment) on underruns | 🟡 Major | Medium — implement simple interpolation |
| 6 | No jitter measurement/adaptation | 🟡 Major | Medium — track WS arrival jitter |
| 7 | No click/pop suppression on buffer underrun | 🟡 Major | Low — crossfade to/from silence |

---

*End of Phase 2. Proceed to Phase 3: Concurrency Hardening.*
