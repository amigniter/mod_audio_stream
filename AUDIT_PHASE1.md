# PHASE 1: FULL AUDIT — mod_audio_stream

**Auditor:** AI System Architect  
**Date:** 2026-02-13  
**Scope:** Complete codebase — all `.c`, `.cpp`, `.h` files  
**Method:** Manual trace of every allocation, mutex pair, atomic, lifecycle path, thread boundary, stale pointer risk

---

## TABLE OF CONTENTS

1. [File-by-File Allocation Trace](#1-file-by-file-allocation-trace)
2. [Mutex Lock/Unlock Pair Trace](#2-mutex-lockunlock-pair-trace)
3. [Atomic Operations Trace](#3-atomic-operations-trace)
4. [Full Lifecycle Trace](#4-full-lifecycle-trace)
5. [Thread Boundary Analysis](#5-thread-boundary-analysis)
6. [Findings: Numbered & Classified](#6-findings-numbered--classified)
7. [Answers to Architecture Questions Q1–Q12](#7-answers-to-architecture-questions-q1q12)
8. [Risk Assessment](#8-risk-assessment)
9. [Recommended Code Changes](#9-recommended-code-changes)

---

## 1. FILE-BY-FILE ALLOCATION TRACE

### 1.1 `audio_streamer_glue.cpp` — `stream_data_init()`

| # | Allocation | Type | Free Location | Status |
|---|-----------|------|---------------|--------|
| 1 | `switch_core_session_alloc(session, sizeof(private_t))` | Pool | Auto (session pool) | ✅ OK |
| 2 | `switch_mutex_init(&tech_pvt->mutex, ...)` | Pool | Auto (session pool) | ✅ OK |
| 3 | `switch_buffer_create(pool, &tech_pvt->sbuffer, ...)` | Pool | `destroy_tech_pvt` sets to `nullptr` | ⚠️ See Finding #1 |
| 4 | `switch_buffer_create(pool, &tech_pvt->inject_buffer, ...)` | Pool | `destroy_tech_pvt` sets to `nullptr` | ⚠️ See Finding #1 |
| 5 | `switch_core_session_alloc(session, read_scratch_len)` | Pool | Auto (session pool) | ✅ OK |
| 6 | `switch_core_session_alloc(session, inject_scratch_len)` | Pool | Auto (session pool) | ✅ OK |
| 7 | `speex_resampler_init(channels, sampling, desired, ...)` | **Heap** | `destroy_tech_pvt()` | ✅ OK |
| 8 | `tech_pvt->inject_resampler` — lazy init in `processMessage()` | **Heap** | `destroy_tech_pvt()` | ✅ OK |
| 9 | `new std::shared_ptr<AudioStreamer>(sp)` | **Heap** | `stream_session_cleanup` — `delete sp_wrap` | ✅ OK |
| 10 | `AudioStreamer::create()` → `shared_ptr` | **Heap** (ref-counted) | Last `shared_ptr` release | ⚠️ See Finding #2 |

### 1.2 `audio_streamer_glue.cpp` — `stream_session_cleanup()`

| Step | Action | Notes |
|------|--------|-------|
| 1 | Lock mutex, set `cleanup_started = TRUE` | Guard against double cleanup |
| 2 | Set `channel private = nullptr` | ✅ |
| 3 | Save `sp_wrap`, set `pAudioStreamer = nullptr` | ✅ |
| 4 | Unlock mutex | ✅ |
| 5 | `switch_core_media_bug_remove(session, &bug)` if not closing | ✅ |
| 6 | `delete sp_wrap` | Releases one ref-count | ✅ |
| 7 | `streamer->deleteFiles()` | ✅ |
| 8 | `streamer->markCleanedUp()` | Sets `m_cleanedUp` atomic | ✅ |
| 9 | `streamer->disconnect()` | ✅ |
| 10 | `destroy_tech_pvt(tech_pvt)` | Frees resamplers, nulls buffers | ✅ |

### 1.3 `ai_engine_glue.cpp` — `ai_engine_session_init()`

| # | Allocation | Type | Free Location | Status |
|---|-----------|------|---------------|--------|
| 1 | `switch_core_session_alloc(session, sizeof(private_t))` | Pool | Auto | ✅ OK |
| 2 | `switch_mutex_init(&tech_pvt->mutex, ...)` | Pool | Auto | ✅ OK |
| 3 | `switch_buffer_create(pool, &tech_pvt->inject_buffer, ...)` | Pool | Never explicitly freed | 🔴 See Finding #3 |
| 4 | `switch_core_session_alloc(session, read_scratch_len)` | Pool | Auto | ✅ OK |
| 5 | `switch_core_session_alloc(session, inject_scratch_len)` | Pool | Auto | ✅ OK |
| 6 | `speex_resampler_init(channels, ...)` | **Heap** | `ai_engine_session_cleanup` | ✅ OK |
| 7 | `new ai_engine::AIEngine()` | **Heap** | `ai_engine_session_cleanup` — `delete engine` | ✅ OK |

### 1.4 `ai_engine_glue.cpp` — `ai_engine_session_cleanup()`

| Step | Action | Notes |
|------|--------|-------|
| 1 | Lock mutex, set `cleanup_started = TRUE` | ✅ |
| 2 | Set `pAIEngine = nullptr` | ✅ |
| 3 | Unlock mutex | ✅ |
| 4 | `engine->stop()` | Joins threads, frees resamplers | ✅ |
| 5 | `delete engine` | ✅ |
| 6 | `speex_resampler_destroy(tech_pvt->resampler)` | ✅ |
| 7 | **NO** `inject_resampler` destruction | 🟡 N/A — AI mode doesn't create one in glue |

### 1.5 `ai_engine/ai_engine.cpp` — `AIEngine::start()`

| # | Allocation | Type | Free Location | Status |
|---|-----------|------|---------------|--------|
| 1 | `SPSCRingBuffer` (posix_memalign) | **Heap** | `AIEngine::stop()` → `ring_buffer_.reset()` | ✅ OK |
| 2 | `speex_resampler_init` (upsample) | **Heap** | `AIEngine::stop()` → `speex_resampler_destroy` | ✅ OK |
| 3 | `speex_resampler_init` (downsample, lazy) | **Heap** | `AIEngine::stop()` → `speex_resampler_destroy` | ✅ OK |
| 4 | `OpenAIRealtimeClient` (unique_ptr) | **Heap** | `AIEngine::stop()` → `openai_.reset()` | ✅ OK |
| 5 | TTS engine (unique_ptr) | **Heap** | `AIEngine::stop()` → `tts_engine_.reset()` | ✅ OK |
| 6 | TTS cache (unique_ptr) | **Heap** | `AIEngine::stop()` → `tts_cache_.reset()` | ✅ OK |
| 7 | TTS worker thread | Thread | `AIEngine::stop()` → `tts_thread_.join()` | ✅ OK |
| 8 | Reconnect thread | Thread | `AIEngine::stop()` → `reconnect_thread_.join()` | ✅ OK |

### 1.6 `ai_engine/openai_realtime.cpp`

| # | Allocation | Type | Free Location | Status |
|---|-----------|------|---------------|--------|
| 1 | `WebSocketClient` (unique_ptr) | **Heap** | Destructor → `disconnect()` | ✅ OK |
| 2 | base64 encoded strings | Stack/Heap (std::string) | Auto RAII | ✅ OK |

### 1.7 `ai_engine/tts_elevenlabs.cpp`

| # | Allocation | Type | Free Location | Status |
|---|-----------|------|---------------|--------|
| 1 | `curl_easy_init()` | **Heap** | `curl_easy_cleanup(curl)` | ✅ OK |
| 2 | `curl_slist_append` for headers | **Heap** | `curl_slist_free_all` | Need to verify |
| 3 | `CurlWriteContext::pcm_buffer` | Stack (vector) | Auto RAII | ✅ OK |

---

## 2. MUTEX LOCK/UNLOCK PAIR TRACE

### 2.1 `tech_pvt->mutex` (FreeSWITCH `switch_mutex_t`)

| Location | Lock | Unlock | Risk |
|----------|------|--------|------|
| `capture_callback` WRITE_REPLACE (line ~96) | `switch_mutex_lock` | `switch_mutex_unlock` (line ~145) | ⚠️ Long hold — reads inject_buffer, may re-lock |
| `capture_callback` WRITE_REPLACE partial (line ~152) | `switch_mutex_lock` | `switch_mutex_unlock` (line ~161) | ⚠️ **Nested lock** in same callback |
| `capture_callback` telemetry (line ~175) | `switch_mutex_lock` | `switch_mutex_unlock` (line ~183) | ✅ Short hold |
| `stream_frame` (line ~1299) | `switch_mutex_trylock` | `switch_mutex_unlock` (line ~1304) | ✅ Non-blocking — good |
| `stream_frame` buffer flush (line ~1323) | `switch_mutex_lock` | `switch_mutex_unlock` (lines ~1339,1342) | ⚠️ **Unlocks inside while loop**, re-locks |
| `processMessage` inject resampler (line ~699) | `switch_mutex_lock` | `switch_mutex_unlock` (line ~732) | 🔴 **Holds mutex during `speex_resampler_init`** |
| `processMessage` inject write (line ~756) | `switch_mutex_lock` | `switch_mutex_unlock` (line ~803) | ⚠️ Long hold during buffer overflow handling |
| `stream_session_cleanup` (line ~1463) | `switch_mutex_lock` | `switch_mutex_unlock` (line ~1475) | ✅ OK |
| `ai_engine_session_cleanup` (line ~305) | `switch_mutex_lock` | `switch_mutex_unlock` (line ~319) | ✅ OK |
| `stream_session_send_text` (line ~1103) | `switch_mutex_lock` | `switch_mutex_unlock` (line ~1111) | ✅ Short hold |

### 2.2 AI Engine internal mutexes

| Mutex | Location | Risk |
|-------|----------|------|
| `resampler_mutex_` | `resample_up()`, `resample_down()`, `stop()` | ✅ Properly scoped with `lock_guard` |
| `tts_queue_mutex_` | `tts_worker_loop()`, `on_openai_text_delta()`, `flush_tts_queue()` | ✅ OK |
| `sentence_mutex_` | `on_openai_text_delta()`, `on_openai_response_done()`, `handle_barge_in()` | ✅ OK |
| `stats_mutex_` | Various | ✅ OK — low contention |
| `reconnect_mutex_` | `on_openai_connection_change()`, `stop()` | ✅ OK |
| `m_stateMutex` (AudioStreamer) | `deleteFiles()` | ✅ OK |

---

## 3. ATOMIC OPERATIONS TRACE

### 3.1 `private_data` atomics (C-level, `switch_atomic_t`)

| Atomic | Writers | Readers | Memory Order | Risk |
|--------|---------|---------|-------------|------|
| `audio_paused` | `stream_session_pauseresume` | `stream_frame`, `ai_engine_feed_frame` | **None specified** (volatile only) | 🔴 Finding #4 |
| `close_requested` | `capture_callback` CLOSE, `cleanup` | `capture_callback` READ, `media_bug_close` | **None specified** (volatile only) | 🔴 Finding #4 |
| `cleanup_started` | `cleanup` functions | `ai_engine_feed_frame`, `stream_frame` | **None specified** (volatile only) | 🔴 Finding #4 |

**Critical Issue:** `switch_atomic_t` is defined as `volatile uint32_t` in FreeSWITCH. The code uses direct assignment (`= SWITCH_TRUE`) rather than `switch_atomic_set()` / `switch_atomic_read()`. On x86 this is effectively `seq_cst` due to strong memory model, but on ARM/AARCH64 **this is a data race** under C11/C++11 rules. The volatile keyword alone does NOT guarantee atomicity or memory ordering on non-x86.

### 3.2 AI Engine atomics (C++ `std::atomic`)

| Atomic | Orders Used | Status |
|--------|-------------|--------|
| `running_` | `relaxed` read / `acq_rel` exchange / `release` store | ✅ Correct |
| `tts_abort_` | `relaxed` read / `release` store | ✅ OK for flag |
| `state_` | `acq_rel` exchange / `acquire` load | ✅ Correct |
| `reconnect_attempts_` | `relaxed` | ✅ OK — advisory |
| `tts_sequence_` | `relaxed` | ✅ OK — monotonic counter |
| `m_cleanedUp` (AudioStreamer) | `release` store / `acquire` load | ✅ Correct |
| `connected_`, `session_configured_`, `is_responding_`, `is_speech_active_` | `release`/`acquire` | ✅ Correct |

---

## 4. FULL LIFECYCLE TRACE

### 4.1 WebSocket Streaming Mode

```
start_capture()
  ├─ switch_core_session_alloc(private_t)
  ├─ stream_session_init()
  │   ├─ read config from channel vars
  │   ├─ stream_data_init()
  │   │   ├─ switch_mutex_init()
  │   │   ├─ switch_buffer_create(sbuffer)
  │   │   ├─ switch_buffer_create(inject_buffer)
  │   │   ├─ switch_core_session_alloc(read_scratch)
  │   │   ├─ switch_core_session_alloc(inject_scratch)
  │   │   ├─ AudioStreamer::create() → connects WS
  │   │   ├─ new shared_ptr<AudioStreamer> → pAudioStreamer
  │   │   └─ speex_resampler_init(resampler) if needed
  │   └─ return tech_pvt via ppUserData
  ├─ switch_core_media_bug_add(capture_callback, tech_pvt)
  └─ switch_channel_set_private(MY_BUG_NAME, bug)

─── STREAMING ───
  capture_callback(READ)
  ├─ stream_frame()
  │   ├─ trylock mutex → copy streamer shared_ptr
  │   ├─ switch_core_media_bug_read() in loop
  │   ├─ resample if needed
  │   └─ streamer->writeBinary()

  capture_callback(WRITE_REPLACE)
  ├─ lock mutex
  ├─ read inject_buffer → frame->data
  └─ unlock mutex

  WS callback → eventCallback()
  ├─ switch_core_session_locate() → session
  ├─ processMessage() → decode audio → inject_buffer
  └─ switch_core_session_rwunlock()

─── TEARDOWN ───
  capture_callback(CLOSE)         OR         do_stop() → stream_session_cleanup()
  ├─ close_requested = TRUE                  ├─ lock mutex
  ├─ stream_session_cleanup()                ├─ cleanup_started = TRUE
  │   ├─ lock mutex                          ├─ set channel private = NULL
  │   ├─ cleanup_started = TRUE              ├─ save sp_wrap, clear pAudioStreamer
  │   ├─ set channel private = NULL          ├─ unlock mutex
  │   ├─ save sp_wrap, clear pAudioStreamer   ├─ bug_remove (if not closing)
  │   ├─ unlock mutex                        ├─ delete sp_wrap
  │   ├─ (skip bug_remove if closing)        ├─ streamer->markCleanedUp()
  │   ├─ delete sp_wrap                      ├─ streamer->disconnect()
  │   ├─ streamer->markCleanedUp()           └─ destroy_tech_pvt()
  │   ├─ streamer->disconnect()
  │   └─ destroy_tech_pvt()
```

### 4.2 AI Engine Mode

```
start_capture_ai()
  ├─ switch_core_session_alloc(private_t)
  ├─ switch_mutex_init()
  ├─ ai_engine_session_init()
  │   ├─ read config from channel vars
  │   ├─ switch_buffer_create(inject_buffer)
  │   ├─ switch_core_session_alloc(read_scratch, inject_scratch)
  │   ├─ speex_resampler_init(resampler) if needed
  │   ├─ new AIEngine()
  │   ├─ engine->set_event_callback(lambda captures uuid string)
  │   ├─ engine->start(cfg)
  │   │   ├─ SPSCRingBuffer
  │   │   ├─ DSPPipeline.init()
  │   │   ├─ create_tts_engine()
  │   │   ├─ TTSCache
  │   │   ├─ speex_resampler_init(upsample)
  │   │   ├─ OpenAIRealtimeClient → connect WS to OpenAI
  │   │   └─ spawn tts_thread_
  │   └─ tech_pvt->pAIEngine = engine
  ├─ switch_core_media_bug_add(capture_callback, tech_pvt)
  └─ switch_channel_set_private(MY_BUG_NAME_AI, bug)

─── STREAMING ───
  capture_callback(READ)
  ├─ ai_engine_feed_frame()
  │   ├─ switch_core_media_bug_read() in loop
  │   └─ engine->feed_audio() → upsample → OpenAI WS send

  capture_callback(WRITE_REPLACE)
  ├─ ai_engine_read_audio()
  │   └─ engine->read_audio() → SPSCRingBuffer → dest frame

  AI Callbacks (TTS thread / WS thread):
  ├─ on_openai_text_delta() → sentence_buffer → TTS queue
  ├─ tts_worker_loop() → process_tts_item() → HTTP TTS → on_tts_audio()
  │   └─ on_tts_audio() → resample_down → DSP → ring_buffer_->write_pcm16()
  └─ event callback → switch_core_session_locate → fire FS events

─── TEARDOWN ───
  capture_callback(CLOSE)
  ├─ close_requested = TRUE
  ├─ ai_engine_session_cleanup()
  │   ├─ lock mutex, cleanup_started = TRUE
  │   ├─ pAIEngine = nullptr
  │   ├─ unlock mutex
  │   ├─ (skip bug_remove if closing)
  │   ├─ engine->stop()
  │   │   ├─ tts_abort_ = true
  │   │   ├─ join tts_thread_
  │   │   ├─ join reconnect_thread_
  │   │   ├─ openai_->disconnect()
  │   │   ├─ destroy up/downsample resamplers
  │   │   └─ reset tts_engine_, tts_cache_, ring_buffer_
  │   ├─ delete engine
  │   └─ speex_resampler_destroy(tech_pvt->resampler)
```

---

## 5. THREAD BOUNDARY ANALYSIS

### Threads involved per session:

| Thread | What it does | Data touched |
|--------|-------------|--------------|
| **FreeSWITCH media thread** | `capture_callback` (READ + WRITE_REPLACE) | `tech_pvt->*`, `sbuffer`, `inject_buffer`, ring buffer |
| **WebSocket I/O thread** (libwsc) | WS callbacks in `AudioStreamer::eventCallback()` | `tech_pvt->inject_buffer`, `tech_pvt->inject_resampler` |
| **TTS worker thread** | `AIEngine::tts_worker_loop()` | `ring_buffer_`, `tts_queue_`, `sentence_buffer_`, resamplers |
| **OpenAI WS thread** (libwsc) | `handle_message()` → AI engine callbacks | `sentence_buffer_`, `tts_queue_`, `ring_buffer_` (via barge-in) |
| **Reconnect thread** | `on_openai_connection_change()` | `openai_` unique_ptr |
| **FreeSWITCH API thread** | `stream_function()` / `do_stop()` | `tech_pvt` via bug lookup |

### Cross-thread data flow:

```
MediaThread ──write──→ sbuffer ──read──→ MediaThread (no cross-thread — OK)
MediaThread ──write──→ sbuffer ──read──→ WS I/O (via stream_frame — uses streamer shared_ptr)

WS I/O ──write──→ inject_buffer ──read──→ MediaThread (WRITE_REPLACE)
  Protected by: tech_pvt->mutex
  Risk: WS I/O holds mutex during resample + buffer write (can be >1ms)

TTS Thread ──write──→ ring_buffer_ ──read──→ MediaThread (read_audio)
  Protected by: SPSCRingBuffer lock-free (single-producer single-consumer)
  ✅ Correct design — TTS thread is sole producer, media thread is sole consumer

OpenAI WS ──trigger──→ barge_in ──flush──→ ring_buffer_, tts_queue_
  Protected by: Various mutexes + atomic tts_abort_
  Risk: barge_in calls ring_buffer_->flush() from OpenAI WS thread,
        but media thread reads ring_buffer_ concurrently.
        SPSCRingBuffer::flush() does tail_.store(head_.load()) — this is safe
        only if flush() is called from the PRODUCER side. 🔴 Finding #5
```

---

## 6. FINDINGS: NUMBERED & CLASSIFIED

### 🔴 CRITICAL

**Finding #1: `inject_buffer` written to after `close_requested` in WS streaming mode**

- **File:** `audio_streamer_glue.cpp`, `processMessage()` (line ~756+)
- **Issue:** The WebSocket `processMessage()` method writes to `tech_pvt->inject_buffer` after acquiring the mutex. However, it does NOT check `tech_pvt->close_requested` or `tech_pvt->cleanup_started` before writing. If the cleanup is in progress on another thread, the buffer may be nullified or the session pool may be destroyed.
- **Race window:** Between `stream_session_cleanup()` unlocking the mutex (line ~1475) and `destroy_tech_pvt()` setting `inject_buffer = nullptr`, the WS thread could still call `processMessage()` and use the inject_buffer.
- **Impact:** Use-after-free, crash, data corruption.
- **Mitigation present:** `switch_core_session_locate()` in `eventCallback()` prevents access to a destroyed session, but the session may still be alive while cleanup is in progress.

**Finding #2: AudioStreamer shared_ptr prevents deterministic destruction timing**

- **File:** `audio_streamer_glue.cpp`
- **Issue:** `pAudioStreamer` stores a `std::shared_ptr<AudioStreamer>*` on the heap. During cleanup, `delete sp_wrap` releases one reference. But `stream_frame()` copies the shared_ptr into a local `streamer` variable (line ~1302). If `stream_frame()` is executing concurrently with cleanup, the local shared_ptr keeps AudioStreamer alive. The AudioStreamer's WS callbacks can still fire after `destroy_tech_pvt()` has nullified the buffers.
- **Impact:** AudioStreamer's `eventCallback()` could call `processMessage()` which accesses `tech_pvt->inject_buffer` (now nullptr) → crash.
- **Saving grace:** `markCleanedUp()` sets `m_cleanedUp` and clears WS callbacks, AND `eventCallback` checks `isCleanedUp()`. But `processMessage` calls `get_media_bug(psession)` which goes through channel private — already set to nullptr. So this specific path is likely safe.
- **Residual risk:** The `stream_frame()` local shared_ptr keeps the streamer alive; `writeBinary()` could still be called after cleanup. The WS client should handle disconnected state gracefully.

**Finding #3: AI mode `inject_buffer` allocated but never used**

- **File:** `ai_engine_glue.cpp`, `ai_engine_session_init()` (line ~152)
- **Issue:** AI mode creates an `inject_buffer` via `switch_buffer_create()`, but the WRITE_REPLACE callback in AI mode calls `ai_engine_read_audio()` which reads from `AIEngine::ring_buffer_` (a `SPSCRingBuffer`), NOT from `tech_pvt->inject_buffer`. The inject_buffer is dead weight — it wastes memory and is never read from.
- **Impact:** Memory waste (up to 5000ms × sample_rate × 2 bytes). Not a crash, but confusing and a maintenance hazard.

**Finding #4: `switch_atomic_t` used without proper atomic operations**

- **File:** `mod_audio_stream.c`, `audio_streamer_glue.cpp`, `ai_engine_glue.cpp`
- **Issue:** All three atomics (`audio_paused`, `close_requested`, `cleanup_started`) are read and written with direct assignment (`tech_pvt->close_requested = SWITCH_TRUE`) instead of `switch_atomic_set()` / `switch_atomic_read()`. `switch_atomic_t` is `volatile uint32_t` — volatile alone is NOT sufficient for thread-safe access on ARM/AARCH64.
- **Impact:** On ARM-based FreeSWITCH deployments (e.g., Raspberry Pi, AWS Graviton), this is a data race that can cause missed state transitions, double cleanup, or use-after-free.
- **Impact on x86:** Effectively safe due to x86's strong memory model, but still undefined behavior per C11/C++11 standards.

**Finding #5: SPSCRingBuffer `flush()` called from wrong thread**

- **File:** `ai_engine/ai_engine.cpp`, `handle_barge_in()` (line ~272)
- **Issue:** `handle_barge_in()` is called from `on_openai_speech_started()`, which is triggered by the OpenAI WebSocket message callback. This runs on the **WS I/O thread**. It calls `ring_buffer_->flush()`. But the SPSCRingBuffer contract is: **single producer (TTS thread)**, **single consumer (media thread)**. `flush()` does `tail_.store(head_.load())`, which is a consumer-side operation being called from neither the producer nor consumer thread.
- **Impact:** On weakly-ordered architectures, the tail store may not be visible to the media thread reader, causing stale audio to be read. On x86, likely safe but violates the SPSC contract.

**Finding #6: Reconnect thread replaces `openai_` unique_ptr without synchronization with `feed_audio()`**

- **File:** `ai_engine/ai_engine.cpp`, `on_openai_connection_change()` (line ~482)
- **Issue:** The reconnect thread does `openai_ = std::move(fresh)` (line ~end of reconnect lambda). Meanwhile, the FreeSWITCH media thread calls `feed_audio()` which reads `openai_->is_connected()` and `openai_->send_audio()`. There is NO mutex protecting `openai_`. This is a data race — the unique_ptr move could invalidate the pointer while `feed_audio()` is dereferencing it.
- **Impact:** Crash, segfault during reconnect.

---

### 🟡 MAJOR

**Finding #7: Mutex held during `speex_resampler_init()` in inject path**

- **File:** `audio_streamer_glue.cpp`, `processMessage()` (line ~699–731)
- **Issue:** When the inject resampler needs to be (re)created, `speex_resampler_init()` is called while holding `tech_pvt->mutex`. Speex resampler initialization involves heap allocation and FIR filter coefficient computation, which can take >1ms for high-quality settings. This blocks the media thread from reading the inject_buffer.
- **Impact:** Audio glitch (gap or stutter) during resampler initialization. Happens on first audio injection and whenever sample rate changes.

**Finding #8: `processMessage()` performs `resample_pcm16le_speex()` outside mutex but with raw pointer to `inject_resampler`**

- **File:** `audio_streamer_glue.cpp` (line ~735)
- **Issue:** After unlocking the mutex (line ~732), `processMessage()` calls `resample_pcm16le_speex()` with `local_resampler` — a raw pointer copied from `tech_pvt->inject_resampler`. If cleanup runs concurrently and `destroy_tech_pvt()` calls `speex_resampler_destroy(tech_pvt->inject_resampler)`, the `local_resampler` pointer becomes dangling.
- **Impact:** Use-after-free in resampler, crash.
- **Mitigation:** `processMessage()` runs inside `eventCallback()` which holds a `switch_core_session_locate()` lock, and cleanup checks `cleanup_started`. But the window exists between `sp_wrap` being deleted and the resampler being destroyed in `destroy_tech_pvt()`.

**Finding #9: `stream_frame()` uses `read_scratch` buffer outside mutex**

- **File:** `audio_streamer_glue.cpp`, `stream_frame()` (lines ~1381, ~1401)
- **Issue:** In the resampler path, `stream_frame()` writes to `tech_pvt->read_scratch` via `speex_resampler_process_int()` without holding the mutex (the mutex was unlocked at line ~1304). Then inside the flush loop (line ~1339), it locks the mutex to read `sbuffer` into `read_scratch`, unlocks, and sends via WS. If another thread modifies `read_scratch` or `read_scratch_len` concurrently, data corruption occurs.
- **Practical risk:** Low — `read_scratch` is only written by the media thread. But `processMessage()` in the WS thread doesn't touch it. **Safe in current code**, but fragile.

**Finding #10: `capture_callback` WRITE_REPLACE has nested re-lock pattern**

- **File:** `mod_audio_stream.c` (lines ~100–162)
- **Issue:** The WRITE_REPLACE handler locks `tech_pvt->mutex`, reads the inject_buffer, then if `got < need`, it locks the mutex AGAIN (line ~152). Since the mutex is `SWITCH_MUTEX_NESTED`, this works, but the inner lock section re-reads the inject_buffer and rewrites it — effectively doing a buffer compaction inside the media bug callback.
- **Impact:** This rewrite-in-place is costly and can extend the media thread callback beyond the 1ms budget. The compaction reads all remaining bytes, zeroes the buffer, and rewrites them.

**Finding #11: `barge_in` flushes ring_buffer but TTS thread may still be writing**

- **File:** `ai_engine/ai_engine.cpp`, `handle_barge_in()` (line ~268–292)
- **Issue:** `handle_barge_in()` sets `tts_abort_ = true`, calls `flush_tts_queue()`, then `ring_buffer_->flush()`, then sleeps 10ms, then clears `tts_abort_ = false`. But the TTS thread may have already entered `on_tts_audio()` and be in the middle of `ring_buffer_->write_pcm16()` when flush is called. The flush + write race on the ring buffer violates the SPSC contract (flush modifies the tail from a non-consumer thread, while write modifies the head from the producer).
- **Impact:** Audio corruption — partial old audio mixed with new audio. The 10ms sleep is a heuristic, not a guarantee.

**Finding #12: No backpressure on OpenAI audio send**

- **File:** `ai_engine/ai_engine.cpp`, `feed_audio()` (line ~213)
- **Issue:** `feed_audio()` base64-encodes every audio frame and sends it via WebSocket. The `send_audio()` method calls `ws_->sendMessage()` with no flow control. If the WebSocket write buffer is full (network congestion), this blocks the FreeSWITCH media thread.
- **Impact:** Media thread stall → audio gap on the call.

**Finding #13: TTS cache grows unbounded within TTL**

- **File:** `ai_engine/tts_cache.cpp`
- **Issue:** `evict_if_needed()` only evicts when `lru_list_.size() >= cfg_.max_entries`. The max is 200 entries, each up to `max_audio_bytes = 5MB`. Worst case: 200 × 5MB = 1GB per session.
- **Impact:** Memory exhaustion on systems with many concurrent sessions.
- **Mitigation:** Per-session cache is destroyed on session cleanup. But during a long call with varied responses, memory can spike.

**Finding #14: `capture_callback` CLOSE sets `close_requested` without mutex**

- **File:** `mod_audio_stream.c` (line ~48)
- **Issue:** `tech_pvt->close_requested = SWITCH_TRUE` is set without holding `tech_pvt->mutex`. In `stream_session_cleanup()`, `cleanup_started` IS set under mutex. But `close_requested` is checked without mutex in `capture_callback` READ (line ~61). This is a classic TOCTOU — the READ handler can pass the check, then CLOSE fires, sets the flag, and cleanup starts while READ is still running `stream_frame()`.
- **Impact:** `stream_frame()` accesses `pAudioStreamer` that may be deleted by cleanup. The trylock + shared_ptr copy in `stream_frame()` partially mitigates this, but the window exists.

---

### 🟢 MINOR

**Finding #15: `stream_data_init()` does `memset(tech_pvt, 0, sizeof(private_t))` AFTER saving cfg**

- **File:** `audio_streamer_glue.cpp` (line ~873)
- **Issue:** The pattern `saved_cfg = tech_pvt->cfg; memset(tech_pvt, 0, ...); tech_pvt->cfg = saved_cfg;` is correct for preserving config but zeroes out the mutex that may have been initialized by a previous call. Since mutex is re-initialized right after, this is safe, but it's a code smell.

**Finding #16: `inject_scratch` reallocation on media thread**

- **File:** `mod_audio_stream.c` (lines ~103–110)
- **Issue:** If `inject_scratch_len < need`, a new buffer is allocated via `switch_core_session_alloc()`. This is a pool allocation (not heap), so it's fast, but the old buffer is leaked (never freed — pool will free it at session end). Repeated reallocations waste pool memory.
- **Impact:** Pool memory waste. Not a heap leak.

**Finding #17: Missing `curl_slist_free_all` verification**

- **File:** `ai_engine/tts_elevenlabs.cpp`
- **Issue:** Need to verify that curl headers slist is freed in all paths. (Not fully traceable without reading full synthesize function.)

**Finding #18: Telemetry counters not atomic**

- **File:** `mod_audio_stream.c` (lines ~121–130, 173–183)
- **Issue:** `inject_write_calls`, `inject_bytes`, `inject_underruns` are `uint64_t` — NOT atomic. They're written in WRITE_REPLACE (media thread) and read/reset in the telemetry section (same thread), so this is actually safe. But if ever read from another thread, it would be a data race.

**Finding #19: `mod_audio_stream_shutdown()` doesn't clean up active sessions**

- **File:** `mod_audio_stream.c` (line ~559)
- **Issue:** The shutdown function only frees event subclasses. It does NOT iterate active sessions to clean up bugs. If the module is unloaded while calls are active, the media bug callbacks will call into unloaded code.
- **Impact:** Crash on module unload with active calls. FreeSWITCH generally prevents this, but it's a defense gap.

**Finding #20: `validate_ws_uri` is too restrictive**

- **File:** `audio_streamer_glue.cpp` (line ~1001–1045)
- **Issue:** The URI validator rejects underscores in hostnames, which are valid in some DNS configurations. Also rejects IPv6 addresses.
- **Impact:** Cannot connect to WS servers with underscore hostnames or IPv6.

---

## 7. ANSWERS TO ARCHITECTURE QUESTIONS Q1–Q12

### Q1: Is `speex_resampler_destroy()` called for BOTH resampler AND inject_resampler in ALL exit paths?

**Answer: YES for WS streaming mode, PARTIALLY for AI mode.**

- **WS streaming mode:** `destroy_tech_pvt()` destroys both. Called from `stream_session_cleanup()` and from error path in `stream_session_init()`. ✅
- **AI mode:** `ai_engine_session_cleanup()` destroys `tech_pvt->resampler` but NOT `tech_pvt->inject_resampler`. This is correct because AI mode never creates an `inject_resampler` on `tech_pvt` (it has its own internal resamplers in `AIEngine`). ✅
- **AI Engine internal:** `AIEngine::stop()` destroys both `upsample_resampler_` and `downsample_resampler_`. ✅

### Q2: Is `pAudioStreamer` freed BEFORE or AFTER the mutex is destroyed?

**Answer: BEFORE.** `stream_session_cleanup()` deletes `sp_wrap` (line ~1481) before `destroy_tech_pvt()` (line ~1491). `destroy_tech_pvt()` locks/unlocks the mutex, then the mutex lives until session pool destruction. ✅ Safe ordering.

### Q3: Are `inject_scratch` and `read_scratch` freed, or are they pool-allocated?

**Answer: Pool-allocated.** Both use `switch_core_session_alloc()`, which allocates from the session's APR memory pool. They are freed automatically when the session is destroyed. ✅ No manual free needed.

### Q4: What happens if the WebSocket callback fires AFTER `cleanup_started` is set?

**Answer: Mostly safe, but with a gap.**

1. `eventCallback()` calls `switch_core_session_locate()` which returns NULL if session is gone → early return. ✅
2. `processMessage()` calls `get_media_bug()` which reads channel private → NULL after cleanup clears it. ✅
3. **BUT:** Between `cleanup_started` being set and channel private being cleared, there's a window where `processMessage()` could find the bug and access `tech_pvt`. The mutex and `cleanup_started` check inside `processMessage()` don't exist — processMessage doesn't check `cleanup_started` at all. 🔴

### Q5: Is there a TOCTOU race between checking `close_requested` and writing to `sbuffer`?

**Answer: YES.** See Finding #14. `capture_callback` READ checks `close_requested` (line 61), then calls `stream_frame()`. If `close_requested` is set between the check and `stream_frame()` executing, `stream_frame()` accesses buffers that may be in cleanup. The trylock in `stream_frame()` mitigates this — if cleanup holds the mutex, trylock fails and stream_frame returns. **Partially mitigated.**

### Q6: Can the WebSocket thread call `responseHandler` after the session is gone?

**Answer: NO.** `eventCallback()` uses `switch_core_session_locate()` which is session-safe. If the session is gone, locate returns NULL. Also, `markCleanedUp()` clears WS callbacks. ✅ Good defense-in-depth.

For AI mode: The event callback lambda captures a `session_uuid_str` (string copy) and also uses `switch_core_session_locate()`. ✅ Safe.

### Q7: Is the mutex held during inject_buffer writes AND reads, or only one side?

**Answer: BOTH sides.** 
- **Write side:** `processMessage()` holds mutex during `switch_buffer_write()` (line ~803). ✅
- **Read side:** `capture_callback` WRITE_REPLACE holds mutex during `switch_buffer_read()` (line ~96). ✅
- **Risk:** The write side also holds mutex during resampler init (Finding #7) and buffer overflow handling, which can block the read side. 🟡

### Q8: What memory ordering do the `switch_atomic` operations use?

**Answer: NONE.** They use direct volatile assignment. On x86 this is effectively `seq_cst` (store is a MOV which has release semantics, loads are acquire). On ARM, this is **undefined behavior** — no ordering guarantees. See Finding #4.

### Q9: What is the Speex resampler quality setting?

**Answer:** 
- Capture path resampler (WS mode): `INJECT_RESAMPLE_QUALITY = 7` (defined in `audio_streamer_glue.cpp` line ~43)
- Inject path resampler: `INJECT_RESAMPLE_QUALITY = 7`
- AI engine internal (up/down): `RESAMPLE_QUALITY = 7` (defined in `ai_engine.cpp` line ~18)
- Quality 7 is a good balance — moderate latency (group delay ~7 taps), good fidelity. Quality 10 would be better for voice but adds ~3x latency.

### Q10: What happens when `inject_buffer` has PARTIAL frames?

**Answer:** In WS streaming mode, `processMessage()` aligns decoded audio to `frame_align = out_channels * 2` (sample alignment) and optionally to 20ms frames (line ~746–750). So injection data is always sample-aligned. The read side in `capture_callback` reads up to `need` bytes (frame size), which is always a multiple of 2 bytes. ✅ Alignment enforced.

### Q11: Is the compressor applied BEFORE or AFTER resampling?

**Answer: AFTER.** In `AIEngine::on_tts_audio()`:
1. First: `resample_down()` — resample from TTS sample rate to FreeSWITCH rate
2. Then: `dsp_.process()` — applies DC blocker → noise gate → compressor → high shelf → LPF → soft clipper

This is correct — DSP should operate at the final output sample rate. ✅

### Q12: Does the DSP chain handle sample rate mismatches?

**Answer: YES.** `DSPPipeline::init()` takes `dsp_cfg.sample_rate` which is set to `cfg_.freeswitch_sample_rate` in `AIEngine::start()`. All filter coefficients (biquad, compressor attack/release, etc.) are computed based on this rate. The DSP chain only processes audio at the FreeSWITCH rate, after resampling. ✅

---

## 8. RISK ASSESSMENT

### Critical Risks (will cause crashes in production)

| Risk | Trigger Condition | Probability | Impact |
|------|------------------|-------------|--------|
| Finding #6 (openai_ race) | OpenAI disconnects during active streaming | High (network issues) | Segfault |
| Finding #1 (inject_buffer post-close) | WS message arrives during session teardown | Medium | Corruption/crash |
| Finding #5 (SPSC violation in barge-in) | Caller speaks during TTS playback | High (normal use) | Audio corruption |
| Finding #4 (volatile vs atomic on ARM) | Any ARM deployment | Low (most deploy x86) | Data race |

### Major Risks (will cause quality degradation)

| Risk | Trigger Condition | Probability | Impact |
|------|------------------|-------------|--------|
| Finding #7 (mutex during resampler init) | First audio injection per session | 100% | Audio gap 1-5ms |
| Finding #11 (barge-in + TTS race) | Rapid barge-in | Medium | Stale audio leak |
| Finding #12 (no WS backpressure) | Network congestion | Medium | Media thread stall |
| Finding #10 (nested lock compaction) | Inject underrun with partial data | Frequent | Extended callback time |

---

## 9. RECOMMENDED CODE CHANGES

### Priority 1: Fix Finding #6 — Protect `openai_` with mutex

Add a `std::mutex openai_mutex_` to AIEngine. Protect `openai_` access in `feed_audio()`, `handle_barge_in()`, and reconnect.

### Priority 2: Fix Finding #5 — Ring buffer barge-in

Add a `flush()` method that sets an atomic flag, and have the consumer (media thread) check it on each read and reset head/tail from the consumer side.

### Priority 3: Fix Finding #4 — Use proper FreeSWITCH atomic APIs

Replace `tech_pvt->close_requested = SWITCH_TRUE` with `switch_atomic_set(&tech_pvt->close_requested, SWITCH_TRUE)` and reads with `switch_atomic_read(&tech_pvt->close_requested)`.

### Priority 4: Fix Finding #1 — Check cleanup state in processMessage

Add `if (tech_pvt->cleanup_started || tech_pvt->close_requested)` check in `processMessage()` after acquiring the mutex.

### Priority 5: Fix Finding #3 — Remove dead inject_buffer from AI mode

Remove the `switch_buffer_create()` for `inject_buffer` in `ai_engine_session_init()`.

### Priority 6: Fix Finding #7 — Move resampler init outside mutex

Create the resampler before acquiring the mutex, then swap it in.

---

*End of Phase 1 Audit. Proceed to Phase 2: Audio Pipeline Validation.*
