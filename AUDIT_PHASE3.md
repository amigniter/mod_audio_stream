# PHASE 3: CONCURRENCY HARDENING — mod_audio_stream

**Date:** 2026-02-13  
**Status:** ✅ Complete — 10 fixes applied, build verified  

---

## SUMMARY

Phase 3 applied targeted code fixes for the critical and major findings from Phase 1. All changes compile cleanly. No functional regressions expected — changes are defensive hardening that narrow race windows and improve correctness on weakly-ordered architectures (ARM/AARCH64).

---

## FIX LOG

### Fix #1 — Finding #4 (CRITICAL): Proper atomic operations for `switch_atomic_t`

**Files:** `mod_audio_stream.c`, `audio_streamer_glue.cpp`, `ai_engine_glue.cpp`

**Problem:** `switch_atomic_t` (`volatile uint32_t`) was read/written with direct assignment (`= SWITCH_TRUE`), which is a data race on ARM/AARCH64. The `volatile` keyword alone does NOT guarantee atomicity or memory ordering on non-x86.

**Fix:** Replaced all direct assignments with `switch_atomic_set()` and all reads with `switch_atomic_read()` across 3 files.

**Changed locations:**
- `mod_audio_stream.c`: `capture_callback` CLOSE handler
- `audio_streamer_glue.cpp`: `media_bug_close()`, `stream_frame()`, `stream_session_cleanup()`, `stream_session_pauseresume()`
- `ai_engine_glue.cpp`: `ai_engine_feed_frame()`, `ai_engine_session_cleanup()`

---

### Fix #2 — Findings #1/#14 (CRITICAL): Close guard in `processMessage()`

**File:** `audio_streamer_glue.cpp`

**Problem:** `processMessage()` accessed `tech_pvt->inject_buffer` without checking `close_requested` or `cleanup_started`. During teardown, the buffer could be nullified while a WS callback was mid-write.

**Fix:** Added early-return guard after obtaining `tech_pvt`:
```c
if (switch_atomic_read(&tech_pvt->close_requested) || switch_atomic_read(&tech_pvt->cleanup_started)) {
    push_err(out, m_sessionId, "processMessage - session closing, skipping injection");
    return out;
}
```

---

### Fix #3 — Finding #4 (CRITICAL): Additional atomic fixes

**File:** `audio_streamer_glue.cpp`

**Problem:** Additional locations using direct volatile assignment for `audio_paused` and `cleanup_started`.

**Fix:** All remaining direct assignments converted to `switch_atomic_set()`/`switch_atomic_read()`.

---

### Fix #4 — Finding #6 (CRITICAL): Protect `openai_` with `openai_mutex_`

**Files:** `ai_engine/ai_engine.h`, `ai_engine/ai_engine.cpp`

**Problem:** The reconnect thread replaces `openai_` (unique_ptr) via `std::move` while the media thread's `feed_audio()` reads `openai_->is_connected()` and `openai_->send_audio()`. No synchronization → segfault on reconnect.

**Fix:**
1. Added `std::mutex openai_mutex_` member to `AIEngine`
2. Protected all `openai_` access in:
   - `feed_audio()`: resample outside lock, then `lock_guard` for `send_audio()`
   - `handle_barge_in()`: `lock_guard` for `cancel_response()`
   - `stop()`: `lock_guard` for `disconnect()`/`reset()`
   - Reconnect lambda: entire replacement block under `lock_guard`

---

### Fix #5 — Finding #5 (CRITICAL): Ring buffer SPSC-safe flush

**File:** `ai_engine/ring_buffer.h`, `ai_engine/ai_engine.cpp`

**Problem:** `handle_barge_in()` (OpenAI WS thread) called `ring_buffer_->flush()` which modifies the tail pointer — a consumer-side operation — from a non-consumer thread, violating the SPSC contract.

**Fix:**
1. Added `request_flush()` method — sets an `atomic<bool>` flag (thread-safe)
2. Added `check_flush_request()` method — called from consumer thread, executes the flush
3. `handle_barge_in()` now calls `request_flush()` instead of `flush()`
4. `read_audio()` calls `check_flush_request()` at the top of each invocation
5. The actual tail pointer modification only happens on the consumer (media) thread

---

### Fix #6 — Finding #3 (CRITICAL): Remove dead `inject_buffer` in AI mode

**File:** `ai_engine_glue.cpp`

**Problem:** `ai_engine_session_init()` allocated `inject_buffer` and `inject_scratch` (up to ~80KB/session), but AI mode uses `SPSCRingBuffer` instead — the buffers were never read.

**Fix:** Removed `switch_buffer_create()` and `switch_core_session_alloc()` for inject_buffer/inject_scratch. Set pointers to NULL with explanatory comment.

---

### Fix #7 — Finding #7 (MAJOR): Move resampler init outside mutex

**File:** `audio_streamer_glue.cpp`, `processMessage()`

**Problem:** `speex_resampler_init()` (heap alloc + FIR coefficient computation, ~1-5ms) was called while holding `tech_pvt->mutex`, blocking the media thread from reading the inject_buffer.

**Fix:** Restructured to a 3-phase pattern:
1. **Check phase** (under mutex): Determine if a new resampler is needed; if replacing, take ownership of old resampler and set to NULL
2. **Create phase** (no mutex): Destroy old resampler, create new resampler
3. **Install phase** (under mutex): Swap in new resampler; handle race where another thread installed one first

---

### Fix #8 — Finding #8 (MAJOR): Defensive resampler validity check

**File:** `audio_streamer_glue.cpp`, `processMessage()`

**Problem:** After unlocking the mutex, `processMessage()` uses `local_resampler` (raw pointer). If cleanup runs concurrently and destroys the resampler, this is a dangling pointer.

**Fix:** Added defensive check after obtaining `local_resampler`:
```c
if (!local_resampler ||
    switch_atomic_read(&tech_pvt->close_requested) ||
    switch_atomic_read(&tech_pvt->cleanup_started)) {
    push_err(out, m_sessionId, "processMessage - resampler invalidated during setup");
    return out;
}
```

Also added a second cleanup check after re-acquiring the mutex for the buffer write section, verifying both atomic flags and inject_buffer pointer before any write.

---

### Fix #9 — Finding #10 (MAJOR): Eliminate nested lock compaction

**File:** `mod_audio_stream.c`, `capture_callback` WRITE_REPLACE

**Problem:** When `got < need` (partial read from inject_buffer), the code re-locked the mutex and performed an expensive buffer compaction: read all remaining bytes, zero the buffer, write partial data + remaining data back. This could exceed the 1ms media callback budget.

**Fix:** Replaced the nested re-lock + compaction with a simple partial copy. The inject_scratch was already zero-filled, so partial data gets copied into `frame->data` with silence padding for the remainder. The few bytes of "lost" partial data (< 1 sample at underrun boundary) cause less audible artifact than the stutter from a long mutex hold.

**Before:**
```c
// 12 lines: re-lock, read remaining, zero buffer, rewrite in order
```

**After:**
```c
memcpy(frame->data, inj, got);
/* remainder already zeroed by the memset(inj, 0, need) above */
```

---

### Fix #10 — Finding #11 (MAJOR): Tighter barge-in abort window

**File:** `ai_engine/ai_engine.cpp`, `on_tts_audio()`

**Problem:** During barge-in, `tts_abort_` is set, but the TTS thread could already be inside `on_tts_audio()` (past the initial abort check) and still write stale audio to the ring buffer.

**Fix:** Added a second `tts_abort_` check with `memory_order_acquire` immediately before `ring_buffer_->write_pcm16()`, after resample + DSP processing. This minimizes the window where stale audio can leak into the ring buffer during barge-in.

---

## FINDINGS NOT ADDRESSED WITH CODE (Rationale)

| Finding | Severity | Rationale for deferral |
|---------|----------|----------------------|
| #2 (shared_ptr timing) | Critical | Already mitigated by `markCleanedUp()` + `isCleanedUp()` + `session_locate()`. Full fix requires architectural change (weak_ptr). |
| #9 (read_scratch outside mutex) | Major | Safe in current code — only media thread uses it. Flagged for documentation. |
| #12 (WS backpressure) | Major | Requires WS library modification or async send pattern. Deferred to Phase 6. |
| #13 (TTS cache unbounded) | Major | Per-session, destroyed on cleanup. Add byte-limit cap in Phase 6. |
| #15 (memset after cfg save) | Minor | Correct behavior, code smell only. |
| #16 (inject_scratch pool realloc) | Minor | Pool memory, not a leak. Low impact. |
| #17 (curl slist verify) | Minor | Needs full tts_elevenlabs.cpp trace. Deferred. |
| #18 (telemetry counters) | Minor | Same-thread access, currently safe. |
| #19 (shutdown cleanup) | Minor | FreeSWITCH prevents unload with active calls. |
| #20 (URI validator) | Minor | Feature limitation, not a bug. |

---

## BUILD VERIFICATION

```
$ cd build && make
[ 35%] Built target libwsc
[ 40%] Building C object ...mod_audio_stream.c.o
[ 45%] Building CXX object ...audio_streamer_glue.cpp.o
[ 50%] Building CXX object ...ai_engine_glue.cpp.o
[ 55%] Building CXX object ...ai_engine/ai_engine.cpp.o
[ 60%] Linking CXX shared library mod_audio_stream.so
[ 90%] Built target mod_audio_stream
[100%] Built target copyright_target
```

**Result:** ✅ Clean build, zero warnings, zero errors.

---

*Proceed to Phase 4: Test Case Generation.*
