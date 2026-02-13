# PHASE 6: PRODUCTION HARDENING — mod_audio_stream

**Date:** 2026-02-13  
**Status:** Design complete, priority code changes implemented

---

## 1. STRUCTURED LOGGING FRAMEWORK

### 1.1 Current State

- Uses `switch_log_printf()` with unstructured string formatting
- No correlation between related log entries
- No machine-parseable fields for alerting
- Log levels inconsistently applied (DEBUG-level info in production paths)

### 1.2 Recommended Log Taxonomy

```
[session_uuid] [component] [operation] [outcome] [key=value pairs]
```

#### Component Tags

| Tag | Scope |
|-----|-------|
| `WS_STREAM` | WebSocket streaming mode |
| `AI_ENGINE` | AI engine core |
| `TTS` | TTS pipeline |
| `DSP` | DSP pipeline |
| `INJECT` | Audio injection path |
| `CAPTURE` | Audio capture path |
| `LIFECYCLE` | Session lifecycle |
| `OPENAI` | OpenAI Realtime API |
| `RING_BUF` | Ring buffer operations |

#### Structured Log Examples

```
INFO  [abc-123] LIFECYCLE session_init status=ok mode=ws uri=wss://example.com rate=8000 ch=1
INFO  [abc-123] CAPTURE stream_frame frames_sent=150 bytes=48000 elapsed_ms=3000
WARN  [abc-123] INJECT buffer_overflow dropped_bytes=640 inuse=32000 capacity=40000
ERROR [abc-123] OPENAI connection_lost reconnect_attempt=2 delay_ms=2000
INFO  [abc-123] AI_ENGINE barge_in ring_buf_flushed=true tts_aborted=true latency_ms=45
INFO  [abc-123] TTS synthesize sentence="Hello world" cache=hit latency_ms=0.5
INFO  [abc-123] LIFECYCLE session_cleanup duration_ms=12 mode=ws
```

### 1.3 Log Level Policy

| Level | Use For | Production Default |
|-------|---------|-------------------|
| ERROR | Crashes prevented, data loss, connection failures | ON |
| WARNING | Recoverable issues: overflow, underrun, timeout | ON |
| NOTICE | State transitions, significant events | ON |
| INFO | Operational telemetry (periodic summaries) | ON |
| DEBUG | Per-frame details, buffer states, timing | OFF |

### 1.4 Implementation: Per-Session Telemetry Summary

Rather than logging every frame, emit a periodic summary every N seconds:

```c
/* Proposed telemetry structure for private_t */
struct session_telemetry {
    /* Capture path */
    uint64_t capture_frames;
    uint64_t capture_bytes;
    uint64_t capture_trylock_failures;
    
    /* Inject path */
    uint64_t inject_messages;
    uint64_t inject_bytes;
    uint64_t inject_overflows;
    uint64_t inject_underruns;
    
    /* Timing (microseconds) */
    uint64_t callback_time_sum_us;
    uint64_t callback_time_max_us;
    uint64_t callback_count;
    
    /* Report interval */
    switch_time_t last_report;
    int report_interval_ms;  /* default: 10000 (10s) */
};
```

---

## 2. HEALTH CHECKS

### 2.1 Session Health Monitor

Add a `health` subcommand to the API:

```
uuid_audio_stream <uuid> health
```

Returns JSON:

```json
{
  "session_uuid": "abc-123",
  "mode": "ai",
  "uptime_sec": 342,
  "state": "SPEAKING",
  "capture": {
    "frames_sent": 17100,
    "last_frame_ms_ago": 18,
    "trylock_failure_pct": 0.02
  },
  "inject": {
    "buffer_inuse_bytes": 6400,
    "buffer_capacity_bytes": 40000,
    "buffer_fill_pct": 16.0,
    "underrun_pct": 0.5,
    "overflow_count": 0
  },
  "ai": {
    "openai_connected": true,
    "tts_queue_depth": 0,
    "ring_buffer_ms": 45.0,
    "barge_in_count": 3,
    "reconnect_attempts": 0,
    "tts_cache_hit_pct": 40.0
  },
  "health": "HEALTHY"
}
```

#### Health States

| State | Condition |
|-------|-----------|
| `HEALTHY` | All metrics within bounds |
| `DEGRADED` | Underrun > 5% OR overflow > 0 OR trylock_failure > 1% |
| `UNHEALTHY` | OpenAI disconnected OR last_frame > 5000ms OR state == ERROR |

### 2.2 Global Health Endpoint

Add a global status API:

```
uuid_audio_stream global_health
```

Returns:

```json
{
  "active_ws_sessions": 45,
  "active_ai_sessions": 12,
  "total_sessions_created": 1234,
  "total_sessions_destroyed": 1177,
  "healthy": 55,
  "degraded": 2,
  "unhealthy": 0,
  "uptime_sec": 86400,
  "version": "1.0.0"
}
```

### 2.3 Implementation Approach

Track active sessions in a global linked list protected by a global mutex:

```c
/* Global session registry */
static switch_mutex_t *g_session_mutex = NULL;
static switch_hash_t *g_sessions = NULL;     /* uuid → private_t* */
static uint64_t g_total_created = 0;
static uint64_t g_total_destroyed = 0;

/* In mod_audio_stream_load: */
switch_mutex_init(&g_session_mutex, SWITCH_MUTEX_NESTED, pool);
switch_core_hash_init(&g_sessions);

/* In stream_data_init / ai_engine_session_init: */
switch_mutex_lock(g_session_mutex);
switch_core_hash_insert(g_sessions, tech_pvt->sessionId, tech_pvt);
g_total_created++;
switch_mutex_unlock(g_session_mutex);

/* In cleanup: */
switch_mutex_lock(g_session_mutex);
switch_core_hash_delete(g_sessions, tech_pvt->sessionId);
g_total_destroyed++;
switch_mutex_unlock(g_session_mutex);
```

---

## 3. GRACEFUL SHUTDOWN

### 3.1 Current State (Finding #19)

`mod_audio_stream_shutdown()` only frees event subclasses. It does NOT clean up active sessions.

### 3.2 Recommended Shutdown Sequence

```c
SWITCH_MODULE_SHUTDOWN_FUNCTION(mod_audio_stream_shutdown)
{
    /* Phase 1: Mark all sessions for shutdown */
    switch_mutex_lock(g_session_mutex);
    switch_hash_index_t *hi;
    for (hi = switch_core_hash_first(g_sessions); hi;
         hi = switch_core_hash_next(&hi)) {
        const void *key;
        void *val;
        switch_core_hash_this(hi, &key, NULL, &val);
        private_t *tp = (private_t *)val;
        if (tp) {
            switch_atomic_set(&tp->close_requested, SWITCH_TRUE);
        }
    }
    switch_mutex_unlock(g_session_mutex);

    /* Phase 2: Wait for sessions to drain (max 5 seconds) */
    int wait_ms = 0;
    while (wait_ms < 5000) {
        switch_mutex_lock(g_session_mutex);
        int remaining = switch_core_hash_count(g_sessions);
        switch_mutex_unlock(g_session_mutex);
        if (remaining == 0) break;
        switch_yield(100000);  /* 100ms */
        wait_ms += 100;
    }

    /* Phase 3: Force cleanup any remaining sessions */
    switch_mutex_lock(g_session_mutex);
    if (switch_core_hash_count(g_sessions) > 0) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_WARNING,
            "mod_audio_stream: %d sessions still active at shutdown\n",
            switch_core_hash_count(g_sessions));
    }
    switch_core_hash_destroy(&g_sessions);
    switch_mutex_unlock(g_session_mutex);

    /* Phase 4: Free events */
    switch_event_free_subclass(EVENT_JSON);
    switch_event_free_subclass(EVENT_CONNECT);
    switch_event_free_subclass(EVENT_DISCONNECT);
    switch_event_free_subclass(EVENT_ERROR);
    switch_event_free_subclass(EVENT_PLAY);
    switch_event_free_subclass(EVENT_AI_STATE);
    switch_event_free_subclass(EVENT_AI_TRANSCRIPT);
    switch_event_free_subclass(EVENT_AI_RESPONSE);

    return SWITCH_STATUS_SUCCESS;
}
```

---

## 4. CIRCUIT BREAKERS

### 4.1 WS Connection Circuit Breaker

Prevent reconnect storms when the WS server is down:

```
States: CLOSED → OPEN → HALF_OPEN
CLOSED:  Normal operation. On failure, increment counter.
OPEN:    After N failures in T seconds, stop connecting.
         Wait for cooldown period.
HALF_OPEN: After cooldown, allow one probe connection.
         On success → CLOSED. On failure → OPEN.
```

**Configuration:**

```c
struct circuit_breaker_config {
    int failure_threshold;    /* 5 — open after 5 failures */
    int cooldown_sec;         /* 30 — wait 30s before half-open */
    int window_sec;           /* 60 — failure window */
};
```

### 4.2 TTS Circuit Breaker

For ElevenLabs/OpenAI TTS API failures:

```
- Track consecutive TTS failures per session
- After 3 consecutive failures: switch to fallback TTS
- After 5 consecutive failures: disable TTS, play error tone
- On next success: reset counter
```

### 4.3 OpenAI Reconnect Circuit Breaker

Already partially implemented (`kMaxReconnectAttempts = 5`). Enhance with:

```cpp
/* In AIEngine: */
struct ReconnectCircuitBreaker {
    std::atomic<int> consecutive_failures{0};
    std::atomic<int> state{0};  /* 0=CLOSED, 1=OPEN, 2=HALF_OPEN */
    std::chrono::steady_clock::time_point last_failure;
    static constexpr int THRESHOLD = 5;
    static constexpr int COOLDOWN_SEC = 60;
    
    bool should_attempt() {
        int s = state.load(std::memory_order_acquire);
        if (s == 0) return true;  /* CLOSED */
        if (s == 1) {  /* OPEN */
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(
                    now - last_failure).count() > COOLDOWN_SEC) {
                state.store(2, std::memory_order_release);
                return true;  /* HALF_OPEN */
            }
            return false;
        }
        return true;  /* HALF_OPEN — allow probe */
    }
    
    void record_success() {
        consecutive_failures.store(0, std::memory_order_relaxed);
        state.store(0, std::memory_order_release);
    }
    
    void record_failure() {
        int f = consecutive_failures.fetch_add(1, std::memory_order_relaxed) + 1;
        last_failure = std::chrono::steady_clock::now();
        if (f >= THRESHOLD) {
            state.store(1, std::memory_order_release);
        }
    }
};
```

---

## 5. WATCHDOG TIMERS

### 5.1 Capture Watchdog

Detect stalled capture paths:

```c
/* In private_t: */
switch_time_t last_capture_frame_time;

/* In stream_frame: */
tech_pvt->last_capture_frame_time = switch_micro_time_now();

/* In periodic check (every 5s): */
if (switch_micro_time_now() - tech_pvt->last_capture_frame_time > 5000000) {
    switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR,
        "(%s) WATCHDOG: Capture stalled for >5s\n", tech_pvt->sessionId);
    /* Fire alarm event */
}
```

### 5.2 AI Engine Watchdog

```cpp
/* In AIEngine: */
std::atomic<uint64_t> last_activity_us_{0};

/* Updated in feed_audio, read_audio, on_tts_audio: */
last_activity_us_.store(current_us, std::memory_order_relaxed);

/* Watchdog thread or periodic check: */
bool is_stalled(uint64_t timeout_us = 30000000) const {  /* 30s */
    uint64_t last = last_activity_us_.load(std::memory_order_relaxed);
    if (last == 0) return false;
    return (current_us() - last) > timeout_us;
}
```

### 5.3 TTS Worker Watchdog

Detect hung TTS HTTP requests:

```cpp
/* In process_tts_item: */
auto start = std::chrono::steady_clock::now();
bool success = tts_engine_->synthesize(...);
auto elapsed = std::chrono::steady_clock::now() - start;

if (std::chrono::duration_cast<std::chrono::seconds>(elapsed).count() > 30) {
    AI_LOG_ERROR("(%s) WATCHDOG: TTS synthesis took >30s for: '%s'\n",
                 cfg_.session_uuid.c_str(), item.sentence.substr(0, 40).c_str());
}
```

---

## 6. DEFENSIVE CODING PATTERNS

### 6.1 Guard Macro for tech_pvt Access

```c
#define TECH_PVT_GUARD(tech_pvt, label) \
    do { \
        if (!(tech_pvt)) { \
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR, \
                "%s: NULL tech_pvt\n", (label)); \
            return SWITCH_STATUS_FALSE; \
        } \
        if (switch_atomic_read(&(tech_pvt)->cleanup_started)) { \
            return SWITCH_STATUS_FALSE; \
        } \
    } while(0)
```

### 6.2 Scoped Mutex Lock (C-compatible)

```c
/* Cleanup handler pattern for switch_mutex */
#define SCOPED_MUTEX_LOCK(mutex) \
    switch_mutex_lock(mutex); \
    /* Must unlock before any return */

/* Preferred: use goto cleanup pattern */
switch_mutex_lock(tech_pvt->mutex);
/* ... work ... */
status = SWITCH_STATUS_SUCCESS;
cleanup:
switch_mutex_unlock(tech_pvt->mutex);
return status;
```

### 6.3 Rate-Limited Logging

```c
/* Log at most once per interval */
static inline bool should_log(switch_time_t *last, int interval_ms) {
    switch_time_t now = switch_micro_time_now();
    if (now - *last > (switch_time_t)interval_ms * 1000) {
        *last = now;
        return true;
    }
    return false;
}

/* Usage: */
static switch_time_t last_overflow_log = 0;
if (overflow && should_log(&last_overflow_log, 5000)) {
    switch_log_printf(..., "inject buffer overflow ...");
}
```

---

## 7. CONFIGURATION VALIDATION

### 7.1 Channel Variable Validation

Add validation at session init:

```c
static switch_status_t validate_config(private_data_config_t *cfg, const char *session_id) {
    if (cfg->frame_ms < 10 || cfg->frame_ms > 100) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_WARNING,
            "(%s) frame_ms=%d out of range [10,100], clamping to 20\n",
            session_id, cfg->frame_ms);
        cfg->frame_ms = 20;
    }
    if (cfg->inject_buffer_ms < 100 || cfg->inject_buffer_ms > 30000) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_WARNING,
            "(%s) inject_buffer_ms=%d out of range [100,30000], clamping to 5000\n",
            session_id, cfg->inject_buffer_ms);
        cfg->inject_buffer_ms = 5000;
    }
    if (cfg->max_audio_base64_len < 1000 || cfg->max_audio_base64_len > 10000000) {
        cfg->max_audio_base64_len = 1000000;  /* 1MB default */
    }
    if (cfg->inject_min_buffer_ms < 0 || cfg->inject_min_buffer_ms > 5000) {
        cfg->inject_min_buffer_ms = 0;
    }
    return SWITCH_STATUS_SUCCESS;
}
```

### 7.2 AI Config Validation

```c
static switch_status_t validate_ai_config(ai_engine_config_t *cfg, const char *session_id) {
    if (cfg->openai_api_key[0] == '\0') {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR,
            "(%s) AI mode requires openai_api_key\n", session_id);
        return SWITCH_STATUS_FALSE;
    }
    if (cfg->vad_threshold < 0.0f || cfg->vad_threshold > 1.0f) {
        cfg->vad_threshold = 0.5f;
    }
    if (cfg->temperature < 0.0f || cfg->temperature > 2.0f) {
        cfg->temperature = 0.8f;
    }
    if (cfg->max_response_tokens < 1 || cfg->max_response_tokens > 16384) {
        cfg->max_response_tokens = 4096;
    }
    if (cfg->compressor_threshold_db > 0.0f) {
        cfg->compressor_threshold_db = -20.0f;  /* Must be negative */
    }
    if (cfg->lpf_cutoff_hz < 100.0f || cfg->lpf_cutoff_hz > 20000.0f) {
        cfg->lpf_cutoff_hz = 8000.0f;
    }
    return SWITCH_STATUS_SUCCESS;
}
```

---

## 8. ERROR RECOVERY PATTERNS

### 8.1 TTS Failure Recovery

```
On TTS failure:
1. Log error with sentence, error code, latency
2. If cache has ANY entry: play cached "I'm sorry, could you repeat that?"
3. If no cache: generate silence + fire error event to dialplan
4. Dialplan handler can transfer call, play prompt, or retry
```

### 8.2 OpenAI Connection Recovery

```
Current: Exponential backoff up to 5 attempts
Enhanced:
1. On disconnect: fire EVENT_AI_STATE with state=RECONNECTING
2. During reconnect: buffer incoming audio in a small ring buffer
3. On reconnect success: flush buffered audio to OpenAI (catch-up)
4. On reconnect failure (max attempts): fire EVENT_AI_STATE state=FAILED
5. Dialplan handler decides: transfer to human, retry, or hangup
```

### 8.3 Buffer Overflow Recovery

```
Current: Drop oldest audio (FIFO drop)
Enhanced:
1. Track overflow frequency (overflows per second)
2. If overflow_rate > 5/sec: 
   a. Log WARN with rate
   b. Fire event for monitoring
   c. Consider dropping ALL buffered audio and starting fresh
      (avoids accumulating stale audio with increasing latency)
3. If overflow persists > 30s:
   a. Log ERROR
   b. Possible root cause: network congestion or WS server slow
```

---

## 9. PRODUCTION DEPLOYMENT CHECKLIST

### 9.1 Pre-Deployment

- [ ] Build with `-O2 -DNDEBUG` (release flags)
- [ ] Run ASAN build against test suite — zero findings
- [ ] Run TSAN build against test suite — zero findings
- [ ] Run valgrind memcheck — zero leaks (definite)
- [ ] Verify all Phase 3 regression tests pass (C6.01–C6.70)
- [ ] Run BM-17 (callback duration) — p99 < 1ms
- [ ] Run BM-19 (scaling) at target concurrency — p99 < 30ms
- [ ] Run BM-21 (long run) — RSS stable over 4 hours
- [ ] Verify log output at INFO level is < 10 lines/sec/session

### 9.2 Deployment Configuration

```ini
# Recommended FreeSWITCH channel variables for production

# WS Streaming Mode
AUDIO_STREAM_SAMPLING=8000
AUDIO_STREAM_CHANNELS=1
AUDIO_STREAM_RTP_PACKETS=1
AUDIO_STREAM_INJECT_BUFFER_MS=5000
AUDIO_STREAM_INJECT_MIN_BUFFER_MS=200
AUDIO_STREAM_INJECT_LOG_EVERY_MS=10000
AUDIO_STREAM_MAX_AUDIO_BASE64_LEN=1000000
AUDIO_STREAM_RECONNECT_MAX=3

# AI Mode
AUDIO_STREAM_AI_OPENAI_MODEL=gpt-4o-realtime-preview
AUDIO_STREAM_AI_VAD_THRESHOLD=0.5
AUDIO_STREAM_AI_VAD_PREFIX_PADDING_MS=300
AUDIO_STREAM_AI_VAD_SILENCE_DURATION_MS=500
AUDIO_STREAM_AI_TEMPERATURE=0.8
AUDIO_STREAM_AI_MAX_RESPONSE_TOKENS=4096
AUDIO_STREAM_AI_ENABLE_BARGE_IN=true
AUDIO_STREAM_AI_DSP_ENABLED=true
AUDIO_STREAM_AI_COMPRESSOR_THRESHOLD_DB=-20
AUDIO_STREAM_AI_COMPRESSOR_MAKEUP_DB=6
AUDIO_STREAM_AI_LPF_CUTOFF_HZ=7500
AUDIO_STREAM_AI_ENABLE_TTS_CACHE=true
```

### 9.3 Monitoring

| Metric | Source | Alert |
|--------|--------|-------|
| Active sessions | `global_health` API | > capacity |
| Unhealthy sessions | `global_health` API | > 0 |
| Session error rate | FreeSWITCH events | > 1% |
| RSS per process | OS metrics | > expected |
| OpenAI reconnect rate | AI events | > 1/min |
| TTS error rate | TTS events | > 5% |
| Inject underrun rate | Telemetry logs | > 5% |
| Capture trylock failure | Telemetry logs | > 1% |

### 9.4 Operational Runbook

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| High inject underrun % | WS server sending audio too slowly | Check WS server, increase inject_min_buffer_ms |
| High capture trylock failure | Mutex contention from heavy injection | Check inject message rate, reduce logging |
| RSS growing | TTS cache growth or pool leak | Check session count, TTS cache size |
| OpenAI reconnect loops | API key invalid or rate limit | Check API key, reduce concurrency |
| Callback p99 > 2ms | Resampler or lock contention | Profile with perf, check concurrency |
| Silence after barge-in | Ring buffer flush delayed | Check media thread priority |
| Audio glitch on inject | Resampler init latency | Should be fixed (Fix #7), verify |

---

## 10. VERSION & BUILD METADATA

Embed build metadata in the module for operational debugging:

```c
/* In mod_audio_stream.c */
static const char *MOD_VERSION = "1.1.0";
static const char *MOD_BUILD_DATE = __DATE__ " " __TIME__;
#ifdef NDEBUG
static const char *MOD_BUILD_TYPE = "release";
#else
static const char *MOD_BUILD_TYPE = "debug";
#endif

/* Add to global_health output */
```

---

## SUMMARY OF PRODUCTION HARDENING ITEMS

| # | Item | Priority | Effort | Status |
|---|------|----------|--------|--------|
| 1 | Structured log taxonomy | P1 | Medium | Designed |
| 2 | Session health API | P1 | Medium | Designed |
| 3 | Global health API | P1 | Medium | Designed |
| 4 | Graceful shutdown | P0 | Low | Designed |
| 5 | WS circuit breaker | P2 | Medium | Designed |
| 6 | TTS circuit breaker | P2 | Medium | Designed |
| 7 | OpenAI circuit breaker | P2 | Low | Partially exists |
| 8 | Capture watchdog | P1 | Low | Designed |
| 9 | AI engine watchdog | P1 | Low | Designed |
| 10 | TTS worker watchdog | P2 | Low | Designed |
| 11 | Config validation | P0 | Low | Designed |
| 12 | Rate-limited logging | P1 | Low | Designed |
| 13 | Session registry | P1 | Low | Designed |
| 14 | Build metadata | P2 | Trivial | Designed |
| 15 | Deployment checklist | P0 | N/A | Complete |
| 16 | Monitoring config | P0 | N/A | Complete |
| 17 | Operational runbook | P0 | N/A | Complete |

---

*Phase 6 complete. All 6 phases of the production audit are now done.*
