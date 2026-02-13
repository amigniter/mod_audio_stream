# PHASE 5: PERFORMANCE BENCHMARKS — mod_audio_stream

**Date:** 2026-02-13  
**Status:** Benchmark framework defined, targets established

---

## 1. PERFORMANCE TARGETS

### 1.1 Latency Targets

| Path | Target | Budget Breakdown | Measurement Point |
|------|--------|-----------------|-------------------|
| **Capture latency** (RTP→WS send) | ≤ 25ms | RTP frame: 20ms + resample: <0.5ms + WS send: <1ms + overhead: <3.5ms | Timestamp delta: RTP arrival → WS frame sent |
| **Inject latency** (WS receive→RTP write) | ≤ 40ms | WS receive: <1ms + base64 decode: <0.5ms + resample: <0.5ms + buffer: 20ms (1 frame) + write: <1ms | Timestamp delta: WS message → RTP frame written |
| **AI first-audio latency** (text→RTP) | ≤ 500ms | OpenAI text: <50ms + sentence detect: <5ms + TTS HTTP: <300ms + resample+DSP: <5ms + ring buffer: <20ms + RTP: <1ms | Timestamp delta: first text delta → first non-zero RTP frame |
| **Barge-in latency** (speech_started→silence) | ≤ 80ms | Event processing: <5ms + cancel: <10ms + TTS abort: <10ms + ring buffer flush: <20ms + frame drain: <20ms | Timestamp delta: speech_started → first silent RTP frame |
| **Media callback budget** | ≤ 1ms (p99) | Lock acquire: <50μs + buffer read/write: <100μs + overhead: <850μs | Clock delta: callback entry → exit |

### 1.2 Throughput Targets

| Metric | Target | Conditions |
|--------|--------|-----------|
| Concurrent WS sessions | ≥ 200 per core | 8kHz mono, single WS message/20ms |
| Concurrent AI sessions | ≥ 50 per core | 8kHz mono, OpenAI + TTS active |
| Inject messages/sec/session | ≥ 100 | 20ms audio chunks at 8kHz |
| Capture frames/sec/session | 50 (at 20ms ptime) | Steady state |
| Ring buffer throughput | ≥ 5 MB/s per session | Continuous TTS output at 24kHz |

### 1.3 Resource Targets

| Metric | Target | Per Session |
|--------|--------|-------------|
| RSS per WS session | ≤ 2 MB | Includes pool + heap + WS buffers |
| RSS per AI session | ≤ 8 MB | Includes pool + heap + ring buffer + TTS cache |
| CPU per WS session (idle) | ≤ 0.1% | No active audio |
| CPU per WS session (active) | ≤ 1% | Continuous streaming + injection |
| CPU per AI session (active) | ≤ 3% | Continuous streaming + TTS + DSP |
| File descriptors per session | ≤ 4 | Session FD + WS FD + potential TTS HTTP |

---

## 2. BENCHMARK SUITE

### 2.1 Micro-Benchmarks (Component Level)

#### BM-01: Speex Resampler Throughput

```
Benchmark: resample 160 samples (20ms @ 8kHz) → 320 samples (16kHz)
Variants: quality 1, 3, 5, 7, 10
Iterations: 100,000
Measure: ns/call, throughput (samples/sec)
Target: < 50μs at quality 7
```

| Quality | Expected ns/call | Samples/sec |
|---------|-----------------|-------------|
| 1 | ~5μs | ~32M |
| 3 | ~10μs | ~16M |
| 5 | ~20μs | ~8M |
| 7 | ~35μs | ~4.5M |
| 10 | ~80μs | ~2M |

#### BM-02: Speex Resampler Init Time

```
Benchmark: speex_resampler_init(1, 16000, 8000, quality, &err)
Variants: quality 1, 3, 5, 7, 10
Iterations: 10,000
Measure: μs/call
Target: < 500μs at quality 7 (validates Fix #7 impact)
```

#### BM-03: Base64 Decode Throughput

```
Benchmark: base64_decode(encoded_audio)
Variants: 320 bytes (20ms@8kHz), 3200 bytes (200ms), 32000 bytes (2s)
Iterations: 100,000
Measure: MB/s
Target: > 500 MB/s
```

#### BM-04: SPSCRingBuffer Write/Read Throughput

```
Benchmark: write_pcm16(buf, 160) then read_pcm16(buf, 160)
Variants: 160 samples (20ms@8kHz), 480 samples (20ms@24kHz)
Iterations: 1,000,000
Measure: ns/pair, throughput (samples/sec)
Target: < 200ns per write+read pair
```

#### BM-05: SPSCRingBuffer Concurrent Throughput

```
Benchmark: Producer writes 160 samples, Consumer reads 160 samples, concurrent
Duration: 10 seconds
Measure: Total samples transferred, max producer stall
Target: > 50M samples/sec, zero stalls
```

#### BM-06: DSP Pipeline Processing Time

```
Benchmark: dsp.process(buffer, 160) — full chain
Variants: 8kHz (160 samples), 16kHz (320), 48kHz (960)
Iterations: 100,000
Measure: ns/call
Target: < 50μs for 160 samples
```

#### BM-07: cJSON Parse + Encode Throughput

```
Benchmark: cJSON_Parse(json_with_audio_data) + cJSON_PrintUnformatted
Variants: Small (1KB), Medium (10KB), Large (100KB)
Iterations: 10,000
Measure: μs/call
Target: < 100μs for 10KB
```

#### BM-08: switch_mutex Lock/Unlock Cost

```
Benchmark: switch_mutex_lock + switch_mutex_unlock (uncontended)
Iterations: 1,000,000
Measure: ns/pair
Target: < 100ns per pair (uncontended)
```

#### BM-09: switch_atomic_set/read Cost

```
Benchmark: switch_atomic_set + switch_atomic_read
Iterations: 10,000,000
Measure: ns/pair
Target: < 20ns per pair
```

#### BM-10: Sentence Buffer Throughput

```
Benchmark: add_token("word ") × 100, flush()
Iterations: 10,000
Measure: μs/100-token block
Target: < 50μs per block
```

---

### 2.2 Path-Level Benchmarks

#### BM-11: Capture Path End-to-End

```
Setup: FreeSWITCH session, 8kHz mono, WS server (localhost)
Measure: Timestamp at switch_core_media_bug_read → timestamp at WS frame received
Duration: 60 seconds continuous streaming
Metrics:
  - p50, p95, p99 latency
  - Min/max latency
  - Jitter (stddev)
Target: p99 < 25ms, jitter < 5ms
```

#### BM-12: Capture Path with Resampler

```
Setup: 16kHz session → 8kHz WS stream (requires resampler)
Same metrics as BM-11
Target: p99 < 27ms (2ms resampler budget)
```

#### BM-13: Inject Path End-to-End

```
Setup: WS server sends 20ms audio chunks at 50Hz
Measure: Timestamp at WS send → timestamp at RTP frame written
Duration: 60 seconds
Metrics: p50, p95, p99 latency
Target: p99 < 40ms
```

#### BM-14: Inject Path with Resampler

```
Setup: WS sends 16kHz audio, session at 8kHz (requires inject resampler)
Same metrics as BM-13
Target: p99 < 42ms
```

#### BM-15: AI Pipeline End-to-End

```
Setup: Active AI session, OpenAI returns text, ElevenLabs TTS
Measure: Timestamp at first text_delta → first non-zero RTP sample
Duration: 10 responses
Metrics: Min/max/avg/p95 first-audio latency
Target: p95 < 500ms
```

#### BM-16: Barge-In Response Time

```
Setup: AI session actively speaking, trigger speech_started
Measure: speech_started event → last non-zero RTP sample
Duration: 20 barge-in events
Metrics: Min/max/avg/p95
Target: p95 < 80ms
```

#### BM-17: Media Callback Duration

```
Setup: Active session with inject audio
Measure: capture_callback entry → exit (READ + WRITE_REPLACE)
Instrument: switch_micro_time_now() at entry/exit
Duration: 60 seconds (3000 callbacks)
Metrics: p50, p95, p99, max
Target: p99 < 1ms, max < 5ms
```

#### BM-18: Mutex Contention Under Load

```
Setup: Active WS session, inject messages at 100/sec
Measure: switch_mutex_trylock failure rate in stream_frame
Duration: 60 seconds
Metrics: Failure count, failure rate %
Target: < 0.5% trylock failures
```

---

### 2.3 System-Level Benchmarks

#### BM-19: Concurrent Session Scaling (WS)

```
Scenario: Ramp sessions 10→50→100→200→300
Each session: 8kHz mono, bidirectional audio
Duration: 120 seconds per step
Metrics per step:
  - CPU utilization (user + sys)
  - RSS total
  - Capture p99 latency
  - Inject p99 latency
  - Context switches/sec
Target: Linear CPU scaling, p99 latency < 30ms at 200 sessions
```

#### BM-20: Concurrent Session Scaling (AI)

```
Scenario: Ramp AI sessions 5→10→25→50→100
Each session: OpenAI connected, periodic TTS
Duration: 120 seconds per step
Metrics: Same as BM-19 + TTS latency, barge-in latency
Target: p99 AI latency < 600ms at 50 sessions
```

#### BM-21: Memory Stability (Long Run)

```
Setup: 50 WS sessions, continuous bidirectional audio
Duration: 4 hours
Metrics:
  - RSS at start, 1h, 2h, 3h, 4h
  - Heap fragmentation (malloc_info)
  - Pool utilization
Target: RSS growth < 5% over 4 hours
```

#### BM-22: Memory Stability (AI Long Run)

```
Setup: 10 AI sessions, continuous conversation
Duration: 2 hours
Metrics: Same as BM-21 + TTS cache size
Target: RSS growth < 10% (TTS cache bounded)
```

#### BM-23: Spike Test

```
Setup: 100 sessions created in 5 seconds
Duration: 60 seconds active, then all destroyed in 5 seconds
Metrics:
  - Peak CPU during spike
  - Peak RSS during spike
  - RSS after all sessions destroyed (should return to baseline)
  - Any crashes or errors
Target: Graceful handling, RSS returns to within 10% of baseline
```

#### BM-24: Churn Test

```
Setup: Create/destroy sessions at 10/sec for 300 seconds
Metrics:
  - RSS over time (should be sawtooth, not growing)
  - Error count
  - Crash count
Target: Zero crashes, stable RSS envelope
```

---

## 3. CRITICAL PATH LATENCY BUDGETS

### 3.1 Capture Path Budget (20ms frame, 8kHz)

```
┌──────────────────────────────┬─────────┬────────┐
│ Stage                        │ Budget  │ Notes  │
├──────────────────────────────┼─────────┼────────┤
│ RTP frame accumulation       │ 20.0 ms │ Fixed  │
│ capture_callback entry       │  0.01ms │        │
│ switch_mutex_trylock         │  0.05ms │ p99    │
│ shared_ptr copy              │  0.01ms │        │
│ switch_mutex_unlock          │  0.01ms │        │
│ switch_core_media_bug_read   │  0.05ms │        │
│ speex_resampler_process_int  │  0.35ms │ q=7    │
│ writeBinary (WS send)        │  0.50ms │ p99    │
│ ── Total ──                  │ 20.98ms │        │
│ ── Buffer ──                 │  4.02ms │ margin │
└──────────────────────────────┴─────────┴────────┘
```

### 3.2 Inject Path Budget (WS message → RTP frame)

```
┌──────────────────────────────┬─────────┬────────┐
│ Stage                        │ Budget  │ Notes  │
├──────────────────────────────┼─────────┼────────┤
│ WS message receive           │  0.10ms │        │
│ eventCallback overhead       │  0.05ms │        │
│ session_locate               │  0.10ms │        │
│ cJSON_Parse                  │  0.10ms │        │
│ base64_decode                │  0.20ms │ 640B   │
│ byteswap (if needed)         │  0.01ms │        │
│ channel conversion           │  0.01ms │        │
│ speex_resampler_init (first) │  0.00ms │ outside│
│ speex resample               │  0.35ms │ q=7    │
│ frame alignment              │  0.01ms │        │
│ switch_mutex_lock (inject)   │  0.50ms │ p99    │
│ buffer overflow handling     │  0.10ms │        │
│ switch_buffer_write          │  0.05ms │        │
│ switch_mutex_unlock          │  0.01ms │        │
│ ── Queued in buffer ──       │  0.00ms │ async  │
│ WRITE_REPLACE callback       │ 20.0 ms │ wait   │
│ switch_mutex_lock            │  0.05ms │        │
│ switch_buffer_read           │  0.05ms │        │
│ memcpy to frame              │  0.01ms │        │
│ switch_mutex_unlock          │  0.01ms │        │
│ ── Total ──                  │ 21.71ms │        │
│ ── Buffer ──                 │ 18.29ms │ margin │
└──────────────────────────────┴─────────┴────────┘

Note: Total from WS receive to RTP write is ~22ms minimum
(1 frame of buffering). The 40ms target assumes message arrives
mid-frame, averaging to ~30ms.
```

### 3.3 AI Pipeline Budget (text → RTP)

```
┌──────────────────────────────┬──────────┬─────────┐
│ Stage                        │ Budget   │ Notes   │
├──────────────────────────────┼──────────┼─────────┤
│ OpenAI text delta            │   0.0 ms │ trigger │
│ sentence_buffer.add_token    │   0.1 ms │         │
│ sentence detection           │   0.0 ms │         │
│ TTS queue enqueue            │   0.1 ms │         │
│ TTS worker wakeup            │   1.0 ms │ cv wait │
│ TTS cache lookup             │   0.1 ms │ p50 hit │
│ TTS HTTP request (miss)      │ 300.0 ms │ EL API  │
│ TTS streaming chunk          │  50.0 ms │ first   │
│ resample_down                │   0.5 ms │         │
│ DSP pipeline                 │   0.5 ms │         │
│ ring_buffer write            │   0.1 ms │         │
│ ── Ring buffer wait ──       │  20.0 ms │ 1 frame │
│ read_audio in WRITE_REPLACE  │   0.1 ms │         │
│ ── Total (cache miss) ──     │ 372.5 ms │         │
│ ── Total (cache hit) ──      │  22.5 ms │         │
│ ── Buffer ──                 │ 127.5 ms │ margin  │
└──────────────────────────────┴──────────┴─────────┘
```

### 3.4 Barge-In Budget

```
┌──────────────────────────────┬─────────┬────────┐
│ Stage                        │ Budget  │ Notes  │
├──────────────────────────────┼─────────┼────────┤
│ speech_started event         │  0.0 ms │ trigger│
│ on_openai_speech_started     │  0.1 ms │        │
│ has_active_audio check       │  0.1 ms │        │
│ handle_barge_in entry        │  0.1 ms │        │
│ openai_->cancel_response     │  5.0 ms │ WS msg │
│ tts_abort_ = true            │  0.0 ms │ atomic │
│ flush_tts_queue              │  0.5 ms │        │
│ request_flush (ring buffer)  │  0.0 ms │ atomic │
│ sentence_buffer reset        │  0.1 ms │        │
│ sleep(10ms) for TTS drain    │ 10.0 ms │ !!     │
│ tts_abort_ = false           │  0.0 ms │        │
│ ── Consumer flush ──         │ 20.0 ms │ next cb│
│ ── Drain remaining frames ── │ 20.0 ms │ max    │
│ ── Total ──                  │ 55.9 ms │        │
│ ── Buffer ──                 │ 24.1 ms │ margin │
└──────────────────────────────┴─────────┴────────┘

Bottleneck: The 10ms sleep is the largest single contributor.
Consider replacing with an atomic wait/signal pattern.
```

---

## 4. PERFORMANCE REGRESSION GATES

### CI/CD Integration Points

| Gate | Metric | Threshold | Action on Fail |
|------|--------|-----------|----------------|
| PR Gate | BM-17 (callback duration p99) | > 1.5ms | Block merge |
| PR Gate | BM-01 (resample throughput) | < 3M samples/sec | Block merge |
| PR Gate | BM-04 (ring buffer throughput) | < 100ns/pair | Block merge |
| Nightly | BM-19 (scaling p99 @ 200 sessions) | > 30ms | Alert |
| Nightly | BM-21 (4-hour RSS growth) | > 5% | Alert |
| Weekly | BM-23 (spike test crashes) | > 0 | Blocker |
| Release | All BM-* benchmarks | Within targets | Gate release |

### Performance Dashboard Metrics

| Metric | Source | Frequency | Alert Threshold |
|--------|--------|-----------|-----------------|
| Capture p99 latency | BM-11 | Per-commit | > 25ms |
| Inject p99 latency | BM-13 | Per-commit | > 40ms |
| AI first-audio p95 | BM-15 | Nightly | > 500ms |
| Barge-in p95 | BM-16 | Nightly | > 80ms |
| RSS per session | BM-19 | Nightly | > 3MB (WS) / > 10MB (AI) |
| Callback p99 duration | BM-17 | Per-commit | > 1.5ms |
| Mutex contention rate | BM-18 | Nightly | > 1% |

---

## 5. BENCHMARK IMPLEMENTATION NOTES

### 5.1 Instrumentation Points

Add compile-time-gated timing probes at these locations:

```c
/* In capture_callback, wrap the entire WRITE_REPLACE handler */
#ifdef MOD_AUDIO_STREAM_PERF
switch_time_t _cb_start = switch_micro_time_now();
#endif
// ... handler code ...
#ifdef MOD_AUDIO_STREAM_PERF
switch_time_t _cb_end = switch_micro_time_now();
perf_record_callback_us(_cb_end - _cb_start);
#endif
```

Instrument these functions:
1. `capture_callback` — total time per invocation
2. `stream_frame` — time from entry to WS send
3. `processMessage` — time from entry to buffer write
4. `AIEngine::feed_audio` — time including resample
5. `AIEngine::read_audio` — time including flush check
6. `AIEngine::on_tts_audio` — time for resample + DSP + ring buffer write
7. `handle_barge_in` — time from entry to state change

### 5.2 Measurement Tools

| Tool | Purpose | Platform |
|------|---------|----------|
| `switch_micro_time_now()` | In-code timestamps | FreeSWITCH |
| `perf stat` / `perf record` | CPU profiling | Linux |
| `valgrind --tool=massif` | Heap profiling | Linux |
| `valgrind --tool=memcheck` | Memory error detection | Linux |
| `TSAN` (ThreadSanitizer) | Data race detection | GCC/Clang |
| `ASAN` (AddressSanitizer) | Memory error detection | GCC/Clang |
| `/proc/[pid]/status` | RSS monitoring | Linux |
| Custom WS echo server | Latency measurement | Any |

### 5.3 Test Harness Requirements

1. **WS Echo Server**: Simple WebSocket server that echoes timestamps for latency measurement
2. **Audio Generator**: Generates known waveforms (sine, silence, noise) as RTP input
3. **AI Mock Server**: OpenAI Realtime API mock that returns predictable text/audio
4. **TTS Mock Server**: ElevenLabs API mock that returns known PCM audio
5. **Load Generator**: Creates N concurrent FreeSWITCH sessions with configurable parameters
6. **Metrics Collector**: Aggregates timing data, computes percentiles, generates reports

---

*Proceed to Phase 6: Production Hardening.*
