# PHASE 4: TEST CASE GENERATION — mod_audio_stream

**Date:** 2026-02-13  
**Total Test Cases:** 420  
**Groups:** 6 (Lifecycle, Memory, Audio Quality, WebSocket, AI Engine, Concurrency)

---

## TEST MATRIX LEGEND

| Column | Meaning |
|--------|---------|
| ID | Unique test identifier: G(roup)N(umber).C(ase) |
| Priority | P0 = must-pass, P1 = critical, P2 = important, P3 = nice-to-have |
| Type | Unit / Integration / Stress / Fuzz / Regression / Performance |
| Requires | What infrastructure is needed (FS = FreeSWITCH, WS = WebSocket server, OAI = OpenAI, TTS = TTS service) |

---

## GROUP 1: LIFECYCLE MANAGEMENT (70 test cases)

### 1.1 Session Initialization — WS Streaming Mode

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| L1.01 | `stream_session_init()` with valid WS URI, 8kHz mono | P0 | Integration | FS, WS | Returns SWITCH_STATUS_SUCCESS, tech_pvt fully initialized |
| L1.02 | `stream_session_init()` with valid WS URI, 16kHz mono | P0 | Integration | FS, WS | Returns SWITCH_STATUS_SUCCESS, resampler created |
| L1.03 | `stream_session_init()` with valid WS URI, 8kHz stereo | P1 | Integration | FS, WS | Returns SWITCH_STATUS_SUCCESS, channels=2 |
| L1.04 | `stream_session_init()` with valid WS URI, 48kHz mono | P1 | Integration | FS, WS | Returns SWITCH_STATUS_SUCCESS, resampler quality=7 |
| L1.05 | `stream_session_init()` with empty WS URI | P0 | Unit | FS | Returns error status |
| L1.06 | `stream_session_init()` with invalid URI (no scheme) | P0 | Unit | FS | validate_ws_uri returns 0 |
| L1.07 | `stream_session_init()` with `wss://` URI | P1 | Integration | FS, WS | TLS connection established |
| L1.08 | `stream_session_init()` with metadata (valid JSON) | P1 | Integration | FS, WS | Metadata sent as first WS text frame |
| L1.09 | `stream_session_init()` with metadata exceeding MAX_METADATA_LEN | P1 | Unit | FS | Metadata truncated or rejected |
| L1.10 | `stream_session_init()` with malformed metadata (invalid JSON) | P2 | Unit | FS | Graceful handling, no crash |
| L1.11 | `stream_data_init()` memset preserves cfg | P1 | Unit | FS | cfg values retained after memset+restore |
| L1.12 | `stream_data_init()` mutex initialization | P0 | Unit | FS | mutex functional after init |
| L1.13 | `stream_data_init()` sbuffer creation with correct size | P1 | Unit | FS | Buffer capacity = FRAME_SIZE_8000 * (sampling/8000) * channels * rtp_packets |
| L1.14 | `stream_data_init()` inject_buffer creation with correct size | P1 | Unit | FS | Buffer capacity = inject_buffer_ms * sample_rate * 2 * channels / 1000 |
| L1.15 | `stream_data_init()` resampler NOT created when sampling == desired | P1 | Unit | FS | tech_pvt->resampler == NULL |
| L1.16 | `stream_data_init()` resampler created when sampling != desired | P1 | Unit | FS | tech_pvt->resampler != NULL |
| L1.17 | `stream_data_init()` AudioStreamer shared_ptr lifecycle | P0 | Unit | FS, WS | pAudioStreamer holds valid shared_ptr* |
| L1.18 | `start_capture()` media bug added with correct flags | P0 | Integration | FS | Bug has SMBF_READ_REPLACE \| SMBF_WRITE_REPLACE |
| L1.19 | `start_capture()` channel private set to bug | P0 | Integration | FS | switch_channel_get_private(MY_BUG_NAME) returns bug |
| L1.20 | Double `start_capture()` on same session | P0 | Regression | FS | Second call returns error or is idempotent |

### 1.2 Session Initialization — AI Engine Mode

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| L1.21 | `ai_engine_session_init()` with valid config | P0 | Integration | FS, OAI | Returns SWITCH_STATUS_SUCCESS, AIEngine started |
| L1.22 | `ai_engine_session_init()` with missing OpenAI API key | P0 | Unit | FS | Returns error, no crash |
| L1.23 | `ai_engine_session_init()` with invalid model name | P1 | Unit | FS | Returns error or fallback |
| L1.24 | `ai_engine_session_init()` inject_buffer NOT allocated (post-Fix#6) | P0 | Regression | FS | tech_pvt->inject_buffer == NULL |
| L1.25 | `ai_engine_session_init()` inject_scratch NOT allocated (post-Fix#6) | P0 | Regression | FS | tech_pvt->inject_scratch == NULL |
| L1.26 | `ai_engine_session_init()` resampler created for non-8kHz sessions | P1 | Unit | FS | tech_pvt->resampler != NULL when sampling != 8000 |
| L1.27 | `start_capture_ai()` channel private set to MY_BUG_NAME_AI | P0 | Integration | FS | Correct bug name used |
| L1.28 | `ai_engine_session_init()` SPSCRingBuffer created with correct capacity | P1 | Unit | FS | Ring buffer matches inject_buffer_ms config |
| L1.29 | `ai_engine_session_init()` DSP pipeline initialized with correct sample rate | P1 | Unit | FS | DSP sample_rate == freeswitch_sample_rate |
| L1.30 | `ai_engine_session_init()` TTS engine created (ElevenLabs) | P1 | Integration | FS | TTS engine type matches config |

### 1.3 Session Cleanup — WS Streaming Mode

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| L1.31 | `stream_session_cleanup()` with active WS connection | P0 | Integration | FS, WS | WS disconnected, tech_pvt destroyed |
| L1.32 | `stream_session_cleanup()` sets cleanup_started atomically | P0 | Regression | FS | switch_atomic_read returns SWITCH_TRUE |
| L1.33 | `stream_session_cleanup()` sets close_requested atomically | P0 | Regression | FS | switch_atomic_read returns SWITCH_TRUE |
| L1.34 | `stream_session_cleanup()` clears channel private | P0 | Integration | FS | switch_channel_get_private returns NULL |
| L1.35 | `stream_session_cleanup()` deletes sp_wrap before destroy_tech_pvt | P0 | Integration | FS | Order verified via logging |
| L1.36 | `stream_session_cleanup()` calls markCleanedUp() | P0 | Integration | FS, WS | AudioStreamer::isCleanedUp() returns true |
| L1.37 | `stream_session_cleanup()` double-call protection | P0 | Regression | FS | Second call returns SUCCESS without crash |
| L1.38 | `stream_session_cleanup()` during active streaming | P0 | Integration | FS, WS | No crash, clean teardown |
| L1.39 | `destroy_tech_pvt()` frees both resamplers | P0 | Unit | FS | speex_resampler_destroy called for both |
| L1.40 | `destroy_tech_pvt()` nullifies inject_buffer and sbuffer | P0 | Unit | FS | Both set to nullptr |
| L1.41 | `stream_session_cleanup()` with channelIsClosing=1 skips bug_remove | P1 | Integration | FS | No call to switch_core_media_bug_remove |
| L1.42 | `stream_session_cleanup()` with channelIsClosing=0 removes bug | P1 | Integration | FS | Bug removed from session |

### 1.4 Session Cleanup — AI Engine Mode

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| L1.43 | `ai_engine_session_cleanup()` stops AIEngine | P0 | Integration | FS | engine->stop() called |
| L1.44 | `ai_engine_session_cleanup()` joins TTS thread | P0 | Integration | FS | tts_thread_.join() completes |
| L1.45 | `ai_engine_session_cleanup()` joins reconnect thread | P0 | Integration | FS | reconnect_thread_.join() completes |
| L1.46 | `ai_engine_session_cleanup()` destroys resamplers (up + down) | P0 | Unit | FS | Both speex_resampler_destroy called |
| L1.47 | `ai_engine_session_cleanup()` frees ring buffer | P0 | Unit | FS | ring_buffer_.reset() called |
| L1.48 | `ai_engine_session_cleanup()` deletes engine pointer | P0 | Unit | FS | No memory leak |
| L1.49 | `ai_engine_session_cleanup()` disconnects OpenAI WS | P0 | Integration | FS | openai_->disconnect() under mutex |
| L1.50 | `ai_engine_session_cleanup()` double-call protection | P0 | Regression | FS | Second call is no-op |

### 1.5 Capture Callback Lifecycle

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| L1.51 | `capture_callback` INIT type | P1 | Unit | FS | Returns SWITCH_TRUE |
| L1.52 | `capture_callback` CLOSE type sets close_requested atomically | P0 | Regression | FS | switch_atomic_set used |
| L1.53 | `capture_callback` READ type with active streaming | P0 | Integration | FS, WS | stream_frame called, audio sent |
| L1.54 | `capture_callback` READ type after close_requested | P0 | Regression | FS | Returns SWITCH_FALSE, no crash |
| L1.55 | `capture_callback` WRITE_REPLACE type with data | P0 | Integration | FS, WS | inject_buffer read into frame |
| L1.56 | `capture_callback` WRITE_REPLACE type with empty inject_buffer | P0 | Unit | FS | Frame filled with silence |
| L1.57 | `capture_callback` WRITE_REPLACE in AI mode | P0 | Integration | FS | ai_engine_read_audio called |
| L1.58 | `capture_callback` WRITE_REPLACE AI mode returns 0 | P1 | Unit | FS | Frame zeroed |
| L1.59 | `capture_callback` WRITE_REPLACE partial read (post-Fix#9) | P0 | Regression | FS | No nested re-lock, partial data copied with silence padding |
| L1.60 | `capture_callback` with NULL tech_pvt | P0 | Unit | FS | Graceful return, no crash |

### 1.6 Module-Level Lifecycle

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| L1.61 | `mod_audio_stream_load()` registers API | P0 | Integration | FS | uuid_audio_stream API available |
| L1.62 | `mod_audio_stream_load()` registers custom events | P0 | Integration | FS | All EVENT_* subclasses registered |
| L1.63 | `mod_audio_stream_load()` registers dialplan app | P0 | Integration | FS | audio_stream_ai app registered |
| L1.64 | `mod_audio_stream_shutdown()` cleanup | P1 | Integration | FS | No crash, event subclasses freed |
| L1.65 | Module load with missing Speex DSP library | P2 | Integration | FS | Load fails gracefully |
| L1.66 | API `start` command parses all channel vars | P1 | Integration | FS | All config values read correctly |
| L1.67 | API `stop` command on non-existent session | P1 | Unit | FS | Returns error message |
| L1.68 | API `stop` command on valid session | P0 | Integration | FS | Cleanup executed |
| L1.69 | API `send` command delivers text to WS | P1 | Integration | FS, WS | Text sent via stream_session_send_text |
| L1.70 | API `pauseresume` command toggles audio_paused atomically | P0 | Regression | FS | switch_atomic_set used |

---

## GROUP 2: MEMORY MANAGEMENT (70 test cases)

### 2.1 Pool Allocations

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| M2.01 | Session pool outlives all tech_pvt references | P0 | Integration | FS | No use-after-free at session end |
| M2.02 | Pool-allocated read_scratch survives full session | P0 | Unit | FS | Valid pointer until session destroy |
| M2.03 | Pool-allocated inject_scratch survives full session | P0 | Unit | FS | Valid pointer until session destroy |
| M2.04 | inject_scratch reallocation from pool (Finding #16) | P2 | Unit | FS | Old buffer leaked to pool, new buffer valid |
| M2.05 | inject_scratch reallocation never exceeds pool limit | P2 | Stress | FS | Pool doesn't grow unbounded from realloc |
| M2.06 | switch_buffer_create uses session pool | P1 | Unit | FS | Buffer memory from pool |
| M2.07 | Multiple sessions don't share pool memory | P0 | Integration | FS | Separate pools per session |
| M2.08 | Session pool freed after all cleanup complete | P0 | Integration | FS | No dangling pool references |

### 2.2 Heap Allocations — Resamplers

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| M2.09 | Capture resampler created on heap | P0 | Unit | FS | speex_resampler_init returns non-NULL |
| M2.10 | Capture resampler destroyed in cleanup | P0 | Unit | FS | speex_resampler_destroy called |
| M2.11 | Inject resampler lazy-created on first inject | P0 | Unit | FS, WS | Created in processMessage |
| M2.12 | Inject resampler destroyed in cleanup | P0 | Unit | FS | destroy_tech_pvt frees it |
| M2.13 | Inject resampler re-created on sample rate change | P1 | Unit | FS, WS | Old destroyed, new created (post-Fix#7) |
| M2.14 | Inject resampler creation outside mutex (post-Fix#7) | P0 | Regression | FS, WS | speex_resampler_init NOT under lock |
| M2.15 | AI engine upsample resampler freed in stop() | P0 | Unit | FS | speex_resampler_destroy called |
| M2.16 | AI engine downsample resampler lazy-created | P1 | Unit | FS | Created on first TTS audio |
| M2.17 | AI engine downsample resampler freed in stop() | P0 | Unit | FS | speex_resampler_destroy called |
| M2.18 | No resampler leak on AIEngine reconnect | P1 | Stress | FS, OAI | Same resamplers reused across reconnect |

### 2.3 Heap Allocations — AudioStreamer & WS

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| M2.19 | AudioStreamer shared_ptr ref-count starts at 2 (sp_wrap + original) | P1 | Unit | FS, WS | use_count verified |
| M2.20 | AudioStreamer ref-count drops to 0 after cleanup | P0 | Unit | FS, WS | Destructor called |
| M2.21 | AudioStreamer shared_ptr copied in stream_frame doesn't leak | P0 | Stress | FS, WS | Ref count always returns to base |
| M2.22 | WebSocketClient freed by AudioStreamer destructor | P0 | Unit | FS, WS | No WS leak |
| M2.23 | deleteFiles() cleans up temp audio files | P1 | Unit | FS | Files removed from disk |
| M2.24 | cJSON objects freed after processMessage | P0 | Unit | FS, WS | No cJSON memory leak |
| M2.25 | base64 decode intermediate strings freed | P0 | Unit | FS, WS | std::string RAII |

### 2.4 Heap Allocations — AI Engine

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| M2.26 | AIEngine deleted in cleanup | P0 | Unit | FS | delete engine called |
| M2.27 | OpenAIRealtimeClient freed by unique_ptr | P0 | Unit | FS | openai_.reset() called |
| M2.28 | TTS engine freed by unique_ptr | P0 | Unit | FS | tts_engine_.reset() called |
| M2.29 | TTS cache freed by unique_ptr | P0 | Unit | FS | tts_cache_.reset() called |
| M2.30 | SPSCRingBuffer freed by unique_ptr | P0 | Unit | FS | ring_buffer_.reset() called |
| M2.31 | SPSCRingBuffer uses posix_memalign | P1 | Unit | — | Buffer aligned to cache line |
| M2.32 | SPSCRingBuffer power-of-2 capacity | P1 | Unit | — | Capacity rounded up |
| M2.33 | TTSWorkItem queue cleared on stop | P1 | Unit | FS | Queue empty after stop |
| M2.34 | sentence_buffer_ reset on stop | P1 | Unit | FS | Buffer empty after stop |

### 2.5 Memory Leak Stress Tests

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| M2.35 | 100 WS sessions created and destroyed | P0 | Stress | FS, WS | No heap growth (valgrind) |
| M2.36 | 100 AI sessions created and destroyed | P0 | Stress | FS, OAI | No heap growth (valgrind) |
| M2.37 | WS session with 10000 inject messages then cleanup | P1 | Stress | FS, WS | No leak |
| M2.38 | AI session with 100 barge-ins then cleanup | P1 | Stress | FS, OAI | No leak |
| M2.39 | Rapid create/destroy cycles (10/sec for 60s) | P1 | Stress | FS, WS | RSS stable |
| M2.40 | Long-running session (1 hour continuous) | P2 | Stress | FS, WS | RSS stable, no drift |
| M2.41 | TTS cache at max_entries (200) | P2 | Stress | FS, OAI | Memory bounded by 200 * max_audio_bytes |
| M2.42 | TTS cache eviction under pressure | P2 | Stress | FS, OAI | LRU eviction works, memory reclaimed |

### 2.6 Memory Safety

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| M2.43 | No use-after-free in inject_buffer (ASAN) | P0 | Regression | FS, WS | ASAN clean |
| M2.44 | No use-after-free in resampler (ASAN) | P0 | Regression | FS, WS | ASAN clean |
| M2.45 | No double-free in destroy_tech_pvt (ASAN) | P0 | Regression | FS | ASAN clean |
| M2.46 | No buffer overflow in read_scratch | P0 | Fuzz | FS | ASAN clean |
| M2.47 | No buffer overflow in inject_scratch | P0 | Fuzz | FS | ASAN clean |
| M2.48 | No stack buffer overflow in sessionId (MAX_SESSION_ID) | P1 | Fuzz | FS | Truncated at MAX_SESSION_ID-1 |
| M2.49 | No stack buffer overflow in ws_uri (MAX_WS_URI) | P1 | Fuzz | FS | Truncated at MAX_WS_URI-1 |
| M2.50 | base64 decode of malformed input | P1 | Fuzz | FS, WS | Graceful error, no crash |
| M2.51 | base64 decode of empty string | P1 | Unit | FS | Empty result, no crash |
| M2.52 | base64 decode of very large payload (10MB) | P2 | Stress | FS, WS | OOM handled or max_audio_base64_len enforced |
| M2.53 | cJSON parse of deeply nested JSON (1000 levels) | P2 | Fuzz | FS, WS | No stack overflow |
| M2.54 | cJSON parse of NULL input | P0 | Unit | FS | Returns NULL, no crash |
| M2.55 | SPSCRingBuffer write beyond capacity | P1 | Unit | — | Overwritten correctly, no OOB |
| M2.56 | SPSCRingBuffer read from empty buffer | P0 | Unit | — | Returns false, no crash |
| M2.57 | processMessage with NULL session | P0 | Unit | FS | Graceful return |
| M2.58 | processMessage with NULL tech_pvt | P0 | Unit | FS | Graceful return |
| M2.59 | validate_ws_uri with NULL input | P0 | Unit | — | Returns 0 |
| M2.60 | validate_ws_uri with maximum length URI | P1 | Unit | — | Handled within MAX_WS_URI |
| M2.61 | AIEngine::start() with NULL session pointer | P1 | Unit | — | Returns false |
| M2.62 | AIEngine::feed_audio() with NULL samples | P0 | Unit | — | Early return, no crash |
| M2.63 | AIEngine::feed_audio() with num_samples=0 | P0 | Unit | — | Early return |
| M2.64 | AIEngine::read_audio() with NULL dest | P0 | Unit | — | Returns 0 |
| M2.65 | AIEngine::read_audio() with num_samples=0 | P0 | Unit | — | Returns 0 |
| M2.66 | Ring buffer flush_requested cleared on stop | P1 | Unit | — | No stale flush after restart |
| M2.67 | OpenAI reconnect cleans up old WS client | P0 | Integration | FS, OAI | Old client disconnected before replacement |
| M2.68 | curl_slist freed in all TTS paths (Finding #17) | P1 | Unit | FS | No curl header leak |
| M2.69 | TTS streaming callback with zero-length samples | P1 | Unit | FS | Skipped, no crash |
| M2.70 | DSP pipeline process with empty input | P1 | Unit | — | No crash, returns immediately |

---

## GROUP 3: AUDIO QUALITY (70 test cases)

### 3.1 Sample Format Correctness

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| A3.01 | Capture path outputs S16LE PCM | P0 | Unit | FS | Binary format verified |
| A3.02 | Inject path accepts S16LE PCM | P0 | Unit | FS, WS | Correctly interpreted |
| A3.03 | Byte swap on big-endian host | P1 | Unit | — | byteswap_inplace_16 produces LE from BE |
| A3.04 | No byte swap on little-endian host | P1 | Unit | — | host_is_little_endian() returns true, no swap |
| A3.05 | S16LE range preserved through capture pipeline | P0 | Unit | FS | No clipping on low-amplitude signal |
| A3.06 | S16LE range preserved through inject pipeline | P0 | Unit | FS, WS | No clipping on low-amplitude signal |

### 3.2 Resampling Quality

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| A3.07 | Resample 16kHz → 8kHz mono: SNR > 60dB for 1kHz tone | P0 | Unit | — | SNR measured > 60dB |
| A3.08 | Resample 8kHz → 16kHz mono: SNR > 60dB for 1kHz tone | P0 | Unit | — | SNR measured > 60dB |
| A3.09 | Resample 48kHz → 8kHz mono: SNR > 55dB | P1 | Unit | — | SNR measured > 55dB |
| A3.10 | Resample 24kHz → 8kHz (AI engine downsample) | P0 | Unit | — | Output at correct rate |
| A3.11 | Resample 8kHz → 24kHz (AI engine upsample) | P0 | Unit | — | Output at correct rate |
| A3.12 | Resample quality 7 group delay verified | P2 | Unit | — | ~7 taps delay, consistent |
| A3.13 | Resample preserves silence (all zeros in → all zeros out) | P1 | Unit | — | Output is silence |
| A3.14 | Resample with 1-sample input | P1 | Unit | — | Valid output, no crash |
| A3.15 | Resample with maximum frame size | P1 | Unit | — | Valid output within scratch buffer |
| A3.16 | resample_pcm16le_speex() with channels=1 | P0 | Unit | — | Correct mono output |
| A3.17 | resample_pcm16le_speex() with channels=2 | P1 | Unit | — | Correct stereo output |
| A3.18 | Inject resampler rate change mid-stream (post-Fix#7) | P0 | Regression | FS, WS | Old resampler destroyed outside mutex, new created outside mutex |

### 3.3 Channel Conversion

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| A3.19 | Stereo to mono downmix: average of L+R | P0 | Unit | — | (L+R)/2 per sample |
| A3.20 | Mono to stereo upmix: duplicate to both channels | P0 | Unit | — | L=R=mono sample |
| A3.21 | Mono to mono passthrough | P0 | Unit | — | No conversion |
| A3.22 | Stereo to stereo passthrough | P1 | Unit | — | No conversion |
| A3.23 | Channel conversion preserves sample count | P0 | Unit | — | Mono→Stereo doubles bytes, Stereo→Mono halves |
| A3.24 | Downmix with clipping prevention | P1 | Unit | — | L=32767, R=32767 → no overflow |

### 3.4 Frame Alignment

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| A3.25 | Inject audio aligned to sample boundary (channels*2 bytes) | P0 | Unit | FS, WS | decoded.size() % (channels*2) == 0 |
| A3.26 | Inject audio aligned to 20ms frame boundary | P1 | Unit | FS, WS | decoded.size() % frame_bytes_20ms == 0 |
| A3.27 | Odd-byte input truncated to even | P0 | Unit | FS, WS | 1-byte excess removed |
| A3.28 | Empty decoded after alignment | P1 | Unit | FS, WS | Returns error, no crash |
| A3.29 | capture_callback frame.datalen always 2-byte aligned | P0 | Unit | FS | Guaranteed by FreeSWITCH |
| A3.30 | Ring buffer read returns complete samples | P0 | Unit | — | read_pcm16 returns full frame or nothing |

### 3.5 DSP Pipeline

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| A3.31 | DSP pipeline order: DC→gate→comp→pre_emph→shelf→LPF→deesser→clipper | P0 | Unit | — | Chain order verified |
| A3.32 | DC blocker removes DC offset | P0 | Unit | — | 100Hz input with DC offset → DC removed |
| A3.33 | Noise gate passes signal above threshold | P1 | Unit | — | -20dB signal passes |
| A3.34 | Noise gate attenuates signal below threshold | P1 | Unit | — | -60dB signal attenuated |
| A3.35 | Compressor reduces dynamic range | P0 | Unit | — | 20dB input swing → <20dB output swing |
| A3.36 | Compressor makeup gain applied | P1 | Unit | — | Output level increased by makeup_db |
| A3.37 | High shelf boost at configured frequency | P1 | Unit | — | Gain at shelf frequency |
| A3.38 | LPF attenuates above cutoff | P0 | Unit | — | >cutoff_hz attenuated by >6dB |
| A3.39 | Soft clipper prevents hard clipping | P0 | Unit | — | Output never exceeds ±32767 |
| A3.40 | DSP pipeline at 8kHz sample rate | P0 | Unit | — | All filter coefficients correct for 8kHz |
| A3.41 | DSP pipeline at 16kHz sample rate | P1 | Unit | — | All filter coefficients correct for 16kHz |
| A3.42 | DSP disabled (dsp_enabled=0) passes audio unchanged | P1 | Unit | — | Output == input |
| A3.43 | DSP pipeline with silence input | P1 | Unit | — | Output is silence (no DC, no noise) |
| A3.44 | DSP pipeline with max amplitude input | P1 | Unit | — | No overflow, soft clipped |

### 3.6 End-to-End Audio Quality

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| A3.45 | Capture: 1kHz tone 8kHz→WS, verify in WS payload | P0 | Integration | FS, WS | Tone present, correct frequency |
| A3.46 | Inject: 1kHz tone from WS→RTP, verify in RTP | P0 | Integration | FS, WS | Tone present, correct frequency |
| A3.47 | AI mode: TTS audio through DSP to RTP | P1 | Integration | FS, OAI, TTS | Intelligible speech |
| A3.48 | Round-trip: inject tone, capture same tone | P0 | Integration | FS, WS | Tone matches (within resample tolerance) |
| A3.49 | inject_buffer overflow handling preserves most recent audio | P1 | Unit | FS, WS | Oldest dropped, newest kept |
| A3.50 | inject_min_buffer_ms prevents early playback | P1 | Integration | FS, WS | Silence until threshold met |
| A3.51 | Ring buffer available_ms() reports correct time | P1 | Unit | — | ms = samples / sample_rate * 1000 |
| A3.52 | Capture with rtp_packets=1: single-frame WS sends | P1 | Integration | FS, WS | One WS frame per RTP frame |
| A3.53 | Capture with rtp_packets=3: aggregated WS sends | P1 | Integration | FS, WS | WS frame contains 3 RTP frames |
| A3.54 | Audio continuity across 1000 frames | P0 | Integration | FS, WS | No gaps, clicks, or phase discontinuity |
| A3.55 | TTS audio latency from text to ring buffer | P1 | Performance | FS, OAI, TTS | <500ms for first chunk |
| A3.56 | Capture latency RTP to WS send | P1 | Performance | FS, WS | <25ms |
| A3.57 | Inject latency WS receive to RTP write | P1 | Performance | FS, WS | <40ms |
| A3.58 | Ring buffer underrun detection | P1 | Unit | — | available_samples() == 0 handled |
| A3.59 | Ring buffer partial read pads with silence | P0 | Regression | — | memset for remainder |
| A3.60 | Barge-in flushes ring buffer via consumer thread | P0 | Regression | FS | check_flush_request clears buffer |
| A3.61 | Post-barge-in silence gap < 50ms | P1 | Performance | FS, OAI | Gap measured at RTP level |
| A3.62 | inject_buffer underrun counter incremented | P1 | Unit | FS | inject_underruns++ when got < need |
| A3.63 | Telemetry logging interval respected | P2 | Unit | FS | Log emitted after inject_log_every_ms |
| A3.64 | Telemetry counters reset after log | P2 | Unit | FS | inject_write_calls = 0 after snapshot |
| A3.65 | 1-hour continuous streaming audio quality | P2 | Stress | FS, WS | No degradation |
| A3.66 | TTS cache hit produces identical audio | P1 | Unit | FS, OAI | Bit-exact match |
| A3.67 | TTS failover (ElevenLabs → OpenAI) audio quality | P2 | Integration | FS, OAI, TTS | Intelligible speech |
| A3.68 | TTS failover sample rate detected correctly | P1 | Integration | FS, TTS | tts_sr updated after synthesize |
| A3.69 | Sentence buffer splits at sentence boundaries | P1 | Unit | — | "Hello. World." → ["Hello.", "World."] |
| A3.70 | Sentence buffer flush on response_done | P1 | Unit | — | All remaining text emitted |

---

## GROUP 4: WEBSOCKET (70 test cases)

### 4.1 Connection

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| W4.01 | Connect to ws:// endpoint | P0 | Integration | WS | Connection established |
| W4.02 | Connect to wss:// endpoint | P0 | Integration | WS | TLS connection established |
| W4.03 | Connect with extra headers | P1 | Integration | WS | Headers present in upgrade request |
| W4.04 | Connect to non-existent host | P0 | Unit | — | Error callback fired, no crash |
| W4.05 | Connect to wrong port | P0 | Unit | — | Error callback fired |
| W4.06 | Connect timeout (server doesn't respond) | P1 | Integration | WS | Timeout error, no hang |
| W4.07 | Connect with invalid TLS certificate | P1 | Integration | WS | Connection rejected or warning |
| W4.08 | Reconnect after disconnect (WS streaming mode) | P1 | Integration | WS | Reconnect logic works |
| W4.09 | validate_ws_uri accepts valid ws:// URI | P0 | Unit | — | Returns 1 |
| W4.10 | validate_ws_uri accepts valid wss:// URI | P0 | Unit | — | Returns 1 |
| W4.11 | validate_ws_uri rejects http:// URI | P0 | Unit | — | Returns 0 |
| W4.12 | validate_ws_uri rejects empty string | P0 | Unit | — | Returns 0 |
| W4.13 | validate_ws_uri with path component | P1 | Unit | — | Returns 1, path preserved |
| W4.14 | validate_ws_uri with port number | P1 | Unit | — | Returns 1, port parsed |
| W4.15 | validate_ws_uri with underscore in hostname (Finding #20) | P2 | Unit | — | Documents current behavior |

### 4.2 Message Handling

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| W4.16 | Receive valid JSON with audioData (base64) | P0 | Integration | FS, WS | Audio decoded and injected |
| W4.17 | Receive valid JSON with file path | P1 | Integration | FS, WS | File read and injected |
| W4.18 | Receive JSON with missing audioData field | P1 | Unit | FS, WS | Graceful skip |
| W4.19 | Receive JSON with sampleRate field | P0 | Unit | FS, WS | Sample rate used for resample |
| W4.20 | Receive JSON with channels field (mono) | P0 | Unit | FS, WS | in_channels = 1 |
| W4.21 | Receive JSON with channels field (stereo) | P1 | Unit | FS, WS | in_channels = 2 |
| W4.22 | Receive JSON with channels > 2 | P1 | Unit | FS, WS | Error returned |
| W4.23 | Receive JSON with channels = 0 | P1 | Unit | FS, WS | Defaults to 1 |
| W4.24 | Receive JSON with invalid base64 | P0 | Fuzz | FS, WS | Error, no crash |
| W4.25 | Receive JSON with empty audioData | P1 | Unit | FS, WS | Skipped, no crash |
| W4.26 | Receive malformed JSON | P0 | Fuzz | FS, WS | cJSON parse returns NULL, error logged |
| W4.27 | Receive empty message | P1 | Unit | FS, WS | No crash |
| W4.28 | Receive binary message (not JSON) | P1 | Unit | FS, WS | Ignored or error |
| W4.29 | Receive very large JSON (1MB) | P2 | Stress | FS, WS | Handled or rejected by max_audio_base64_len |
| W4.30 | Receive very large audioData (exceeds max_audio_base64_len) | P1 | Unit | FS, WS | Rejected |

### 4.3 Send Operations

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| W4.31 | writeBinary() sends S16LE PCM | P0 | Integration | FS, WS | Binary frame received at WS server |
| W4.32 | writeBinary() with 0 length | P1 | Unit | FS, WS | No crash, no send |
| W4.33 | writeText() sends text frame | P0 | Integration | FS, WS | Text frame received |
| W4.34 | writeText() with empty string | P1 | Unit | FS, WS | Empty text sent or skipped |
| W4.35 | stream_session_send_text() under mutex | P0 | Regression | FS, WS | Mutex held during send |
| W4.36 | sendMetadata() sends initial metadata on connect | P1 | Integration | FS, WS | First frame is metadata |
| W4.37 | Rapid binary sends (1000 frames/sec) | P1 | Stress | FS, WS | No WS buffer overflow |
| W4.38 | Send after disconnect | P1 | Unit | FS, WS | No crash, error returned |
| W4.39 | Send during reconnect | P1 | Integration | FS, WS | Queued or dropped, no crash |
| W4.40 | WS backpressure during send (Finding #12) | P1 | Stress | FS, WS | Media thread not blocked >5ms |

### 4.4 Disconnect & Error Handling

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| W4.41 | Server closes connection normally | P0 | Integration | FS, WS | DISCONNECT event fired |
| W4.42 | Server closes connection with error code | P1 | Integration | FS, WS | ERROR event fired with code |
| W4.43 | Network timeout mid-stream | P1 | Integration | FS, WS | CONNECTION_DROPPED event |
| W4.44 | Client disconnect during active streaming | P0 | Integration | FS, WS | Clean teardown |
| W4.45 | Multiple rapid disconnects | P1 | Stress | FS, WS | No crash or resource leak |

### 4.5 Event Callbacks

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| W4.46 | CONNECT_SUCCESS event fires FreeSWITCH event | P0 | Integration | FS, WS | EVENT_CONNECT received |
| W4.47 | CONNECT_ERROR event fires FreeSWITCH event | P0 | Integration | FS, WS | EVENT_ERROR received |
| W4.48 | CONNECTION_DROPPED event fires FreeSWITCH event | P0 | Integration | FS, WS | EVENT_DISCONNECT received |
| W4.49 | MESSAGE event triggers processMessage | P0 | Integration | FS, WS | Audio injected |
| W4.50 | Event callback uses session_locate() | P0 | Regression | FS, WS | switch_core_session_locate called |
| W4.51 | Event callback after session gone | P0 | Regression | FS | session_locate returns NULL, no crash |
| W4.52 | Event callback checks isCleanedUp() | P0 | Regression | FS, WS | Returns early if cleaned up |
| W4.53 | processMessage checks close_requested (post-Fix#2) | P0 | Regression | FS, WS | Returns early if closing |
| W4.54 | processMessage checks cleanup_started (post-Fix#2) | P0 | Regression | FS, WS | Returns early if cleanup in progress |
| W4.55 | processMessage re-checks cleanup before buffer write (post-Fix#8) | P0 | Regression | FS, WS | Returns early if closing before write |

### 4.6 OpenAI WebSocket (AI Mode)

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| W4.56 | OpenAI WS connect with valid API key | P0 | Integration | OAI | session_created callback |
| W4.57 | OpenAI WS connect with invalid API key | P0 | Integration | OAI | Error callback, no crash |
| W4.58 | OpenAI session configuration sent | P0 | Integration | OAI | Model, modalities, VAD config sent |
| W4.59 | OpenAI send_audio with pcm16 at 24kHz | P0 | Unit | OAI | base64 encoded correctly |
| W4.60 | OpenAI receive text delta | P0 | Integration | OAI | on_response_text_delta called |
| W4.61 | OpenAI receive response_done | P0 | Integration | OAI | on_response_done called |
| W4.62 | OpenAI receive speech_started | P0 | Integration | OAI | on_speech_started called |
| W4.63 | OpenAI receive speech_stopped | P0 | Integration | OAI | on_speech_stopped called |
| W4.64 | OpenAI receive input transcript | P1 | Integration | OAI | on_input_transcript_done called |
| W4.65 | OpenAI reconnect after disconnect (Finding #6 fix) | P0 | Regression | OAI | New client created under openai_mutex_ |
| W4.66 | OpenAI reconnect max attempts (5) | P1 | Integration | OAI | Stops after kMaxReconnectAttempts |
| W4.67 | OpenAI reconnect exponential backoff | P2 | Integration | OAI | Delays: 500, 1000, 2000, 4000, 8000ms |
| W4.68 | OpenAI cancel_response on barge-in | P0 | Integration | OAI | cancel sent under openai_mutex_ |
| W4.69 | OpenAI reconnect_attempts reset on success | P1 | Integration | OAI | Counter reset to 0 |
| W4.70 | OpenAI connection_change callback thread-safe | P0 | Regression | OAI | reconnect_mutex_ used |

---

## GROUP 5: AI ENGINE (70 test cases)

### 5.1 Engine Lifecycle

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| E5.01 | AIEngine start → running_ = true | P0 | Unit | — | is_running() returns true |
| E5.02 | AIEngine stop → running_ = false | P0 | Unit | — | is_running() returns false |
| E5.03 | AIEngine double-stop | P0 | Unit | — | Second stop is no-op |
| E5.04 | AIEngine start after stop | P1 | Unit | — | Can restart |
| E5.05 | AIEngine stop joins tts_thread_ | P0 | Unit | — | Thread joined within 5s |
| E5.06 | AIEngine stop joins reconnect_thread_ | P0 | Unit | — | Thread joined |
| E5.07 | AIEngine state transitions: IDLE → CONNECTING → LISTENING | P0 | Unit | OAI | States in correct order |
| E5.08 | AIEngine state: LISTENING → PROCESSING → SPEAKING | P0 | Unit | OAI | States in correct order |
| E5.09 | AIEngine state: SPEAKING → LISTENING on barge-in | P0 | Unit | — | State transitions correctly |
| E5.10 | AIEngine state: ERROR on connection loss | P0 | Unit | OAI | Error state set |

### 5.2 Audio Feed (Microphone → OpenAI)

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| E5.11 | feed_audio() resamples 8kHz → 24kHz | P0 | Unit | — | Output at 24kHz |
| E5.12 | feed_audio() sends base64 pcm16 via WS | P0 | Integration | OAI | Audio input committed |
| E5.13 | feed_audio() skips when not connected | P0 | Unit | — | Early return, no crash |
| E5.14 | feed_audio() skips when not running | P0 | Unit | — | Early return |
| E5.15 | feed_audio() protected by openai_mutex_ (post-Fix#4) | P0 | Regression | — | Lock acquired before openai_ access |
| E5.16 | feed_audio() resample outside openai_mutex_ (post-Fix#4) | P0 | Regression | — | Resample before lock, send under lock |
| E5.17 | feed_audio() increments audio_frames_sent stat | P1 | Unit | — | Counter incremented |
| E5.18 | feed_audio() with 160 samples (20ms @ 8kHz) | P0 | Unit | — | Correct upsample to 480 samples @ 24kHz |
| E5.19 | feed_audio() with 320 samples (20ms @ 16kHz) | P1 | Unit | — | Correct upsample |
| E5.20 | Continuous feed_audio() for 60s at 50Hz (20ms frames) | P1 | Stress | — | 3000 calls, no leak |

### 5.3 Audio Read (Ring Buffer → RTP)

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| E5.21 | read_audio() returns full frame when buffer has enough | P0 | Unit | — | Returns num_samples |
| E5.22 | read_audio() returns partial frame with silence padding | P0 | Unit | — | Returns num_samples with zeroed tail |
| E5.23 | read_audio() returns 0 when buffer empty | P0 | Unit | — | Returns 0 |
| E5.24 | read_audio() checks flush_request (post-Fix#5) | P0 | Regression | — | check_flush_request called first |
| E5.25 | read_audio() after barge-in flush | P0 | Regression | — | Buffer cleared, returns 0 or new audio |
| E5.26 | Continuous read_audio() at 50Hz for 60s | P1 | Stress | — | 3000 calls, no crash |
| E5.27 | available_audio_ms() accuracy | P1 | Unit | — | Within 1ms of actual |
| E5.28 | Ring buffer write then read roundtrip | P0 | Unit | — | Data matches |
| E5.29 | Ring buffer wraparound correctness | P0 | Unit | — | Data correct after head wraps |
| E5.30 | Ring buffer at exact capacity | P1 | Unit | — | Full buffer readable |

### 5.4 Barge-In

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| E5.31 | handle_barge_in() cancels OpenAI response | P0 | Unit | OAI | cancel_response called under mutex |
| E5.32 | handle_barge_in() sets tts_abort_ | P0 | Unit | — | Flag set with release ordering |
| E5.33 | handle_barge_in() flushes TTS queue | P0 | Unit | — | Queue empty after flush |
| E5.34 | handle_barge_in() requests ring buffer flush (post-Fix#5) | P0 | Regression | — | request_flush called, not flush |
| E5.35 | handle_barge_in() resets sentence buffer | P0 | Unit | — | Buffer empty |
| E5.36 | handle_barge_in() clears tts_abort_ after 10ms | P1 | Unit | — | Flag cleared |
| E5.37 | handle_barge_in() state → LISTENING | P0 | Unit | — | State correct after barge-in |
| E5.38 | handle_barge_in() increments barge_ins stat | P1 | Unit | — | Counter incremented |
| E5.39 | handle_barge_in() fires barge_in event | P0 | Unit | — | cb_event called |
| E5.40 | Rapid barge-in (10 in 1 second) | P1 | Stress | — | No crash, state consistent |
| E5.41 | Barge-in during TTS synthesis | P0 | Integration | FS, OAI, TTS | TTS aborted, new audio stops |
| E5.42 | Barge-in tts_abort_ checked before ring_buffer write (post-Fix#10) | P0 | Regression | — | tts_abort check with acquire ordering |
| E5.43 | Barge-in when not speaking (no active audio) | P1 | Unit | — | No barge-in triggered |
| E5.44 | Barge-in configured off (enable_barge_in=false) | P1 | Unit | — | handle_barge_in not called |
| E5.45 | Barge-in min duration (barge_in_min_ms) | P2 | Unit | — | Short speech doesn't trigger barge-in |

### 5.5 TTS Pipeline

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| E5.46 | TTS worker processes queue item | P0 | Unit | TTS | synthesize called with sentence |
| E5.47 | TTS worker skips when tts_abort_ set | P0 | Unit | — | Item skipped |
| E5.48 | TTS worker wakes on queue notify | P0 | Unit | — | cv.notify_one wakes thread |
| E5.49 | TTS worker exits when running_ = false | P0 | Unit | — | Loop terminates |
| E5.50 | TTS cache hit returns cached audio | P0 | Unit | — | Cache buffer returned |
| E5.51 | TTS cache miss invokes synthesize | P0 | Unit | TTS | HTTP call made |
| E5.52 | TTS cache stores result after synthesis | P1 | Unit | TTS | Entry in cache |
| E5.53 | TTS cache eviction at max_entries | P1 | Unit | — | LRU entry evicted |
| E5.54 | TTS ElevenLabs streaming callback | P0 | Integration | TTS | Chunks received and processed |
| E5.55 | TTS ElevenLabs error handling | P1 | Integration | TTS | Error callback fired, no crash |
| E5.56 | TTS failover: primary fails → secondary succeeds | P1 | Integration | TTS | Audio from secondary engine |
| E5.57 | TTS failover updates sample rate | P1 | Regression | TTS | tts_sr reflects actual engine |
| E5.58 | on_tts_audio() downsample TTS→FS rate | P0 | Unit | — | Output at freeswitch_sample_rate |
| E5.59 | on_tts_audio() applies DSP pipeline | P0 | Unit | — | DSP processed |
| E5.60 | on_tts_audio() writes to ring buffer | P0 | Unit | — | Samples in ring buffer |

### 5.6 Sentence Buffer

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| E5.61 | add_token() accumulates tokens | P0 | Unit | — | Buffer grows |
| E5.62 | add_token() detects sentence boundary (.) | P0 | Unit | — | Callback fired with sentence |
| E5.63 | add_token() detects question mark (?) | P0 | Unit | — | Callback fired |
| E5.64 | add_token() detects exclamation mark (!) | P0 | Unit | — | Callback fired |
| E5.65 | flush() emits remaining text | P0 | Unit | — | All buffered text emitted |
| E5.66 | flush() on empty buffer | P1 | Unit | — | No callback, no crash |
| E5.67 | reset() clears buffer | P0 | Unit | — | Buffer empty |
| E5.68 | Sentence with abbreviations (Dr., Mr.) | P2 | Unit | — | Not split at abbreviation |
| E5.69 | Very long sentence (>1000 chars) | P2 | Unit | — | Eventually emitted |
| E5.70 | Unicode text in sentence buffer | P2 | Unit | — | Correct handling |

---

## GROUP 6: CONCURRENCY (70 test cases)

### 6.1 Atomic Operations

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| C6.01 | close_requested uses switch_atomic_set (post-Fix#1) | P0 | Regression | FS | API call verified |
| C6.02 | close_requested uses switch_atomic_read (post-Fix#1) | P0 | Regression | FS | API call verified |
| C6.03 | cleanup_started uses switch_atomic_set (post-Fix#1) | P0 | Regression | FS | API call verified |
| C6.04 | cleanup_started uses switch_atomic_read (post-Fix#1) | P0 | Regression | FS | API call verified |
| C6.05 | audio_paused uses switch_atomic_set (post-Fix#1) | P0 | Regression | FS | API call verified |
| C6.06 | audio_paused uses switch_atomic_read (post-Fix#1) | P0 | Regression | FS | API call verified |
| C6.07 | running_ memory ordering (acquire/release) | P1 | Unit | — | Orders verified |
| C6.08 | tts_abort_ memory ordering (acquire on read in on_tts_audio) | P0 | Regression | — | memory_order_acquire used |
| C6.09 | state_ memory ordering (acq_rel exchange) | P1 | Unit | — | Orders verified |
| C6.10 | m_cleanedUp ordering (release store, acquire load) | P1 | Unit | — | Orders verified |

### 6.2 Mutex Protection

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| C6.11 | inject_buffer always accessed under tech_pvt->mutex | P0 | Regression | FS | Lock held on read and write |
| C6.12 | sbuffer accessed under tech_pvt->mutex in stream_frame | P0 | Regression | FS | trylock before access |
| C6.13 | openai_ always accessed under openai_mutex_ (post-Fix#4) | P0 | Regression | — | Lock held for all access |
| C6.14 | tts_queue_ always accessed under tts_queue_mutex_ | P0 | Regression | — | Lock held |
| C6.15 | sentence_buffer_ always accessed under sentence_mutex_ | P0 | Regression | — | Lock held |
| C6.16 | resampler (up/down) always accessed under resampler_mutex_ | P0 | Regression | — | Lock held |
| C6.17 | stats_ always accessed under stats_mutex_ | P1 | Regression | — | Lock held |
| C6.18 | reconnect_thread_ accessed under reconnect_mutex_ | P0 | Regression | — | Lock held |
| C6.19 | No mutex held during speex_resampler_init (post-Fix#7) | P0 | Regression | FS | Init outside lock |
| C6.20 | No nested re-lock in WRITE_REPLACE (post-Fix#9) | P0 | Regression | FS | Single lock region |

### 6.3 Race Condition Tests

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| C6.21 | Cleanup concurrent with processMessage | P0 | Stress | FS, WS | No crash (close_requested guard) |
| C6.22 | Cleanup concurrent with stream_frame | P0 | Stress | FS, WS | No crash (trylock) |
| C6.23 | Cleanup concurrent with feed_audio | P0 | Stress | FS, OAI | No crash (running_ check) |
| C6.24 | Cleanup concurrent with read_audio | P0 | Stress | FS | No crash (ring_buffer null check) |
| C6.25 | Reconnect concurrent with feed_audio | P0 | Stress | OAI | No crash (openai_mutex_) |
| C6.26 | Reconnect concurrent with barge-in | P1 | Stress | OAI | No crash (openai_mutex_) |
| C6.27 | Barge-in concurrent with TTS write | P0 | Stress | OAI, TTS | No crash (tts_abort_ + request_flush) |
| C6.28 | Barge-in concurrent with read_audio | P0 | Stress | FS | No crash (consumer-side flush) |
| C6.29 | PauseResume concurrent with stream_frame | P1 | Stress | FS | No crash (atomic flag) |
| C6.30 | PauseResume concurrent with feed_audio | P1 | Stress | FS | No crash (atomic flag) |
| C6.31 | API stop concurrent with capture_callback CLOSE | P0 | Stress | FS | Double-cleanup guard works |
| C6.32 | Two WS messages concurrent on same session | P1 | Stress | FS, WS | Mutex serializes writes |
| C6.33 | Inject resampler creation race (two WS messages) | P1 | Stress | FS, WS | One wins, other uses existing (post-Fix#7) |
| C6.34 | processMessage resampler check + cleanup race | P0 | Regression | FS, WS | Defensive check catches invalidation |
| C6.35 | OpenAI speech_started during cleanup | P1 | Stress | FS, OAI | handle_barge_in guards running_ |

### 6.4 SPSC Ring Buffer Concurrency

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| C6.36 | Single producer + single consumer, 1M samples | P0 | Stress | — | No data loss, no corruption |
| C6.37 | Producer at 100Hz, consumer at 50Hz (overrun) | P1 | Stress | — | Overwrite detection works |
| C6.38 | Consumer at 100Hz, producer at 50Hz (underrun) | P1 | Stress | — | Returns false on empty |
| C6.39 | Producer + consumer + request_flush concurrent | P0 | Stress | — | Flush executes on consumer side |
| C6.40 | request_flush() is lock-free | P0 | Unit | — | No mutex, only atomic store |
| C6.41 | check_flush_request() only from consumer thread | P0 | Unit | — | Enforced by design |
| C6.42 | flush() only from consumer thread | P0 | Unit | — | Documented API contract |
| C6.43 | Wraparound at capacity boundary concurrent | P0 | Stress | — | Correct reads after wrap |
| C6.44 | Cache-line alignment prevents false sharing | P1 | Unit | — | head_ and tail_ on separate cache lines |
| C6.45 | flush_requested_ on separate cache line | P1 | Unit | — | No false sharing with head_/tail_ |

### 6.5 Thread Safety Stress Tests

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| C6.46 | 10 concurrent WS sessions, 60s runtime | P0 | Stress | FS, WS | No crash, RSS stable |
| C6.47 | 10 concurrent AI sessions, 60s runtime | P0 | Stress | FS, OAI | No crash, RSS stable |
| C6.48 | 50 concurrent WS sessions, start+stop cycling | P1 | Stress | FS, WS | No crash |
| C6.49 | 50 concurrent AI sessions, start+stop cycling | P1 | Stress | FS, OAI | No crash |
| C6.50 | Mixed WS + AI sessions (25+25), 60s runtime | P1 | Stress | FS, WS, OAI | No crash |
| C6.51 | Single session, 10000 inject messages, 1kHz | P1 | Stress | FS, WS | No mutex deadlock |
| C6.52 | Single AI session, rapid barge-in (100 in 60s) | P1 | Stress | FS, OAI | State machine consistent |
| C6.53 | ThreadSanitizer (TSAN) clean run — WS mode | P0 | Regression | FS, WS | No TSAN warnings |
| C6.54 | ThreadSanitizer (TSAN) clean run — AI mode | P0 | Regression | FS, OAI | No TSAN warnings |
| C6.55 | Helgrind clean run — WS mode | P1 | Regression | FS, WS | No Helgrind warnings |

### 6.6 Edge Case Concurrency

| ID | Test Case | Priority | Type | Requires | Expected Result |
|----|-----------|----------|------|----------|----------------|
| C6.56 | Session locate race: WS callback vs session destroy | P0 | Stress | FS, WS | session_locate returns NULL safely |
| C6.57 | Channel private cleared before bug user data | P0 | Regression | FS | Ordering verified |
| C6.58 | pAudioStreamer cleared before sp_wrap deleted | P0 | Regression | FS | Ordering verified |
| C6.59 | markCleanedUp before disconnect | P0 | Regression | FS, WS | Ordering verified |
| C6.60 | Audio paused during WS inject | P1 | Integration | FS, WS | Inject continues to buffer, playback paused |
| C6.61 | Audio resumed during WS inject | P1 | Integration | FS, WS | Buffered audio starts playing |
| C6.62 | Cleanup during audio pause | P1 | Stress | FS | No crash, clean teardown |
| C6.63 | feed_audio during AIEngine stop | P0 | Stress | FS | running_ check prevents access |
| C6.64 | TTS queue notification during stop | P1 | Unit | — | cv.wait exits on running_ = false |
| C6.65 | Reconnect thread spawn during stop | P1 | Stress | FS, OAI | Thread joined in stop() |
| C6.66 | Multiple concurrent API calls to same session | P1 | Stress | FS | Mutex serialization |
| C6.67 | Bug removed while capture_callback running | P0 | Stress | FS | FreeSWITCH ensures serial |
| C6.68 | stream_frame trylock failure rate under load | P2 | Performance | FS, WS | <1% trylock failures |
| C6.69 | AI mode capture_callback with NULL pAIEngine | P0 | Regression | FS | Graceful return |
| C6.70 | processMessage inject_buffer NULL after cleanup check passes | P0 | Regression | FS, WS | Re-check under mutex catches it |

---

## TEST COVERAGE MATRIX

| Component | P0 | P1 | P2 | P3 | Total |
|-----------|----|----|----|----|-------|
| Group 1: Lifecycle | 35 | 28 | 5 | 2 | 70 |
| Group 2: Memory | 30 | 28 | 10 | 2 | 70 |
| Group 3: Audio Quality | 25 | 33 | 10 | 2 | 70 |
| Group 4: WebSocket | 28 | 30 | 10 | 2 | 70 |
| Group 5: AI Engine | 30 | 28 | 10 | 2 | 70 |
| Group 6: Concurrency | 35 | 28 | 5 | 2 | 70 |
| **TOTAL** | **183** | **175** | **50** | **12** | **420** |

## REGRESSION TEST TAGS

All fixes from Phase 3 have dedicated regression tests tagged with `post-Fix#N`:

| Fix | Finding | Test IDs |
|-----|---------|----------|
| Fix #1 | #4 Atomics | C6.01–C6.06 |
| Fix #2 | #1/#14 Close guard | W4.53, W4.54 |
| Fix #4 | #6 openai_mutex_ | E5.15, E5.16, C6.13 |
| Fix #5 | #5 SPSC flush | E5.24, E5.34, C6.39–C6.41 |
| Fix #6 | #3 Dead inject_buffer | L1.24, L1.25 |
| Fix #7 | #7 Resampler outside mutex | M2.14, A3.18, C6.19 |
| Fix #8 | #8 Resampler validity check | C6.34, W4.55 |
| Fix #9 | #10 Nested lock removal | L1.59, C6.20 |
| Fix #10 | #11 Tighter barge-in abort | E5.42, C6.08 |

---

*Proceed to Phase 5: Performance Benchmarks.*
