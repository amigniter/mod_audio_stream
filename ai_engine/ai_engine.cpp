#include "ai_engine.h"
#include <speex/speex_resampler.h>
#ifdef HAVE_SWITCH_H
#include <switch.h>
#define AI_LOG_INFO(fmt, ...)  switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, fmt, ##__VA_ARGS__)
#define AI_LOG_ERROR(fmt, ...) switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR, fmt, ##__VA_ARGS__)
#define AI_LOG_DEBUG(fmt, ...) switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, fmt, ##__VA_ARGS__)
#else
#include <cstdio>
#define AI_LOG_INFO(fmt, ...)  fprintf(stderr, "[AI-INFO] " fmt, ##__VA_ARGS__)
#define AI_LOG_ERROR(fmt, ...) fprintf(stderr, "[AI-ERROR] " fmt, ##__VA_ARGS__)
#define AI_LOG_DEBUG(fmt, ...) fprintf(stderr, "[AI-DEBUG] " fmt, ##__VA_ARGS__)
#endif
#include <chrono>
#include <algorithm>
#include <cstring>
#define RESAMPLE_QUALITY 7

namespace ai_engine {

AIEngine::AIEngine() = default;

AIEngine::~AIEngine() {
    stop();
}

bool AIEngine::start(const AIEngineConfig& cfg, switch_core_session_t*) {
    if (running_.load(std::memory_order_relaxed)) {
        AI_LOG_ERROR("(%s) AIEngine::start — already running\n", cfg.session_uuid.c_str());
        return false;
    }

    cfg_ = cfg;
    set_state(AIEngineState::CONNECTING, "Initializing AI engine");

    AI_LOG_INFO("(%s) AIEngine starting: model=%s tts=%s sr=%d\n",
                cfg_.session_uuid.c_str(),
                cfg_.openai.model.c_str(),
                cfg_.tts.provider.c_str(),
                cfg_.freeswitch_sample_rate);

    {
        size_t ring_bytes = (size_t)cfg_.freeswitch_sample_rate * 2 *
                            cfg_.inject_buffer_ms / 1000;
        ring_buffer_ = std::make_unique<SPSCRingBuffer>(ring_bytes);
        AI_LOG_INFO("(%s) Ring buffer: %zu bytes (%.1f ms)\n",
                    cfg_.session_uuid.c_str(),
                    ring_buffer_->capacity(),
                    (double)ring_buffer_->capacity() / (cfg_.freeswitch_sample_rate * 2) * 1000.0);
    }

    {
        DSPConfig dsp_cfg = cfg_.dsp;
        dsp_cfg.sample_rate = cfg_.freeswitch_sample_rate;
        dsp_.init(dsp_cfg);
        AI_LOG_DEBUG("(%s) DSP pipeline initialized at %d Hz\n",
                     cfg_.session_uuid.c_str(), cfg_.freeswitch_sample_rate);
    }

    {
        tts_engine_ = create_tts_engine(cfg_.tts);
        if (!tts_engine_) {
            AI_LOG_ERROR("(%s) Failed to create TTS engine\n", cfg_.session_uuid.c_str());
            set_state(AIEngineState::ERROR, "TTS engine creation failed");
            return false;
        }
        AI_LOG_INFO("(%s) TTS engine: %s (output_sr=%d)\n",
                    cfg_.session_uuid.c_str(),
                    tts_engine_->name(),
                    tts_engine_->output_sample_rate());
    }

    if (cfg_.tts.enable_cache) {
        tts_cache_ = std::make_unique<TTSCache>(cfg_.cache);
        AI_LOG_DEBUG("(%s) TTS cache enabled: max=%zu ttl=%ds\n",
                     cfg_.session_uuid.c_str(),
                     cfg_.cache.max_entries,
                     cfg_.cache.ttl_seconds);
    }

    sentence_buffer_ = SentenceBuffer(cfg_.sentence);

    {
        int err = 0;

        if (cfg_.freeswitch_sample_rate != cfg_.openai_send_rate) {
            upsample_resampler_ = speex_resampler_init(
                1,
                cfg_.freeswitch_sample_rate,
                cfg_.openai_send_rate,
                RESAMPLE_QUALITY,
                &err
            );
            if (err != 0 || !upsample_resampler_) {
                AI_LOG_ERROR("(%s) Failed to init upsample resampler: %s\n",
                             cfg_.session_uuid.c_str(),
                             speex_resampler_strerror(err));
                set_state(AIEngineState::ERROR, "Resampler init failed");
                return false;
            }
            AI_LOG_DEBUG("(%s) Upsample resampler: %d → %d Hz\n",
                         cfg_.session_uuid.c_str(),
                         cfg_.freeswitch_sample_rate,
                         cfg_.openai_send_rate);
        }

    }

    {
        openai_ = std::make_unique<OpenAIRealtimeClient>();

        openai_->on_session_created([this](const std::string& sid) {
            on_openai_connected(sid);
        });
        openai_->on_response_text_delta([this](const std::string& delta, const std::string& rid) {
            on_openai_text_delta(delta, rid);
        });
        openai_->on_response_done([this](const std::string& text, const std::string& rid) {
            on_openai_response_done(text, rid);
        });
        openai_->on_speech_started([this]() {
            on_openai_speech_started();
        });
        openai_->on_speech_stopped([this]() {
            on_openai_speech_stopped();
        });
        openai_->on_input_transcript_done([this](const std::string& transcript) {
            on_openai_input_transcript(transcript);
        });
        openai_->on_error([this](const std::string& error, const std::string& code) {
            on_openai_error(error, code);
        });
        openai_->on_connection_change([this](bool connected) {
            on_openai_connection_change(connected);
        });

        openai_->connect(cfg_.openai);
    }

    {
        running_.store(true, std::memory_order_release);
        tts_abort_.store(false, std::memory_order_release);

        tts_thread_ = std::thread([this]() {
            tts_worker_loop();
        });

        AI_LOG_INFO("(%s) TTS worker thread started\n", cfg_.session_uuid.c_str());
    }

    return true;
}

void AIEngine::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
        return; 
    }

    AI_LOG_INFO("(%s) AIEngine stopping...\n", cfg_.session_uuid.c_str());
    set_state(AIEngineState::SHUTDOWN, "Shutting down");

    tts_abort_.store(true, std::memory_order_release);

    {
        std::lock_guard<std::mutex> lock(tts_queue_mutex_);
        tts_queue_cv_.notify_all();
    }

    if (tts_thread_.joinable()) {
        tts_thread_.join();
    }

    /* Join any in-flight reconnect thread */
    {
        std::lock_guard<std::mutex> rlock(reconnect_mutex_);
        if (reconnect_thread_.joinable()) {
            reconnect_thread_.join();
        }
    }

    if (openai_) {
        std::lock_guard<std::mutex> lock(openai_mutex_);
        openai_->disconnect();
        openai_.reset();
    }

    {
        std::lock_guard<std::mutex> lock(resampler_mutex_);
        if (upsample_resampler_) {
            speex_resampler_destroy(upsample_resampler_);
            upsample_resampler_ = nullptr;
        }
        if (downsample_resampler_) {
            speex_resampler_destroy(downsample_resampler_);
            downsample_resampler_ = nullptr;
        }
    }

    tts_engine_.reset();
    tts_cache_.reset();
    ring_buffer_.reset();
    sentence_buffer_.reset();

    AI_LOG_INFO("(%s) AIEngine stopped\n", cfg_.session_uuid.c_str());
}

bool AIEngine::is_running() const {
    return running_.load(std::memory_order_acquire);
}

void AIEngine::feed_audio(const int16_t* samples, size_t num_samples) {
    if (!running_.load(std::memory_order_relaxed)) return;
    if (!samples || num_samples == 0) return;

    /* Resample under resampler_mutex_ (already protected) */
    std::vector<int16_t> upsampled;
    if (upsample_resampler_) {
        resample_up(samples, num_samples, upsampled);
        if (upsampled.empty()) return;
    }

    /* Access openai_ under openai_mutex_ to prevent race with reconnect thread */
    {
        std::lock_guard<std::mutex> lock(openai_mutex_);
        if (!openai_ || !openai_->is_connected()) return;
        if (!upsampled.empty()) {
            openai_->send_audio(upsampled.data(), upsampled.size());
        } else {
            openai_->send_audio(samples, num_samples);
        }
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.audio_frames_sent++;
    }
}

size_t AIEngine::read_audio(int16_t* dest, size_t num_samples) {
    if (!ring_buffer_ || !dest || num_samples == 0) return 0;

    /* Check if barge-in requested a flush — execute from consumer thread */
    ring_buffer_->check_flush_request();

    /* Try to read exactly the requested number of samples */
    if (ring_buffer_->read_pcm16(dest, num_samples)) {
        return num_samples;
    }

    /*
     * If we couldn't get a full frame, try to read whatever is available.
     * This prevents audio gaps when the ring buffer has partial data.
     */
    size_t avail = ring_buffer_->available_samples();
    if (avail > 0 && avail < num_samples) {
        if (ring_buffer_->read_pcm16(dest, avail)) {
            /* Zero-fill the remainder */
            memset(dest + avail, 0, (num_samples - avail) * sizeof(int16_t));
            return num_samples;
        }
    }

    return 0;
}

double AIEngine::available_audio_ms() const {
    if (!ring_buffer_) return 0.0;
    return ring_buffer_->available_ms(cfg_.freeswitch_sample_rate);
}

void AIEngine::handle_barge_in() {
    if (!running_.load(std::memory_order_relaxed)) return;

    AI_LOG_INFO("(%s) Barge-in triggered\n", cfg_.session_uuid.c_str());

    {
        std::lock_guard<std::mutex> lock(openai_mutex_);
        if (openai_) {
            openai_->cancel_response();
        }
    }

    /* Set abort flag — TTS worker will see this and skip current work */
    tts_abort_.store(true, std::memory_order_release);

    flush_tts_queue();

    /*
     * Request flush via atomic flag — the consumer (media thread)
     * will execute the actual flush on its next read_audio() call.
     * This avoids violating the SPSC contract by modifying the tail
     * from a non-consumer thread.
     */
    if (ring_buffer_) {
        ring_buffer_->request_flush();
    }

    {
        std::lock_guard<std::mutex> slock(sentence_mutex_);
        sentence_buffer_.reset();
    }

    /*
     * Give the TTS worker time to observe the abort flag before clearing it.
     * The worker checks tts_abort_ at the top of each iteration and before
     * each TTS chunk callback, so a short sleep is sufficient.
     */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    tts_abort_.store(false, std::memory_order_release);

    set_state(AIEngineState::LISTENING, "Barge-in — listening");

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.barge_ins++;
    }

    if (cb_event_) {
        cb_event_("barge_in", "{}");
    }
}

void AIEngine::on_openai_connected(const std::string& session_id) {
    AI_LOG_INFO("(%s) OpenAI session created: %s\n",
                cfg_.session_uuid.c_str(), session_id.c_str());
    set_state(AIEngineState::LISTENING, "Connected — listening for speech");

    if (cb_event_) {
        cb_event_("openai_connected", "{\"session_id\":\"" + json_escape(session_id) + "\"}");
    }
}

void AIEngine::on_openai_text_delta(const std::string& delta, const std::string& ) {
    if (!running_.load(std::memory_order_relaxed)) return;

    if (cfg_.debug_logging) {
        AI_LOG_DEBUG("(%s) Text delta: '%s'\n", cfg_.session_uuid.c_str(), delta.c_str());
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.text_tokens_received++;
    }

    {
        std::lock_guard<std::mutex> slock(sentence_mutex_);
        sentence_buffer_.add_token(delta, [this](const std::string& sentence, bool is_final) {
            if (sentence.empty()) return;
            if (tts_abort_.load(std::memory_order_relaxed)) return;

            AI_LOG_INFO("(%s) Sentence ready for TTS: '%s' (final=%d)\n",
                        cfg_.session_uuid.c_str(), sentence.c_str(), is_final);

            TTSWorkItem item;
            item.sentence = sentence;
            item.is_final = is_final;
            item.sequence = tts_sequence_.fetch_add(1, std::memory_order_relaxed);

            {
                std::lock_guard<std::mutex> lock(tts_queue_mutex_);
                tts_queue_.push(std::move(item));
            }
            tts_queue_cv_.notify_one();

            set_state(AIEngineState::PROCESSING, "Synthesizing speech");
        });
    }
}

void AIEngine::on_openai_response_done(const std::string& full_text,
                                        const std::string& ) {
    AI_LOG_INFO("(%s) OpenAI response done: %zu chars\n",
                cfg_.session_uuid.c_str(), full_text.size());

    {
        std::lock_guard<std::mutex> slock(sentence_mutex_);
        sentence_buffer_.flush([this](const std::string& sentence, bool ) {
            if (sentence.empty()) return;
            if (tts_abort_.load(std::memory_order_relaxed)) return;

            AI_LOG_INFO("(%s) Flushed sentence for TTS: '%s'\n",
                        cfg_.session_uuid.c_str(), sentence.c_str());

            TTSWorkItem item;
            item.sentence = sentence;
            item.is_final = true;
            item.sequence = tts_sequence_.fetch_add(1, std::memory_order_relaxed);

            {
                std::lock_guard<std::mutex> lock(tts_queue_mutex_);
                tts_queue_.push(std::move(item));
            }
            tts_queue_cv_.notify_one();
        });
    }

    if (cb_event_) {
        cb_event_("response_done", "{\"length\":" + std::to_string(full_text.size()) + "}");
    }
}

void AIEngine::on_openai_speech_started() {
    AI_LOG_INFO("(%s) Speech started (VAD)\n", cfg_.session_uuid.c_str());

    bool has_active_audio = false;
    if (is_speaking()) {
        has_active_audio = true;
    } else {
        std::lock_guard<std::mutex> lock(tts_queue_mutex_);
        if (!tts_queue_.empty()) has_active_audio = true;
        if (ring_buffer_ && ring_buffer_->available_samples() > 0) has_active_audio = true;
        if (state() == AIEngineState::PROCESSING) has_active_audio = true;
    }

    if (cfg_.enable_barge_in && has_active_audio) {
        handle_barge_in();
    } else {
        set_state(AIEngineState::LISTENING, "User speaking");
    }

    if (cb_event_) {
        cb_event_("speech_started", "{}");
    }
}

void AIEngine::on_openai_speech_stopped() {
    AI_LOG_INFO("(%s) Speech stopped (VAD)\n", cfg_.session_uuid.c_str());

    if (cb_event_) {
        cb_event_("speech_stopped", "{}");
    }
}

void AIEngine::on_openai_input_transcript(const std::string& transcript) {
    AI_LOG_INFO("(%s) User said: '%s'\n", cfg_.session_uuid.c_str(), transcript.c_str());

    if (cb_event_) {
        cb_event_("user_transcript", "{\"transcript\":\"" + json_escape(transcript) + "\"}");
    }
}

void AIEngine::on_openai_error(const std::string& error, const std::string& code) {
    AI_LOG_ERROR("(%s) OpenAI error: %s (code=%s)\n",
                 cfg_.session_uuid.c_str(), error.c_str(), code.c_str());
    set_state(AIEngineState::ERROR, error);

    if (cb_event_) {
        cb_event_("openai_error", "{\"error\":\"" + json_escape(error) +
                  "\",\"code\":\"" + json_escape(code) + "\"}");
    }
}

void AIEngine::on_openai_connection_change(bool connected) {
    if (!connected) {
        AI_LOG_ERROR("(%s) OpenAI disconnected\n", cfg_.session_uuid.c_str());
        if (running_.load(std::memory_order_relaxed)) {
            set_state(AIEngineState::ERROR, "OpenAI connection lost");

            int retries = reconnect_attempts_.fetch_add(1, std::memory_order_relaxed);
            if (retries < kMaxReconnectAttempts) {
                int delay_ms = 500 * (1 << std::min(retries, 4));
                AI_LOG_INFO("(%s) Reconnecting to OpenAI in %dms (attempt %d/%d)\n",
                            cfg_.session_uuid.c_str(), delay_ms,
                            retries + 1, kMaxReconnectAttempts);

                /* Use a joinable thread, replacing any previous reconnect thread */
                {
                    std::lock_guard<std::mutex> rlock(reconnect_mutex_);
                    if (reconnect_thread_.joinable()) {
                        reconnect_thread_.join();
                    }
                    reconnect_thread_ = std::thread([this, delay_ms]() {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(delay_ms));
                        if (running_.load(std::memory_order_relaxed)) {
                            AI_LOG_INFO("(%s) Attempting OpenAI reconnect...\n",
                                        cfg_.session_uuid.c_str());
                            set_state(AIEngineState::CONNECTING, "Reconnecting");

                            /*
                             * Create a fresh WebSocket client for the reconnect.
                             * Many WS libraries don't support reconnect on same
                             * object after close/error. This guarantees clean state.
                             */
                            {
                                std::lock_guard<std::mutex> olock(openai_mutex_);
                                if (openai_) {
                                    openai_->disconnect();
                                }

                                auto fresh = std::make_unique<OpenAIRealtimeClient>();
                                fresh->on_session_created([this](const std::string& sid) {
                                    on_openai_connected(sid);
                                });
                                fresh->on_response_text_delta([this](const std::string& delta, const std::string& rid) {
                                    on_openai_text_delta(delta, rid);
                                });
                                fresh->on_response_done([this](const std::string& text, const std::string& rid) {
                                    on_openai_response_done(text, rid);
                                });
                                fresh->on_speech_started([this]() {
                                    on_openai_speech_started();
                                });
                                fresh->on_speech_stopped([this]() {
                                    on_openai_speech_stopped();
                                });
                                fresh->on_input_transcript_done([this](const std::string& transcript) {
                                    on_openai_input_transcript(transcript);
                                });
                                fresh->on_error([this](const std::string& error, const std::string& code) {
                                    on_openai_error(error, code);
                                });
                                fresh->on_connection_change([this](bool connected) {
                                    on_openai_connection_change(connected);
                                });

                                openai_ = std::move(fresh);
                                openai_->connect(cfg_.openai);
                            }
                        }
                    });
                }
            } else {
                AI_LOG_ERROR("(%s) Max reconnect attempts (%d) reached\n",
                             cfg_.session_uuid.c_str(), kMaxReconnectAttempts);
            }
        }
    } else {
        reconnect_attempts_.store(0, std::memory_order_relaxed);
    }
}

void AIEngine::tts_worker_loop() {
    AI_LOG_INFO("(%s) TTS worker thread running\n", cfg_.session_uuid.c_str());

    while (running_.load(std::memory_order_relaxed)) {
        TTSWorkItem item;

        {
            std::unique_lock<std::mutex> lock(tts_queue_mutex_);
            tts_queue_cv_.wait(lock, [this]() {
                return !tts_queue_.empty() || !running_.load(std::memory_order_relaxed);
            });

            if (!running_.load(std::memory_order_relaxed)) break;
            if (tts_queue_.empty()) continue;

            item = std::move(tts_queue_.front());
            tts_queue_.pop();
        }

        if (tts_abort_.load(std::memory_order_relaxed)) {
            continue;
        }

        process_tts_item(item);
    }

    AI_LOG_INFO("(%s) TTS worker thread exiting\n", cfg_.session_uuid.c_str());
}

void AIEngine::process_tts_item(const TTSWorkItem& item) {
    if (item.sentence.empty()) return;

    auto start_time = std::chrono::steady_clock::now();

    set_state(AIEngineState::SPEAKING, "Synthesizing: " + item.sentence.substr(0, 50));

    if (tts_cache_) {
        auto cached = tts_cache_->get(item.sentence);
        if (cached.valid) {
            AI_LOG_INFO("(%s) TTS cache hit: '%s'\n",
                        cfg_.session_uuid.c_str(),
                        item.sentence.substr(0, 40).c_str());

            on_tts_audio(cached.samples.data(), cached.samples.size(),
                        true, item.sentence, cached.sample_rate);

            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.tts_cache_hits++;
            return;
        }
    }

    std::vector<int16_t> cache_buffer;
    /*
     * Capture a mutable SR that can be updated by the TTS callback.
     * With FailoverTTS, the primary might fail and fallback runs at a
     * different sample rate. We get the correct SR by querying the engine
     * AFTER synthesize() returns (which reflects which engine actually ran).
     * But for streaming chunks we need a reasonable initial value.
     */
    int tts_sr = tts_engine_->output_sample_rate();

    bool success = tts_engine_->synthesize(
        item.sentence,
        [this, &cache_buffer, &tts_sr](const int16_t* samples, size_t count,
                                       bool is_final, const std::string& sentence) {
            if (samples && count > 0) {
                cache_buffer.insert(cache_buffer.end(), samples, samples + count);
                on_tts_audio(samples, count, is_final, sentence, tts_sr);
            }
        },

        [this](const std::string& error, int code) {
            AI_LOG_ERROR("(%s) TTS error: %s (code=%d)\n",
                         cfg_.session_uuid.c_str(), error.c_str(), code);
        },
        tts_abort_
    );

    /*
     * After synthesize returns, get the actual SR of whatever engine ran.
     * This is used for caching — the cached entry must store the correct SR
     * so on cache hit the resampler uses the right rate.
     */
    tts_sr = tts_engine_->output_sample_rate();

    if (success && tts_cache_ && !cache_buffer.empty()) {
        tts_cache_->put(item.sentence, cache_buffer, tts_sr);
    }

    {
        auto end_time = std::chrono::steady_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end_time - start_time).count();

        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.sentences_synthesized++;
        double n = (double)stats_.sentences_synthesized;
        stats_.avg_tts_latency_ms = stats_.avg_tts_latency_ms * ((n-1)/n) + latency / n;
    }

    if (item.is_final) {
        std::lock_guard<std::mutex> lock(tts_queue_mutex_);
        if (tts_queue_.empty()) {
            set_state(AIEngineState::LISTENING, "Response complete — listening");
        }
    }
}

void AIEngine::on_tts_audio(const int16_t* samples, size_t count,
                             bool, const std::string&,
                             int tts_sr) {
    if (!samples || count == 0) return;
    if (!ring_buffer_) return;
    if (tts_abort_.load(std::memory_order_relaxed)) return;
    std::vector<int16_t> resampled;
    if (tts_sr != cfg_.freeswitch_sample_rate) {
        resample_down(samples, count, tts_sr, resampled);
    } else {
        resampled.assign(samples, samples + count);
    }
    if (resampled.empty()) return;
    dsp_.process(resampled.data(), resampled.size());

    /* Re-check abort flag after resample + DSP processing to minimize
     * stale audio written to ring buffer during barge-in. */
    if (tts_abort_.load(std::memory_order_acquire)) return;

    ring_buffer_->write_pcm16(resampled.data(), resampled.size());
    if (cb_audio_) {
        cb_audio_(resampled.data(), resampled.size(), cfg_.freeswitch_sample_rate);
    }
}

void AIEngine::resample_up(const int16_t* in, size_t in_samples,
                            std::vector<int16_t>& out) {
    if (!upsample_resampler_ || !in || in_samples == 0) return;

    uint64_t out_samples_est = (uint64_t)in_samples *
        (uint64_t)cfg_.openai_send_rate / (uint64_t)cfg_.freeswitch_sample_rate + 16;
    out.resize((size_t)out_samples_est);

    spx_uint32_t in_len = (spx_uint32_t)in_samples;
    spx_uint32_t out_len = (spx_uint32_t)out.size();

    std::lock_guard<std::mutex> lock(resampler_mutex_);
    int err = speex_resampler_process_int(
        upsample_resampler_, 0,
        in, &in_len,
        out.data(), &out_len
    );

    if (err == RESAMPLER_ERR_SUCCESS) {
        out.resize(out_len);
    } else {
        out.clear();
    }
}

void AIEngine::resample_down(const int16_t* in, size_t in_samples,
                              int in_rate, std::vector<int16_t>& out) {
    if (!in || in_samples == 0 || in_rate <= 0) return;

    std::lock_guard<std::mutex> lock(resampler_mutex_);

    if (!downsample_resampler_) {
        int err = 0;
        downsample_resampler_ = speex_resampler_init(
            1, in_rate, cfg_.freeswitch_sample_rate,
            RESAMPLE_QUALITY, &err
        );
        if (err != 0 || !downsample_resampler_) {
            AI_LOG_ERROR("(%s) Failed to create downsample resampler\n",
                         cfg_.session_uuid.c_str());
            out.clear();
            return;
        }
    } else {
        spx_uint32_t cur_in = 0, cur_out = 0;
        speex_resampler_get_rate(downsample_resampler_, &cur_in, &cur_out);
        if ((int)cur_in != in_rate || (int)cur_out != cfg_.freeswitch_sample_rate) {
            speex_resampler_destroy(downsample_resampler_);
            int err = 0;
            downsample_resampler_ = speex_resampler_init(
                1, in_rate, cfg_.freeswitch_sample_rate,
                RESAMPLE_QUALITY, &err
            );
            if (err != 0 || !downsample_resampler_) {
                downsample_resampler_ = nullptr;
                out.clear();
                return;
            }
        }
    }

    uint64_t out_est = (uint64_t)in_samples *
        (uint64_t)cfg_.freeswitch_sample_rate / (uint64_t)in_rate + 16;
    out.resize((size_t)out_est);

    spx_uint32_t in_len = (spx_uint32_t)in_samples;
    spx_uint32_t out_len = (spx_uint32_t)out.size();

    int err = speex_resampler_process_int(
        downsample_resampler_, 0,
        in, &in_len,
        out.data(), &out_len
    );

    if (err == RESAMPLER_ERR_SUCCESS) {
        out.resize(out_len);
    } else {
        out.clear();
    }
}

void AIEngine::set_state(AIEngineState new_state, const std::string& detail) {
    AIEngineState old_state = state_.exchange(new_state, std::memory_order_acq_rel);
    if (old_state != new_state && cb_state_) {
        cb_state_(new_state, detail);
    }
}

void AIEngine::flush_tts_queue() {
    std::lock_guard<std::mutex> lock(tts_queue_mutex_);
    while (!tts_queue_.empty()) {
        tts_queue_.pop();
    }
}

AIEngine::Stats AIEngine::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

}