#include "mod_audio_stream.h"
#include "audio_streamer_glue.h"
#include "ai_engine/ai_engine.h"
#include <memory>
#include <cstring>
#include <cstdlib>

static inline const char* get_channel_var(switch_core_session_t* session,
                                           const char* name,
                                           const char* def) {
    switch_channel_t* channel = switch_core_session_get_channel(session);
    const char* val = channel ? switch_channel_get_variable(channel, name) : NULL;
    return (val && *val) ? val : def;
}

static inline int get_channel_var_int(switch_core_session_t* session,
                                       const char* name, int def) {
    const char* val = get_channel_var(session, name, NULL);
    if (!val) return def;
    return atoi(val);
}

static inline float get_channel_var_float(switch_core_session_t* session,
                                           const char* name, float def) {
    const char* val = get_channel_var(session, name, NULL);
    if (!val) return def;
    return (float)atof(val);
}

static inline void safe_strncpy(char* dest, const char* src, size_t max) {
    if (!src) { dest[0] = '\0'; return; }
    strncpy(dest, src, max);
    dest[max - 1] = '\0';
}

static void ai_event_handler(switch_core_session_t* session,
                               responseHandler_t responseHandler,
                               const std::string& event_name,
                               const std::string& json) {
    if (!responseHandler || !session) return;

    const char* fs_event = EVENT_AI_STATE;
    if (event_name == "user_transcript") {
        fs_event = EVENT_AI_TRANSCRIPT;
    } else if (event_name == "response_done") {
        fs_event = EVENT_AI_RESPONSE;
    }

    responseHandler(session, fs_event, json.c_str());
}

extern "C" {

switch_status_t ai_engine_session_init(switch_core_session_t *session,
                                        responseHandler_t responseHandler,
                                        uint32_t samples_per_second,
                                        int sampling, int channels,
                                        void **ppUserData) {

    switch_channel_t *channel = switch_core_session_get_channel(session);
    switch_memory_pool_t *pool = switch_core_session_get_pool(session);
    const char* uuid = switch_core_session_get_uuid(session);

    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
                      "(%s) ai_engine_session_init: initializing AI mode\n", uuid);

    auto* tech_pvt = (private_t*)*ppUserData;
    if (!tech_pvt) {
        tech_pvt = (private_t*)switch_core_session_alloc(session, sizeof(private_t));
        if (!tech_pvt) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                              "(%s) Error allocating memory for AI engine\n", uuid);
            return SWITCH_STATUS_FALSE;
        }
        memset(tech_pvt, 0, sizeof(*tech_pvt));
        switch_mutex_init(&tech_pvt->mutex, SWITCH_MUTEX_NESTED, pool);
    }

    strncpy(tech_pvt->sessionId, uuid, MAX_SESSION_ID);
    tech_pvt->sessionId[MAX_SESSION_ID - 1] = '\0';
    tech_pvt->sampling = sampling;
    if (channels > 0) tech_pvt->channels = channels;
    tech_pvt->responseHandler = responseHandler;
    ai_engine_config_t& ai_cfg = tech_pvt->ai_cfg;
    ai_cfg.ai_mode_enabled = 1;

    safe_strncpy(ai_cfg.openai_api_key,
                 get_channel_var(session, "AI_OPENAI_API_KEY", ""),
                 MAX_API_KEY_LEN);
    safe_strncpy(ai_cfg.openai_model,
                 get_channel_var(session, "AI_OPENAI_MODEL", "gpt-4o-realtime-preview"),
                 MAX_MODEL_LEN);
    safe_strncpy(ai_cfg.system_prompt,
                 get_channel_var(session, "AI_SYSTEM_PROMPT",
                     "You are a helpful AI assistant on a phone call. "
                     "Keep responses brief and conversational. "
                     "Respond in the same language the caller uses."),
                 MAX_PROMPT_LEN);

    ai_cfg.vad_threshold = get_channel_var_float(session, "AI_VAD_THRESHOLD", 0.5f);
    ai_cfg.vad_prefix_padding_ms = get_channel_var_int(session, "AI_VAD_PREFIX_PADDING_MS", 300);
    ai_cfg.vad_silence_duration_ms = get_channel_var_int(session, "AI_VAD_SILENCE_DURATION_MS", 500);
    ai_cfg.temperature = get_channel_var_float(session, "AI_TEMPERATURE", 0.8f);
    ai_cfg.max_response_tokens = get_channel_var_int(session, "AI_MAX_RESPONSE_TOKENS", 4096);

    safe_strncpy(ai_cfg.tts_provider,
                 get_channel_var(session, "AI_TTS_PROVIDER", "elevenlabs"),
                 sizeof(ai_cfg.tts_provider));
    safe_strncpy(ai_cfg.elevenlabs_api_key,
                 get_channel_var(session, "AI_ELEVENLABS_API_KEY", ""),
                 MAX_API_KEY_LEN);
    safe_strncpy(ai_cfg.elevenlabs_voice_id,
                 get_channel_var(session, "AI_ELEVENLABS_VOICE_ID", ""),
                 MAX_VOICE_ID_LEN);
    safe_strncpy(ai_cfg.elevenlabs_model_id,
                 get_channel_var(session, "AI_ELEVENLABS_MODEL_ID", "eleven_turbo_v2_5"),
                 MAX_MODEL_LEN);
    ai_cfg.elevenlabs_stability = get_channel_var_float(session, "AI_ELEVENLABS_STABILITY", 0.5f);
    ai_cfg.elevenlabs_similarity_boost = get_channel_var_float(session, "AI_ELEVENLABS_SIMILARITY_BOOST", 0.75f);

    safe_strncpy(ai_cfg.openai_tts_voice,
                 get_channel_var(session, "AI_OPENAI_TTS_VOICE", "alloy"),
                 sizeof(ai_cfg.openai_tts_voice));

    ai_cfg.dsp_enabled = get_channel_var_int(session, "AI_DSP_ENABLED", 1);
    ai_cfg.compressor_threshold_db = get_channel_var_float(session, "AI_COMPRESSOR_THRESHOLD_DB", -18.0f);
    ai_cfg.compressor_makeup_db = get_channel_var_float(session, "AI_COMPRESSOR_MAKEUP_DB", 6.0f);
    ai_cfg.high_shelf_gain_db = get_channel_var_float(session, "AI_HIGH_SHELF_GAIN_DB", 3.0f);
    ai_cfg.lpf_cutoff_hz = get_channel_var_float(session, "AI_LPF_CUTOFF_HZ", 3800.0f);
    ai_cfg.enable_barge_in = get_channel_var_int(session, "AI_ENABLE_BARGE_IN", 1);
    ai_cfg.enable_tts_cache = get_channel_var_int(session, "AI_ENABLE_TTS_CACHE", 1);
    ai_cfg.debug_ai = get_channel_var_int(session, "AI_DEBUG", 0);

    if (strlen(ai_cfg.openai_api_key) == 0) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                          "(%s) AI_OPENAI_API_KEY not set — cannot start AI engine\n", uuid);
        return SWITCH_STATUS_FALSE;
    }
    if (strcmp(ai_cfg.tts_provider, "elevenlabs") == 0 && strlen(ai_cfg.elevenlabs_api_key) == 0) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                          "(%s) AI_ELEVENLABS_API_KEY not set — cannot start ElevenLabs TTS\n", uuid);
        return SWITCH_STATUS_FALSE;
    }

    {
        int inject_ms = get_channel_var_int(session, "STREAM_INJECT_BUFFER_MS", 5000);
        size_t inject_bytes_per_ms = (size_t)sampling * 2 * channels / 1000;
        size_t inject_buflen = inject_bytes_per_ms * inject_ms;
        if (inject_buflen < 3200) inject_buflen = 3200;

        if (switch_buffer_create(pool, &tech_pvt->inject_buffer, inject_buflen) != SWITCH_STATUS_SUCCESS) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                              "(%s) Error creating inject buffer for AI mode\n", uuid);
            return SWITCH_STATUS_FALSE;
        }
        tech_pvt->inject_sample_rate = sampling;
        tech_pvt->inject_bytes_per_sample = 2;
    }

    tech_pvt->read_scratch_len = SWITCH_RECOMMENDED_BUFFER_SIZE;
    tech_pvt->read_scratch = (uint8_t*)switch_core_session_alloc(session, tech_pvt->read_scratch_len);
    tech_pvt->inject_scratch_len = SWITCH_RECOMMENDED_BUFFER_SIZE;
    tech_pvt->inject_scratch = (uint8_t*)switch_core_session_alloc(session, tech_pvt->inject_scratch_len);

    if (!tech_pvt->read_scratch || !tech_pvt->inject_scratch) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                          "(%s) Error allocating scratch buffers\n", uuid);
        return SWITCH_STATUS_FALSE;
    }

    if ((int)samples_per_second != sampling) {
        int err = 0;
        tech_pvt->resampler = speex_resampler_init(channels, samples_per_second, sampling, 7, &err);
        if (err != 0 || !tech_pvt->resampler) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                              "(%s) Error creating resampler for AI mode\n", uuid);
            return SWITCH_STATUS_FALSE;
        }
    }

    {
        ai_engine::AIEngineConfig engine_cfg;

        engine_cfg.openai.api_key = ai_cfg.openai_api_key;
        engine_cfg.openai.model = ai_cfg.openai_model;
        engine_cfg.openai.system_prompt = ai_cfg.system_prompt;
        engine_cfg.openai.vad_threshold = ai_cfg.vad_threshold;
        engine_cfg.openai.vad_prefix_padding_ms = ai_cfg.vad_prefix_padding_ms;
        engine_cfg.openai.vad_silence_duration_ms = ai_cfg.vad_silence_duration_ms;
        engine_cfg.openai.temperature = ai_cfg.temperature;
        engine_cfg.openai.max_response_output_tokens = ai_cfg.max_response_tokens;
        engine_cfg.openai.input_sample_rate = 16000;
        engine_cfg.tts.provider = ai_cfg.tts_provider;
        engine_cfg.tts.elevenlabs_api_key = ai_cfg.elevenlabs_api_key;
        engine_cfg.tts.elevenlabs_voice_id = ai_cfg.elevenlabs_voice_id;
        engine_cfg.tts.elevenlabs_model_id = ai_cfg.elevenlabs_model_id;
        engine_cfg.tts.elevenlabs_stability = ai_cfg.elevenlabs_stability;
        engine_cfg.tts.elevenlabs_similarity_boost = ai_cfg.elevenlabs_similarity_boost;
        engine_cfg.tts.elevenlabs_output_sample_rate = sampling;
        engine_cfg.tts.openai_api_key = ai_cfg.openai_api_key;
        engine_cfg.tts.openai_voice = ai_cfg.openai_tts_voice;
        engine_cfg.tts.enable_cache = ai_cfg.enable_tts_cache;
        engine_cfg.dsp.sample_rate = sampling;
        engine_cfg.dsp.compressor_enabled = ai_cfg.dsp_enabled;
        engine_cfg.dsp.high_shelf_enabled = ai_cfg.dsp_enabled;
        engine_cfg.dsp.lpf_enabled = ai_cfg.dsp_enabled;
        engine_cfg.dsp.soft_clipper_enabled = ai_cfg.dsp_enabled;
        engine_cfg.dsp.dc_blocker_enabled = ai_cfg.dsp_enabled;
        engine_cfg.dsp.noise_gate_enabled = ai_cfg.dsp_enabled;
        engine_cfg.dsp.compressor_threshold_db = ai_cfg.compressor_threshold_db;
        engine_cfg.dsp.compressor_makeup_db = ai_cfg.compressor_makeup_db;
        engine_cfg.dsp.high_shelf_gain_db = ai_cfg.high_shelf_gain_db;
        engine_cfg.dsp.lpf_cutoff_hz = ai_cfg.lpf_cutoff_hz;
        engine_cfg.freeswitch_sample_rate = sampling;
        engine_cfg.openai_send_rate = 16000;
        engine_cfg.enable_barge_in = ai_cfg.enable_barge_in;
        engine_cfg.debug_logging = ai_cfg.debug_ai;
        engine_cfg.session_uuid = uuid;
        engine_cfg.inject_buffer_ms = 5000;
        auto* engine = new ai_engine::AIEngine();

        std::string session_uuid_str(uuid);
        engine->set_event_callback(
            [session_uuid_str, responseHandler](const std::string& event_name, const std::string& json) {
                switch_core_session_t* psession = switch_core_session_locate(
                    session_uuid_str.c_str()
                );
                if (psession) {
                    ai_event_handler(psession, responseHandler, event_name, json);
                    switch_core_session_rwunlock(psession);
                }
            }
        );

        if (!engine->start(engine_cfg, session)) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                              "(%s) Failed to start AI engine\n", uuid);
            delete engine;
            return SWITCH_STATUS_FALSE;
        }

        tech_pvt->pAIEngine = engine;
    }

    *ppUserData = tech_pvt;

    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
                      "(%s) AI engine initialized: model=%s tts=%s voice=%s barge_in=%d\n",
                      uuid, ai_cfg.openai_model, ai_cfg.tts_provider,
                      ai_cfg.elevenlabs_voice_id, ai_cfg.enable_barge_in);

    return SWITCH_STATUS_SUCCESS;
}

switch_bool_t ai_engine_feed_frame(switch_media_bug_t *bug) {
    auto *tech_pvt = (private_t*)switch_core_media_bug_get_user_data(bug);
    if (!tech_pvt) return SWITCH_TRUE;
    if (tech_pvt->audio_paused || tech_pvt->cleanup_started) return SWITCH_TRUE;

    auto* engine = static_cast<ai_engine::AIEngine*>(tech_pvt->pAIEngine);
    if (!engine || !engine->is_running()) return SWITCH_TRUE;

    uint8_t data[SWITCH_RECOMMENDED_BUFFER_SIZE];
    switch_frame_t frame = {};
    frame.data = data;
    frame.buflen = SWITCH_RECOMMENDED_BUFFER_SIZE;

    while (switch_core_media_bug_read(bug, &frame, SWITCH_TRUE) == SWITCH_STATUS_SUCCESS) {
        if (!frame.datalen) continue;

        const int16_t* samples = (const int16_t*)frame.data;
        size_t num_samples = frame.datalen / sizeof(int16_t);

        engine->feed_audio(samples, num_samples);
    }

    return SWITCH_TRUE;
}

switch_size_t ai_engine_read_audio(private_t *tech_pvt, int16_t* dest, size_t num_samples) {
    if (!tech_pvt || !dest || num_samples == 0) return 0;

    auto* engine = static_cast<ai_engine::AIEngine*>(tech_pvt->pAIEngine);
    if (!engine || !engine->is_running()) return 0;

    return engine->read_audio(dest, num_samples);
}

switch_status_t ai_engine_session_cleanup(switch_core_session_t *session,
                                           int channelIsClosing) {
    switch_channel_t *channel = switch_core_session_get_channel(session);
    auto *bug = (switch_media_bug_t*)switch_channel_get_private(channel, MY_BUG_NAME);

    if (!bug) {
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG,
                          "ai_engine_session_cleanup: no bug found\n");
        return SWITCH_STATUS_FALSE;
    }

    auto* tech_pvt = (private_t*)switch_core_media_bug_get_user_data(bug);
    if (!tech_pvt) return SWITCH_STATUS_FALSE;

    switch_mutex_lock(tech_pvt->mutex);

    if (tech_pvt->cleanup_started) {
        switch_mutex_unlock(tech_pvt->mutex);
        return SWITCH_STATUS_SUCCESS;
    }
    tech_pvt->cleanup_started = SWITCH_TRUE;
    tech_pvt->close_requested = SWITCH_TRUE;

    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
                      "(%s) ai_engine_session_cleanup\n", tech_pvt->sessionId);

    switch_channel_set_private(channel, MY_BUG_NAME, nullptr);

    auto* engine = static_cast<ai_engine::AIEngine*>(tech_pvt->pAIEngine);
    tech_pvt->pAIEngine = nullptr;

    switch_mutex_unlock(tech_pvt->mutex);

    if (!channelIsClosing) {
        switch_core_media_bug_remove(session, &bug);
    }

    if (engine) {
        engine->stop();
        delete engine;
    }

    if (tech_pvt->resampler) {
        speex_resampler_destroy(tech_pvt->resampler);
        tech_pvt->resampler = nullptr;
    }

    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
                      "(%s) AI engine cleanup complete\n", tech_pvt->sessionId);

    return SWITCH_STATUS_SUCCESS;
}

}