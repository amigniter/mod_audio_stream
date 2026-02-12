#ifndef MOD_AUDIO_STREAM_H
#define MOD_AUDIO_STREAM_H

#include <switch.h>
#include <speex/speex_resampler.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MY_BUG_NAME        "audio_stream"
#define MY_BUG_NAME_AI     "audio_stream_ai"
#define MAX_SESSION_ID     (256)
#define MAX_WS_URI         (4096)
#define MAX_METADATA_LEN   (8192)
#define MAX_API_KEY_LEN    (512)
#define MAX_VOICE_ID_LEN   (256)
#define MAX_MODEL_LEN      (256)
#define MAX_PROMPT_LEN     (4096)
#define EVENT_CONNECT      "mod_audio_stream::connect"
#define EVENT_DISCONNECT   "mod_audio_stream::disconnect"
#define EVENT_ERROR        "mod_audio_stream::error"
#define EVENT_JSON         "mod_audio_stream::json"
#define EVENT_PLAY         "mod_audio_stream::play"
#define EVENT_AI_STATE     "mod_audio_stream::ai_state"
#define EVENT_AI_TRANSCRIPT "mod_audio_stream::ai_transcript"
#define EVENT_AI_RESPONSE  "mod_audio_stream::ai_response"

typedef void (*responseHandler_t)(
    switch_core_session_t* session,
    const char* eventName,
    const char* json
);

struct private_data_config {
    int frame_ms;
    int inject_buffer_ms;
    int inject_min_buffer_ms;
    int inject_log_every_ms;
    int allow_file_injection;
    int max_audio_base64_len;
    int debug_json;
    int reconnect_max;
    int max_queue_ms;
};

typedef struct private_data_config private_data_config_t;

struct ai_engine_config {
    int    ai_mode_enabled;   
    char   openai_api_key[MAX_API_KEY_LEN];
    char   openai_model[MAX_MODEL_LEN];
    char   system_prompt[MAX_PROMPT_LEN];
    float  vad_threshold;
    int    vad_prefix_padding_ms;
    int    vad_silence_duration_ms;
    float  temperature;
    int    max_response_tokens;
    char   tts_provider[64];           
    char   elevenlabs_api_key[MAX_API_KEY_LEN];
    char   elevenlabs_voice_id[MAX_VOICE_ID_LEN];
    char   elevenlabs_model_id[MAX_MODEL_LEN];
    float  elevenlabs_stability;
    float  elevenlabs_similarity_boost;
    char   openai_tts_voice[64];
    int    dsp_enabled;
    float  compressor_threshold_db;
    float  compressor_makeup_db;
    float  high_shelf_gain_db;
    float  lpf_cutoff_hz;
    int    enable_barge_in;
    int    enable_tts_cache;
    int    debug_ai;
};

typedef struct ai_engine_config ai_engine_config_t;

struct private_data {

    switch_mutex_t *mutex;

    char sessionId[MAX_SESSION_ID];
    char ws_uri[MAX_WS_URI];

    int sampling;
    int channels;

    SpeexResamplerState *resampler;
    SpeexResamplerState *inject_resampler;

    responseHandler_t responseHandler;
    void *pAudioStreamer;
    volatile switch_atomic_t audio_paused;
    volatile switch_atomic_t close_requested;
    volatile switch_atomic_t cleanup_started;

    char initialMetadata[MAX_METADATA_LEN];

    switch_buffer_t *sbuffer;
    int rtp_packets;

    switch_buffer_t *inject_buffer;
    int inject_sample_rate;
    int inject_bytes_per_sample;

    uint8_t *inject_scratch;
    switch_size_t inject_scratch_len;

    uint8_t *read_scratch;
    switch_size_t read_scratch_len;

    private_data_config_t cfg;

    uint64_t inject_write_calls;
    uint64_t inject_bytes;
    uint64_t inject_underruns;
    switch_time_t inject_last_report;

    ai_engine_config_t ai_cfg;
    void *pAIEngine;            
};

typedef struct private_data private_t;

enum notifyEvent_t {
    CONNECT_SUCCESS,
    CONNECT_ERROR,
    CONNECTION_DROPPED,
    MESSAGE
};

#ifdef __cplusplus
}
#endif

#endif 
