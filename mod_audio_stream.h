#ifndef MOD_AUDIO_STREAM_H
#define MOD_AUDIO_STREAM_H

#include <switch.h>
#include <speex/speex_resampler.h>

#define MY_BUG_NAME "audio_stream"
#define MY_STREAM_CONTEXT "audio_stream_context"
#define MAX_SESSION_ID (256)
#define MAX_WS_URI (4096)
#define MAX_METADATA_LEN (8192)

#define EVENT_CONNECT           "mod_audio_stream::connect"
#define EVENT_DISCONNECT        "mod_audio_stream::disconnect"
#define EVENT_ERROR             "mod_audio_stream::error"
#define EVENT_JSON              "mod_audio_stream::json"
#define EVENT_PLAY              "mod_audio_stream::play"

typedef void (*responseHandler_t)(switch_core_session_t* session, const char* eventName, const char* json);

struct private_data {
    switch_mutex_t *mutex;
    char sessionId[MAX_SESSION_ID];
    SpeexResamplerState *resampler;
    responseHandler_t responseHandler;
    void *pAudioStreamer;
    char ws_uri[MAX_WS_URI];
    int sampling;
    int channels;
    int audio_paused:1;
    int close_requested:1;
    int cleanup_started:1;
    char initialMetadata[8192];
    switch_buffer_t *sbuffer;
    int rtp_packets;
};

typedef struct private_data private_t;

typedef enum {
    STREAM_STATE_IDLE = 0,
    STREAM_STATE_STARTING,
    STREAM_STATE_ACTIVE,
    STREAM_STATE_PAUSED,
    STREAM_STATE_STOPPING
} stream_state_t;

typedef struct stream_context {
    switch_mutex_t *mutex;
    stream_state_t state;
    switch_media_bug_t *bug;
} stream_context_t;

enum notifyEvent_t {
    CONNECT_SUCCESS,
    CONNECT_ERROR,
    CONNECTION_DROPPED,
    MESSAGE
};

#endif //MOD_AUDIO_STREAM_H
