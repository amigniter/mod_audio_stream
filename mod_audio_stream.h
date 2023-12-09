#ifndef MOD_AUDIO_STREAM_H
#define MOD_AUDIO_STREAM_H

#include <switch.h>
#include <speex/speex_resampler.h>
#include "buffer/ringbuffer.h"

#define MY_BUG_NAME "audio_stream"
#define MAX_SESSION_ID (256)
#define MAX_WS_URI (4096)

//#define EVENT_CONNECT           "mod_audio_stream::connect"
#define EVENT_DISCONNECT        "mod_audio_stream::disconnect"
#define EVENT_ERROR             "mod_audio_stream::error"
#define EVENT_JSON              "mod_audio_stream::json"

typedef void (*responseHandler_t)(switch_core_session_t* session, const char* eventName, const char* json);


struct private_data {
    switch_mutex_t *mutex;
    char sessionId[MAX_SESSION_ID];
    SpeexResamplerState *resampler;
    responseHandler_t responseHandler;
    void *pSharedStreamer;
    char ws_uri[MAX_WS_URI];
    int sampling;
    int channels;
    int audio_paused: 1;
    RingBuffer *buffer;
    uint8_t *data;
    int rtp_packets;
};

typedef struct private_data private_t;

enum notifyEvent_t {
    ERROR,
    DISCONNECTED,
    MESSAGE
};

#endif //MOD_AUDIO_STREAM_H
