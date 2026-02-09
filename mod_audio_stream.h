#ifndef MOD_AUDIO_STREAM_H
#define MOD_AUDIO_STREAM_H

#include <switch.h>
#include <speex/speex_resampler.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MY_BUG_NAME        "audio_stream"
#define MAX_SESSION_ID     (256)
#define MAX_WS_URI         (4096)
#define MAX_METADATA_LEN   (8192)
#define EVENT_CONNECT      "mod_audio_stream::connect"
#define EVENT_DISCONNECT   "mod_audio_stream::disconnect"
#define EVENT_ERROR        "mod_audio_stream::error"
#define EVENT_JSON         "mod_audio_stream::json"
#define EVENT_PLAY         "mod_audio_stream::play"

typedef void (*responseHandler_t)(
    switch_core_session_t* session,
    const char* eventName,
    const char* json
);

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

     /* Use explicit boolean types to avoid surprising bitfield atomicity/packing issues.
         Guard access with tech_pvt->mutex when accessed from multiple threads. */
     switch_bool_t audio_paused;
     switch_bool_t close_requested;
     switch_bool_t cleanup_started;

    char initialMetadata[MAX_METADATA_LEN];

    switch_buffer_t *sbuffer;
    int rtp_packets;

    switch_buffer_t *inject_buffer;
    int inject_sample_rate;     
    int inject_bytes_per_sample;

     uint8_t *inject_scratch;
     switch_size_t inject_scratch_len;

     /* Read-side scratch buffer for zero-allocation hot paths in stream_frame().
         Allocated once from the session pool in stream_data_init and reused. */
     uint8_t *read_scratch;
     switch_size_t read_scratch_len;

    int frame_ms;                   
    int reconnect_max;             
    int max_queue_ms;               

     int inject_buffer_ms;            
     int inject_min_buffer_ms;        
     int inject_log_every_ms;         
     int allow_file_injection;        
     int max_audio_base64_len;       
     int debug_json;                 

    uint64_t inject_write_calls;
    uint64_t inject_bytes;
    uint64_t inject_underruns;
    switch_time_t inject_last_report;
};

typedef struct private_data private_t;

enum notifyEvent_t {
    CONNECT_SUCCESS,
    CONNECT_ERROR,
    CONNECTION_DROPPED,
    MESSAGE
};

#endif 

#ifdef __cplusplus
}
#endif
