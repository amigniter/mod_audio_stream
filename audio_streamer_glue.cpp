#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include "mod_audio_stream.h"
//#include <ixwebsocket/IXWebSocket.h>
#include "WebSocketClient.h"
#include <switch_json.h>
#include <fstream>
#include <switch_buffer.h>
#include <unordered_map>
#include <unordered_set>
#include "base64.h"

#define FRAME_SIZE_8000  320 /* 1000x0.02 (20ms)= 160 x(16bit= 2 bytes) 320 frame size*/

class AudioStreamer {
public:

    AudioStreamer(const char* uuid, const char* wsUri, responseHandler_t callback, int deflate, int heart_beat,
                    bool suppressLog, const char* extra_headers, bool no_reconnect,
                    const char* tls_cafile, const char* tls_keyfile, const char* tls_certfile,
                    bool tls_disable_hostname_validation): m_sessionId(uuid), m_notify(callback),
                    m_suppress_log(suppressLog), m_extra_headers(extra_headers), m_playFile(0){

        WebSocketHeaders hdrs;
        WebSocketTLSOptions tls;

        if (m_extra_headers) {
            cJSON *headers_json = cJSON_Parse(m_extra_headers);
            if (headers_json) {
                cJSON *iterator = headers_json->child;
                while (iterator) {
                    if (iterator->type == cJSON_String && iterator->valuestring != nullptr) {
                        hdrs.set(iterator->string, iterator->valuestring);
                    }
                    iterator = iterator->next;
                }
                cJSON_Delete(headers_json);
            }
        }

        client.setUrl(wsUri);

        // Setup eventual TLS options.
        // tls_cafile may hold the special values
        // NONE, which disables validation and SYSTEM which uses
        // the system CAs bundle
        if (tls_cafile) {
            tls.caFile = tls_cafile;
        }

        if (tls_keyfile) {
            tls.keyFile = tls_keyfile;
        }

        if (tls_certfile) {
            tls.certFile = tls_certfile;
        }

        tls.disableHostnameValidation = tls_disable_hostname_validation;
        client.setTLSOptions(tls);

        // Optional heart beat, sent every xx seconds when there is not any traffic
        // to make sure that load balancers do not kill an idle connection.
        if(heart_beat)
            client.setPingInterval(heart_beat);

        // Per message deflate connection is enabled by default. You can tweak its parameters or disable it
        if(deflate)
            client.enableCompression(false);

        // Set extra headers if any
        if(!hdrs.empty())
            client.setHeaders(hdrs);

        // Setup a callback to be fired when a message or an event (open, close, error) is received
        client.setMessageCallback([this](const std::string& message) {
            eventCallback(MESSAGE, message.c_str());
        });

        // Binary callback: the server pushes raw L16 PCM frames straight into the
        // playback ring buffer (true streaming egress — no streamAudio JSON, no temp
        // file, no ::play event). The WRITE_REPLACE media-bug callback drains this
        // buffer 20ms at a time and fills the channel's write frame (see
        // stream_playback_frame). PCM is expected at the bug's `sampling` rate
        // (desiredSampling, e.g. 16000), 16-bit mono.
        client.setBinaryCallback([this](const void* data, size_t len) {
            writePlayback(static_cast<const uint8_t*>(data), len);
        });

        client.setOpenCallback([this]() {
            cJSON *root;
            root = cJSON_CreateObject();
            cJSON_AddStringToObject(root, "status", "connected");
            char *json_str = cJSON_PrintUnformatted(root);
            eventCallback(CONNECT_SUCCESS, json_str);
            cJSON_Delete(root);
            switch_safe_free(json_str);
        });

        client.setErrorCallback([this](int code, const std::string &msg) {
            cJSON *root, *message;
            root = cJSON_CreateObject();
            cJSON_AddStringToObject(root, "status", "error");
            message = cJSON_CreateObject();
            cJSON_AddNumberToObject(message, "code", code);
            cJSON_AddStringToObject(message, "error", msg.c_str());
            cJSON_AddItemToObject(root, "message", message);

            char *json_str = cJSON_PrintUnformatted(root);

            eventCallback(CONNECT_ERROR, json_str);

            cJSON_Delete(root);
            switch_safe_free(json_str);
        });

        client.setCloseCallback([this](int code, const std::string &reason) {
            cJSON *root, *message;
            root = cJSON_CreateObject();
            cJSON_AddStringToObject(root, "status", "disconnected");
            message = cJSON_CreateObject();
            cJSON_AddNumberToObject(message, "code", code);
            cJSON_AddStringToObject(message, "reason", reason.c_str());
            cJSON_AddItemToObject(root, "message", message);
            char *json_str = cJSON_PrintUnformatted(root);

            eventCallback(CONNECTION_DROPPED, json_str);

            cJSON_Delete(root);
            switch_safe_free(json_str);
        });

        // Now that our callback is setup, we can start our background thread and receive messages
        init_playback_buffer();
        client.connect();
    }

    // (Ring buffer storage is sized in a helper so the constructor body can
    // populate it without brace-init narrowing.)
    void init_playback_buffer() {
        m_playback.assign(PLAYBACK_BUF_BYTES, 0);
        m_playback_head = 0;
        m_playback_datalen = 0;
    }

    switch_media_bug_t *get_media_bug(switch_core_session_t *session) {
        switch_channel_t *channel = switch_core_session_get_channel(session);
        if(!channel) {
            return nullptr;
        }
        auto *bug = (switch_media_bug_t *) switch_channel_get_private(channel, MY_BUG_NAME);
        return bug;
    }

    inline void media_bug_close(switch_core_session_t *session) {
        auto *bug = get_media_bug(session);
        if(bug) {
            auto* tech_pvt = (private_t*) switch_core_media_bug_get_user_data(bug);
            tech_pvt->close_requested = 1;
            switch_core_media_bug_close(&bug, SWITCH_FALSE);
        }
    }

    inline void send_initial_metadata(switch_core_session_t *session) {
        auto *bug = get_media_bug(session);
        if(bug) {
            auto* tech_pvt = (private_t*) switch_core_media_bug_get_user_data(bug);
            if(tech_pvt && strlen(tech_pvt->initialMetadata) > 0) {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG,
                                          "sending initial metadata %s\n", tech_pvt->initialMetadata);
                writeText(tech_pvt->initialMetadata);
            }
        }
    }

    void eventCallback(notifyEvent_t event, const char* message) {
        switch_core_session_t* psession = switch_core_session_locate(m_sessionId.c_str());
        if(psession) {
            switch (event) {
                case CONNECT_SUCCESS:
                    send_initial_metadata(psession);
                    m_notify(psession, EVENT_CONNECT, message);
                    break;
                case CONNECTION_DROPPED:
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO, "connection closed\n");
                    m_notify(psession, EVENT_DISCONNECT, message);
                    break;
                case CONNECT_ERROR:
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO, "connection error\n");
                    m_notify(psession, EVENT_ERROR, message);

                    media_bug_close(psession);

                    break;
                case MESSAGE:
                    std::string msg(message);
                    if(processMessage(psession, msg) != SWITCH_TRUE) {
                        m_notify(psession, EVENT_JSON, msg.c_str());
                    }
                    if(!m_suppress_log)
                        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG, "response: %s\n", msg.c_str());
                    break;
            }
            switch_core_session_rwunlock(psession);
        }
    }

    switch_bool_t processMessage(switch_core_session_t* session, std::string& message) {
        cJSON* json = cJSON_Parse(message.c_str());
        switch_bool_t status = SWITCH_FALSE;
        if (!json) {
            return status;
        }
        const char* jsType = cJSON_GetObjectCstr(json, "type");
        if(jsType && strcmp(jsType, "streamAudio") == 0) {
            cJSON* jsonData = cJSON_GetObjectItem(json, "data");
            if(jsonData) {
                cJSON* jsonFile = nullptr;
                cJSON* jsonAudio = cJSON_DetachItemFromObject(jsonData, "audioData");
                const char* jsAudioDataType = cJSON_GetObjectCstr(jsonData, "audioDataType");
                std::string fileType;
                int sampleRate;
                if (0 == strcmp(jsAudioDataType, "raw")) {
                    cJSON* jsonSampleRate = cJSON_GetObjectItem(jsonData, "sampleRate");
                    sampleRate = jsonSampleRate && jsonSampleRate->valueint ? jsonSampleRate->valueint : 0;
                    std::unordered_map<int, const char*> sampleRateMap = {
                            {8000, ".r8"},
                            {16000, ".r16"},
                            {24000, ".r24"},
                            {32000, ".r32"},
                            {48000, ".r48"},
                            {64000, ".r64"}
                    };
                    auto it = sampleRateMap.find(sampleRate);
                    fileType = (it != sampleRateMap.end()) ? it->second : "";
                } else if (0 == strcmp(jsAudioDataType, "wav")) {
                    fileType = ".wav";
                } else if (0 == strcmp(jsAudioDataType, "mp3")) {
                    fileType = ".mp3";
                } else if (0 == strcmp(jsAudioDataType, "ogg")) {
                    fileType = ".ogg";
                } else {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "(%s) processMessage - unsupported audio type: %s\n",
                                      m_sessionId.c_str(), jsAudioDataType);
                }

                if(jsonAudio && jsonAudio->valuestring != nullptr && !fileType.empty()) {
                    char filePath[256];
                    std::string rawAudio;
                    try {
                        rawAudio = base64_decode(jsonAudio->valuestring);
                    } catch (const std::exception& e) {
                        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "(%s) processMessage - base64 decode error: %s\n",
                                          m_sessionId.c_str(), e.what());
                        cJSON_Delete(jsonAudio); cJSON_Delete(json);
                        return status;
                    }
                    switch_snprintf(filePath, 256, "%s%s%s_%d.tmp%s", SWITCH_GLOBAL_dirs.temp_dir,
                                    SWITCH_PATH_SEPARATOR, m_sessionId.c_str(), m_playFile++, fileType.c_str());
                    std::ofstream fstream(filePath, std::ofstream::binary);
                    fstream << rawAudio;
                    fstream.close();
                    m_Files.insert(filePath);
                    jsonFile = cJSON_CreateString(filePath);
                    cJSON_AddItemToObject(jsonData, "file", jsonFile);
                }

                if(jsonFile) {
                    char *jsonString = cJSON_PrintUnformatted(jsonData);
                    m_notify(session, EVENT_PLAY, jsonString);
                    message.assign(jsonString);
                    free(jsonString);
                    status = SWITCH_TRUE;
                }
                if (jsonAudio)
                    cJSON_Delete(jsonAudio);

            } else {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "(%s) processMessage - no data in streamAudio\n", m_sessionId.c_str());
            }
        }
        // streamComplete: backend signals TTS is done; next buffer-drain fires finished immediately.
        if(jsType && strcmp(jsType, "streamComplete") == 0) {
            m_stream_ended = true;
            status = SWITCH_TRUE;
        }
        cJSON_Delete(json);
        return status;
    }

    ~AudioStreamer()= default;

    void disconnect() {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, "disconnecting...\n");
        client.disconnect();
    }

    bool isConnected() {
        return client.isConnected();
    }

    void writeBinary(uint8_t* buffer, size_t len) {
        if(!this->isConnected()) return;
        client.sendBinary(buffer, len);
    }

    void writeText(const char* text) {
        if(!this->isConnected()) return;
        client.sendMessage(text, strlen(text));
    }

    void deleteFiles() {
        if(m_playFile >0) {
            for (const auto &fileName: m_Files) {
                remove(fileName.c_str());
            }
        }
    }

    // ---- playback ring buffer (binary PCM egress) ----

    // Push raw PCM bytes received over the WS binary callback into the ring
    // buffer. Called on the libwsc event thread. If the buffer is full we drop
    // the oldest data (the bot is behind realtime) so memory stays bounded and
    // fresh audio is preferred over stale.
    void writePlayback(const uint8_t* data, size_t len) {
        if (!data || !len) return;
        std::lock_guard<std::mutex> lock(m_playback_mutex);
        for (size_t i = 0; i < len; i++) {
            size_t pos = (m_playback_head + m_playback_datalen) % m_playback.size();
            m_playback[pos] = data[i];
            if (m_playback_datalen < m_playback.size()) {
                m_playback_datalen++;
            } else {
                m_playback_head = (m_playback_head + 1) % m_playback.size();
            }
        }
        m_playback_recv_frames++;
        m_playback_recv_bytes += len;
        if (m_playback_recv_frames == 1) {
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_NOTICE,
                "(%s) writePlayback: FIRST binary PCM frame bytes=%zu buffered=%zu\n",
                m_sessionId.c_str(), len, m_playback_datalen);
        } else if (m_playback_recv_frames % 250 == 0) {
            switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG,
                "(%s) writePlayback: frames=%llu bytes=%llu buffered=%zu\n",
                m_sessionId.c_str(), (unsigned long long)m_playback_recv_frames,
                (unsigned long long)m_playback_recv_bytes, m_playback_datalen);
        }
    }

    // Pull up to `len` bytes of PCM out of the ring buffer. Called on the FS
    // session thread (WRITE_REPLACE media-bug callback). Returns bytes actually
    // read (0 when the buffer is empty — caller then leaves the write frame
    // untouched so silence passes through).
    size_t readPlayback(uint8_t* dst, size_t len) {
        std::lock_guard<std::mutex> lock(m_playback_mutex);
        size_t n = std::min(len, m_playback_datalen);
        for (size_t i = 0; i < n; i++) {
            dst[i] = m_playback[(m_playback_head + i) % m_playback.size()];
        }
        m_playback_head = (m_playback_head + n) % m_playback.size();
        m_playback_datalen -= n;
        return n;
    }

    void clearPlayback() {
        std::lock_guard<std::mutex> lock(m_playback_mutex);
        m_playback_head = 0;
        m_playback_datalen = 0;
        m_playback_was_playing = false;
        m_empty_ticks = 0;
        m_stream_ended = false;
    }

    // Public so stream_playback_frame (free function) can access it.
    bool m_playback_was_playing = false;
    int m_empty_ticks = 0;           // consecutive empty-buffer ticks (debounce)
    bool m_stream_ended = false;     // backend sent streamComplete (explicit end)
    static constexpr int PLAYBACK_FINISH_GRACE_TICKS = 100; // ~2s at 20ms/tick

private:
    std::string m_sessionId;
    responseHandler_t m_notify;
    WebSocketClient client;
    bool m_suppress_log;
    const char* m_extra_headers;
    int m_playFile;
    std::unordered_set<std::string> m_Files;

    // Ring buffer for incoming binary PCM (L16 @ sampling, mono). 60s of headroom
    // at 16k/16-bit = 1.92MB; enough for long TTS responses that burst ahead of
    // real-time playback (CosyVoice returns 10-15s of audio in ~2s).
    enum { PLAYBACK_BUF_BYTES = 16000 * 2 * 60 };
    std::mutex m_playback_mutex;
    std::vector<uint8_t> m_playback;
    size_t m_playback_head = 0;      // ring read position
    size_t m_playback_datalen = 0;   // bytes currently buffered
    unsigned long long m_playback_recv_frames = 0;
    unsigned long long m_playback_recv_bytes = 0;
};


namespace {

    switch_status_t stream_data_init(private_t *tech_pvt, switch_core_session_t *session, char *wsUri,
                                     uint32_t sampling, int desiredSampling, int channels, char *metadata, responseHandler_t responseHandler,
                                     int deflate, int heart_beat, bool suppressLog, int rtp_packets, const char* extra_headers,
                                     bool no_reconnect, const char *tls_cafile, const char *tls_keyfile,
                                     const char *tls_certfile, bool tls_disable_hostname_validation)
    {
        int err; //speex

        switch_memory_pool_t *pool = switch_core_session_get_pool(session);

        memset(tech_pvt, 0, sizeof(private_t));

        strncpy(tech_pvt->sessionId, switch_core_session_get_uuid(session), MAX_SESSION_ID);
        strncpy(tech_pvt->ws_uri, wsUri, MAX_WS_URI);
        tech_pvt->sampling = desiredSampling;
        tech_pvt->responseHandler = responseHandler;
        tech_pvt->rtp_packets = rtp_packets;
        tech_pvt->channels = channels;
        tech_pvt->audio_paused = 0;

        if (metadata) strncpy(tech_pvt->initialMetadata, metadata, MAX_METADATA_LEN);

        //size_t buflen = (FRAME_SIZE_8000 * desiredSampling / 8000 * channels * 1000 / RTP_PERIOD * BUFFERED_SEC);
        const size_t buflen = (FRAME_SIZE_8000 * desiredSampling / 8000 * channels * rtp_packets);

        auto* as = new AudioStreamer(tech_pvt->sessionId, wsUri, responseHandler, deflate, heart_beat,
                                        suppressLog, extra_headers, no_reconnect,
                                        tls_cafile, tls_keyfile, tls_certfile, tls_disable_hostname_validation);

        tech_pvt->pAudioStreamer = static_cast<void *>(as);

        switch_mutex_init(&tech_pvt->mutex, SWITCH_MUTEX_NESTED, pool);
        
        if (switch_buffer_create(pool, &tech_pvt->sbuffer, buflen) != SWITCH_STATUS_SUCCESS) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                "%s: Error creating switch buffer.\n", tech_pvt->sessionId);
            return SWITCH_STATUS_FALSE;
        }

        if (desiredSampling != sampling) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) resampling from %u to %u\n", tech_pvt->sessionId, sampling, desiredSampling);
            tech_pvt->resampler = speex_resampler_init(channels, sampling, desiredSampling, SWITCH_RESAMPLE_QUALITY, &err);
            if (0 != err) {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "Error initializing resampler: %s.\n", speex_resampler_strerror(err));
                return SWITCH_STATUS_FALSE;
            }
        }
        else {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) no resampling needed for this call\n", tech_pvt->sessionId);
        }

        // ---- egress (playback) resampler: binary PCM arrives at `desiredSampling`
        // (the bug's sampling rate, e.g. 16000) and must be converted to the
        // channel's write-codec rate before filling the WRITE_REPLACE frame.
        // desiredSampling -> write_rate. Falls back to desiredSampling if the
        // write codec isn't available yet.
        {
            switch_codec_t *write_codec = switch_core_session_get_write_codec(session);
            int write_rate = (write_codec && write_codec->implementation) ?
                (int)write_codec->implementation->actual_samples_per_second : desiredSampling;
            tech_pvt->write_rate = write_rate;
            if (desiredSampling != write_rate) {
                tech_pvt->egress_resampler = speex_resampler_init(channels, desiredSampling, write_rate, SWITCH_RESAMPLE_QUALITY, &err);
                if (0 != err) {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                        "(%s) Error initializing egress resampler (%u->%d): %s.\n",
                        tech_pvt->sessionId, desiredSampling, write_rate, speex_resampler_strerror(err));
                    return SWITCH_STATUS_FALSE;
                }
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG,
                    "(%s) egress resampler %u->%d\n", tech_pvt->sessionId, desiredSampling, write_rate);
            }
            tech_pvt->egress_enabled = 1;
        }

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) stream_data_init\n", tech_pvt->sessionId);

        return SWITCH_STATUS_SUCCESS;
    }

    void destroy_tech_pvt(private_t* tech_pvt) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "%s destroy_tech_pvt\n", tech_pvt->sessionId);
        if (tech_pvt->resampler) {
            speex_resampler_destroy(tech_pvt->resampler);
            tech_pvt->resampler = nullptr;
        }
        if (tech_pvt->egress_resampler) {
            speex_resampler_destroy(tech_pvt->egress_resampler);
            tech_pvt->egress_resampler = nullptr;
        }
        if (tech_pvt->mutex) {
            switch_mutex_destroy(tech_pvt->mutex);
            tech_pvt->mutex = nullptr;
        }
        if (tech_pvt->pAudioStreamer) {
            auto* as = (AudioStreamer *) tech_pvt->pAudioStreamer;
            delete as;
            tech_pvt->pAudioStreamer = nullptr;
        }
    }

    void finish(private_t* tech_pvt) {
        std::shared_ptr<AudioStreamer> aStreamer;
        aStreamer.reset((AudioStreamer *)tech_pvt->pAudioStreamer);
        tech_pvt->pAudioStreamer = nullptr;

        std::thread t([aStreamer]{
            aStreamer->disconnect();
        });
        t.detach();
    }

}

extern "C" {
    int validate_ws_uri(const char* url, char* wsUri) {
        const char* scheme = nullptr;
        const char* hostStart = nullptr;
        const char* hostEnd = nullptr;
        const char* portStart = nullptr;

        // Check scheme
        if (strncmp(url, "ws://", 5) == 0) {
            scheme = "ws";
            hostStart = url + 5;
        } else if (strncmp(url, "wss://", 6) == 0) {
            scheme = "wss";
            hostStart = url + 6;
        } else {
            return 0;
        }

        // Find host end or port start
        hostEnd = hostStart;
        while (*hostEnd && *hostEnd != ':' && *hostEnd != '/') {
            if (!std::isalnum(*hostEnd) && *hostEnd != '-' && *hostEnd != '.') {
                return 0;
            }
            ++hostEnd;
        }

        // Check if host is empty
        if (hostStart == hostEnd) {
            return 0;
        }

        // Check for port
        if (*hostEnd == ':') {
            portStart = hostEnd + 1;
            while (*portStart && *portStart != '/') {
                if (!std::isdigit(*portStart)) {
                    return 0;
                }
                ++portStart;
            }
        }

        // Copy valid URI to wsUri
        std::strncpy(wsUri, url, MAX_WS_URI);
        return 1;
    }

    switch_status_t is_valid_utf8(const char *str) {
        switch_status_t status = SWITCH_STATUS_FALSE;
        while (*str) {
            if ((*str & 0x80) == 0x00) {
                // 1-byte character
                str++;
            } else if ((*str & 0xE0) == 0xC0) {
                // 2-byte character
                if ((str[1] & 0xC0) != 0x80) {
                    return status;
                }
                str += 2;
            } else if ((*str & 0xF0) == 0xE0) {
                // 3-byte character
                if ((str[1] & 0xC0) != 0x80 || (str[2] & 0xC0) != 0x80) {
                    return status;
                }
                str += 3;
            } else if ((*str & 0xF8) == 0xF0) {
                // 4-byte character
                if ((str[1] & 0xC0) != 0x80 || (str[2] & 0xC0) != 0x80 || (str[3] & 0xC0) != 0x80) {
                    return status;
                }
                str += 4;
            } else {
                // invalid character
                return status;
            }
        }
        return SWITCH_STATUS_SUCCESS;
    }

    switch_status_t stream_session_send_text(switch_core_session_t *session, char* text) {
        switch_channel_t *channel = switch_core_session_get_channel(session);
        auto *bug = (switch_media_bug_t*) switch_channel_get_private(channel, MY_BUG_NAME);
        if (!bug) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "stream_session_send_text failed because no bug\n");
            return SWITCH_STATUS_FALSE;
        }
        auto *tech_pvt = (private_t*) switch_core_media_bug_get_user_data(bug);

        if (!tech_pvt) return SWITCH_STATUS_FALSE;
        auto *pAudioStreamer = static_cast<AudioStreamer *>(tech_pvt->pAudioStreamer);
        if (pAudioStreamer && text) pAudioStreamer->writeText(text);

        return SWITCH_STATUS_SUCCESS;
    }

    switch_status_t stream_session_pauseresume(switch_core_session_t *session, int pause) {
        switch_channel_t *channel = switch_core_session_get_channel(session);
        auto *bug = (switch_media_bug_t*) switch_channel_get_private(channel, MY_BUG_NAME);
        if (!bug) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "stream_session_pauseresume failed because no bug\n");
            return SWITCH_STATUS_FALSE;
        }
        auto *tech_pvt = (private_t*) switch_core_media_bug_get_user_data(bug);

        if (!tech_pvt) return SWITCH_STATUS_FALSE;

        switch_core_media_bug_flush(bug);
        tech_pvt->audio_paused = pause;
        return SWITCH_STATUS_SUCCESS;
    }

    switch_status_t stream_session_clear_playback(switch_core_session_t *session) {
        switch_channel_t *channel = switch_core_session_get_channel(session);
        auto *bug = (switch_media_bug_t*) switch_channel_get_private(channel, MY_BUG_NAME);
        if (!bug) {
            return SWITCH_STATUS_FALSE;
        }
        auto *tech_pvt = (private_t*) switch_core_media_bug_get_user_data(bug);
        if (!tech_pvt) return SWITCH_STATUS_FALSE;
        auto *pAudioStreamer = static_cast<AudioStreamer *>(tech_pvt->pAudioStreamer);
        if (pAudioStreamer) pAudioStreamer->clearPlayback();
        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "stream_session_clear_playback: buffer cleared\n");
        return SWITCH_STATUS_SUCCESS;
    }

    switch_status_t stream_session_init(switch_core_session_t *session,
                                        responseHandler_t responseHandler,
                                        uint32_t samples_per_second,
                                        char *wsUri,
                                        int sampling,
                                        int channels,
                                        char* metadata,
                                        void **ppUserData)
    {
        int deflate, heart_beat;
        bool suppressLog = false;
        const char* buffer_size;
        const char* extra_headers;
        int rtp_packets = 1; //20ms burst
        bool no_reconnect = false;
        const char* tls_cafile = NULL;;
        const char* tls_keyfile = NULL;;
        const char* tls_certfile = NULL;;
        bool tls_disable_hostname_validation = false;

        switch_channel_t *channel = switch_core_session_get_channel(session);

        if (switch_channel_var_true(channel, "STREAM_MESSAGE_DEFLATE")) {
            deflate = 1;
        }

        if (switch_channel_var_true(channel, "STREAM_SUPPRESS_LOG")) {
            suppressLog = true;
        }

        if (switch_channel_var_true(channel, "STREAM_NO_RECONNECT")) {
            no_reconnect = true;
        }

        tls_cafile = switch_channel_get_variable(channel, "STREAM_TLS_CA_FILE");
        tls_keyfile = switch_channel_get_variable(channel, "STREAM_TLS_KEY_FILE");
        tls_certfile = switch_channel_get_variable(channel, "STREAM_TLS_CERT_FILE");

        if (switch_channel_var_true(channel, "STREAM_TLS_DISABLE_HOSTNAME_VALIDATION")) {
            tls_disable_hostname_validation = true;
        }

        const char* heartBeat = switch_channel_get_variable(channel, "STREAM_HEART_BEAT");
        if (heartBeat) {
            char *endptr;
            long value = strtol(heartBeat, &endptr, 10);
            if (*endptr == '\0' && value <= INT_MAX && value >= INT_MIN) {
                heart_beat = (int) value;
            }
        }

        if ((buffer_size = switch_channel_get_variable(channel, "STREAM_BUFFER_SIZE"))) {
            int bSize = atoi(buffer_size);
            if(bSize % 20 != 0) {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_WARNING, "%s: Buffer size of %s is not a multiple of 20ms. Using default 20ms.\n",
                                  switch_channel_get_name(channel), buffer_size);
            } else if(bSize >= 20){
                rtp_packets = bSize/20;
            }
        }

        extra_headers = switch_channel_get_variable(channel, "STREAM_EXTRA_HEADERS");

        // allocate per-session tech_pvt
        auto* tech_pvt = (private_t *) switch_core_session_alloc(session, sizeof(private_t));

        if (!tech_pvt) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR, "error allocating memory!\n");
            return SWITCH_STATUS_FALSE;
        }
        if (SWITCH_STATUS_SUCCESS != stream_data_init(tech_pvt, session, wsUri, samples_per_second, sampling, channels, metadata, responseHandler, deflate, heart_beat,
                                                        suppressLog, rtp_packets, extra_headers, no_reconnect, tls_cafile, tls_keyfile, tls_certfile, tls_disable_hostname_validation)) {
            destroy_tech_pvt(tech_pvt);
            return SWITCH_STATUS_FALSE;
        }

        *ppUserData = tech_pvt;

        return SWITCH_STATUS_SUCCESS;
    }

    switch_bool_t stream_frame(switch_media_bug_t *bug) {
        auto *tech_pvt = (private_t *)switch_core_media_bug_get_user_data(bug);
        if (!tech_pvt || tech_pvt->audio_paused) return SWITCH_TRUE;
        /*
        auto flush_sbuffer = [&]() {
            switch_size_t inuse = switch_buffer_inuse(tech_pvt->sbuffer);
            if (inuse > 0) {
                std::vector<uint8_t> tmp(inuse);
                switch_buffer_read(tech_pvt->sbuffer, tmp.data(), inuse);
                switch_buffer_zero(tech_pvt->sbuffer);
                pAudioStreamer->writeBinary(tmp.data(), inuse);
            }
        };
        */
        if (switch_mutex_trylock(tech_pvt->mutex) == SWITCH_STATUS_SUCCESS) {

            if (!tech_pvt->pAudioStreamer) {
                switch_mutex_unlock(tech_pvt->mutex);
                return SWITCH_TRUE;
            }

            auto *pAudioStreamer = static_cast<AudioStreamer *>(tech_pvt->pAudioStreamer);

            if (!pAudioStreamer->isConnected()) {
                switch_mutex_unlock(tech_pvt->mutex);
                return SWITCH_TRUE;
            }

            if (nullptr == tech_pvt->resampler) {
                
                uint8_t data_buf[SWITCH_RECOMMENDED_BUFFER_SIZE];
                switch_frame_t frame = {0};
                frame.data = data_buf;
                frame.buflen = SWITCH_RECOMMENDED_BUFFER_SIZE;
                size_t available = switch_buffer_freespace(tech_pvt->sbuffer);

                while (switch_core_media_bug_read(bug, &frame, SWITCH_TRUE) == SWITCH_STATUS_SUCCESS) {
                    if (frame.datalen) {
                        if (1 == tech_pvt->rtp_packets) {
                            pAudioStreamer->writeBinary((uint8_t *) frame.data, frame.datalen);
                            continue;
                        }
                        if (available >= frame.datalen) {
                            switch_buffer_write(tech_pvt->sbuffer, static_cast<uint8_t *>(frame.data), frame.datalen);
                        }
                        if (0 == switch_buffer_freespace(tech_pvt->sbuffer)) {
                            switch_size_t inuse = switch_buffer_inuse(tech_pvt->sbuffer);
                            if (inuse > 0) {
                                std::vector<uint8_t> tmp(inuse);
                                switch_buffer_read(tech_pvt->sbuffer, tmp.data(), inuse);
                                switch_buffer_zero(tech_pvt->sbuffer);
                                pAudioStreamer->writeBinary(tmp.data(), inuse);
                            }
                        }
                    }
                }
                
            } else {

                uint8_t data[SWITCH_RECOMMENDED_BUFFER_SIZE];
                switch_frame_t frame = {};
                frame.data = data;
                frame.buflen = SWITCH_RECOMMENDED_BUFFER_SIZE;
                const size_t available = switch_buffer_freespace(tech_pvt->sbuffer);

                while (switch_core_media_bug_read(bug, &frame, SWITCH_TRUE) == SWITCH_STATUS_SUCCESS) {
                    if(frame.datalen) {
                        spx_uint32_t in_len = frame.samples;
                        spx_uint32_t out_len = (available / (tech_pvt->channels * sizeof(spx_int16_t)));
                        spx_int16_t out[available / sizeof(spx_int16_t)];

                        if(tech_pvt->channels == 1) {
                            speex_resampler_process_int(tech_pvt->resampler,
                                            0,
                                            (const spx_int16_t *)frame.data,
                                            &in_len,
                                            &out[0],
                                            &out_len);
                        } else {
                            speex_resampler_process_interleaved_int(tech_pvt->resampler,
                                            (const spx_int16_t *)frame.data,
                                            &in_len,
                                            &out[0],
                                            &out_len);
                        }

                        if(out_len > 0) {
                            const size_t bytes_written = out_len * tech_pvt->channels * sizeof(spx_int16_t);
                            if (tech_pvt->rtp_packets == 1) { //20ms packet
                                pAudioStreamer->writeBinary((uint8_t *) out, bytes_written);
                                continue;
                            }
                            if (bytes_written <= available) {
                                switch_buffer_write(tech_pvt->sbuffer, (const uint8_t *)out, bytes_written);
                            }
                        }

                        if(switch_buffer_freespace(tech_pvt->sbuffer) == 0) {
                            switch_size_t inuse = switch_buffer_inuse(tech_pvt->sbuffer);
                            if (inuse > 0) {
                                std::vector<uint8_t> tmp(inuse);
                                switch_buffer_read(tech_pvt->sbuffer, tmp.data(), inuse);
                                switch_buffer_zero(tech_pvt->sbuffer);
                                pAudioStreamer->writeBinary(tmp.data(), inuse);
                            }
                        }
                    }
                }
            }
            
            switch_mutex_unlock(tech_pvt->mutex);
        }

        return SWITCH_TRUE;
    }

    // ---- true streaming playback (binary PCM egress) ----
    //
    // Called from the media-bug WRITE_REPLACE callback (capture_function in
    // mod_audio_stream.c). It drains the AudioStreamer playback ring buffer
    // (fed by the WS binary callback) one 20ms frame at a time and fills the
    // channel's write-replace frame with the PCM. No streamAudio JSON, no temp
    // file, no ::play event, no switch_core_media_bug_write — the raw L16 PCM
    // pushed in over the WebSocket goes straight to FreeSWITCH's audio output.
    //
    // If the buffer is empty (nothing to play) the frame is left untouched so the
    // channel's normal (silent, for a parked channel) audio passes through.
    switch_bool_t stream_playback_frame(switch_media_bug_t *bug) {
        auto *tech_pvt = (private_t *) switch_core_media_bug_get_user_data(bug);
        if (!tech_pvt || !tech_pvt->egress_enabled || tech_pvt->audio_paused || tech_pvt->close_requested) {
            return SWITCH_TRUE;
        }
        auto *pAudioStreamer = static_cast<AudioStreamer *>(tech_pvt->pAudioStreamer);
        if (!pAudioStreamer) return SWITCH_TRUE;

        // (diagnostic) confirm WRITE_REPLACE fires and show buffer fill.
        tech_pvt->playback_ticks++;
        if (tech_pvt->playback_ticks == 1) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(switch_core_media_bug_get_session(bug)),
                SWITCH_LOG_NOTICE, "(%s) stream_playback_frame: FIRST WRITE_REPLACE tick\n", tech_pvt->sessionId);
        } else if (tech_pvt->playback_ticks == 1000 || tech_pvt->playback_ticks == 2000) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(switch_core_media_bug_get_session(bug)),
                SWITCH_LOG_DEBUG, "(%s) stream_playback_frame: tick=%llu\n", tech_pvt->sessionId, (unsigned long long)tech_pvt->playback_ticks);
        }

        switch_frame_t *rframe = switch_core_media_bug_get_write_replace_frame(bug);
        if (!rframe || !rframe->data || !rframe->buflen) return SWITCH_TRUE;

        int channels = tech_pvt->channels ? tech_pvt->channels : 1;
        int in_rate = tech_pvt->sampling;   // PCM rate in the ring buffer (desiredSampling, e.g. 16000)
        int out_rate = tech_pvt->write_rate ? tech_pvt->write_rate : (int) rframe->rate;

        // One 20ms frame's worth of samples at each rate.
        size_t in_samples = (size_t)(in_rate / 50);
        size_t out_samples = (size_t)(out_rate / 50);
        size_t in_bytes = in_samples * channels * sizeof(int16_t);
        size_t out_bytes = out_samples * channels * sizeof(int16_t);
        if (out_bytes > rframe->buflen) out_bytes = rframe->buflen;

        uint8_t inbuf[SWITCH_RECOMMENDED_BUFFER_SIZE];
        if (in_bytes > sizeof(inbuf)) in_bytes = sizeof(inbuf);

        size_t got = pAudioStreamer->readPlayback(inbuf, in_bytes);
        if (got == 0) {
            // Buffer empty. Don't immediately fire "playback finished" — TTS streams in
            // bursts with gaps; the buffer legitimately drains between bursts. Use a debounce:
            // only fire after the buffer has been continuously empty for a grace period,
            // OR if the backend has sent an explicit streamComplete signal.
            if (pAudioStreamer->m_playback_was_playing) {
                pAudioStreamer->m_empty_ticks++;
                bool stream_ended = pAudioStreamer->m_stream_ended;
                int grace = pAudioStreamer->PLAYBACK_FINISH_GRACE_TICKS;
                if (stream_ended || pAudioStreamer->m_empty_ticks >= grace) {
                    pAudioStreamer->m_playback_was_playing = false;
                    pAudioStreamer->m_empty_ticks = 0;
                    pAudioStreamer->m_stream_ended = false;
                    const char *fin_msg = "{\"type\":\"playback_finished\"}";
                    pAudioStreamer->writeText(fin_msg);
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(switch_core_media_bug_get_session(bug)),
                        SWITCH_LOG_NOTICE, "(%s) stream_playback_frame: playback finished (%s)\n",
                        tech_pvt->sessionId, stream_ended ? "streamComplete" : "buffer drained");
                }
            }
            return SWITCH_TRUE;
        }
        // We have audio — mark as playing and reset the empty-tick debounce counter.
        pAudioStreamer->m_playback_was_playing = true;
        pAudioStreamer->m_empty_ticks = 0;
        // Pad a short read with silence so the resampler stays clocked at 20ms.
        if (got < in_bytes) {
            memset(inbuf + got, 0, in_bytes - got);
        }

        if (tech_pvt->egress_resampler) {
            spx_uint32_t in_len = (spx_uint32_t)(in_bytes / (channels * sizeof(int16_t)));
            spx_uint32_t out_len = (spx_uint32_t)(out_bytes / (channels * sizeof(int16_t)));
            if (channels == 1) {
                speex_resampler_process_int(tech_pvt->egress_resampler, 0,
                    (const spx_int16_t *) inbuf, &in_len,
                    (spx_int16_t *) rframe->data, &out_len);
            } else {
                speex_resampler_process_interleaved_int(tech_pvt->egress_resampler,
                    (const spx_int16_t *) inbuf, &in_len,
                    (spx_int16_t *) rframe->data, &out_len);
            }
            rframe->datalen = (uint32_t)(out_len * channels * sizeof(int16_t));
            rframe->samples = (uint32_t)out_len;
        } else {
            // No resampling needed (rates equal) — copy straight through.
            size_t n = std::min(in_bytes, out_bytes);
            memcpy(rframe->data, inbuf, n);
            rframe->datalen = (uint32_t)n;
            rframe->samples = (uint32_t)(n / (channels * sizeof(int16_t)));
        }
        rframe->rate = (uint32_t)out_rate;
        rframe->channels = (uint32_t)channels;
        return SWITCH_TRUE;
    }

    switch_status_t stream_session_cleanup(switch_core_session_t *session, char* text, int channelIsClosing) {
        switch_channel_t *channel = switch_core_session_get_channel(session);
        auto *bug = (switch_media_bug_t*) switch_channel_get_private(channel, MY_BUG_NAME);
        if(bug)
        {
            auto* tech_pvt = (private_t*) switch_core_media_bug_get_user_data(bug);
            char sessionId[MAX_SESSION_ID];
            strcpy(sessionId, tech_pvt->sessionId);

            switch_mutex_lock(tech_pvt->mutex);
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) stream_session_cleanup\n", sessionId);

            switch_channel_set_private(channel, MY_BUG_NAME, nullptr);
            if (!channelIsClosing) {
                switch_core_media_bug_remove(session, &bug);
            }

            auto* audioStreamer = (AudioStreamer *) tech_pvt->pAudioStreamer;
            if(audioStreamer) {
                audioStreamer->deleteFiles();
                if (text) audioStreamer->writeText(text);
                finish(tech_pvt);
            }

            destroy_tech_pvt(tech_pvt);

            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "(%s) stream_session_cleanup: connection closed\n", sessionId);
            return SWITCH_STATUS_SUCCESS;
        }

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "stream_session_cleanup: no bug - websocket connection already closed\n");
        return SWITCH_STATUS_FALSE;
    }
}

