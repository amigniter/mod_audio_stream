#include <string>
#include <cstring>
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
        client.connect();
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

private:
    std::string m_sessionId;
    responseHandler_t m_notify;
    WebSocketClient client;
    bool m_suppress_log;
    const char* m_extra_headers;
    int m_playFile;
    std::unordered_set<std::string> m_Files;
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

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) stream_data_init\n", tech_pvt->sessionId);

        return SWITCH_STATUS_SUCCESS;
    }

    void destroy_tech_pvt(private_t* tech_pvt) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "%s destroy_tech_pvt\n", tech_pvt->sessionId);
        if (tech_pvt->resampler) {
            speex_resampler_destroy(tech_pvt->resampler);
            tech_pvt->resampler = nullptr;
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

