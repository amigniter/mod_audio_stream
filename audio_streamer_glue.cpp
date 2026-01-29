#include <string>
#include <cstring>
#include "mod_audio_stream.h"
#include "WebSocketClient.h"
#include <switch_json.h>
#include <fstream>
#include <switch_buffer.h>
#include <unordered_set>
#include <atomic>
#include <vector>
#include <memory>
#include <mutex>
#include "base64.h"
#include <algorithm>
#include <climits>

#define FRAME_SIZE_8000  320 /* 1000x0.02 (20ms)= 160 x(16bit= 2 bytes) 320 frame size*/

// AI audio injection defaults
// Keep this reasonably small to bound memory and latency. 500ms is a good start.
#define INJECT_BUFFER_MS_DEFAULT 500
// Reject obviously huge base64 payloads to avoid OOM/disk abuse.
// Note: base64 inflates by ~4/3.
#define MAX_AUDIO_BASE64_LEN (4 * 1024 * 1024) /* 4MB base64 string */

class AudioStreamer {
public:
    // Factory
    static std::shared_ptr<AudioStreamer> create(
        const char* uuid, const char* wsUri, responseHandler_t callback, int deflate, int heart_beat,
        bool suppressLog, const char* extra_headers, const char* tls_cafile, const char* tls_keyfile, 
        const char* tls_certfile, bool tls_disable_hostname_validation) {

        std::shared_ptr<AudioStreamer> sp(new AudioStreamer(
            uuid, wsUri, callback, deflate, heart_beat,
            suppressLog, extra_headers, tls_cafile, tls_keyfile, 
            tls_certfile, tls_disable_hostname_validation
        ));

        sp->bindCallbacks(std::weak_ptr<AudioStreamer>(sp));

        sp->client.connect();

        return sp;
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
        std::vector<std::string> files;

        {
            std::lock_guard<std::mutex> lk(m_stateMutex);
            if (m_Files.empty())
                return;

            files.assign(m_Files.begin(), m_Files.end());
            m_Files.clear();
            m_playFile = 0;
        }

        for (const auto& fn : files) {
            ::remove(fn.c_str());
        }
    }

    void markCleanedUp() {
        m_cleanedUp.store(true, std::memory_order_release);
        client.setMessageCallback({});
        client.setOpenCallback({});
        client.setErrorCallback({});
        client.setCloseCallback({});
    }

    bool isCleanedUp() const {
        return m_cleanedUp.load(std::memory_order_acquire);
    }

private:
    // Ctor
    AudioStreamer(
        const char* uuid, const char* wsUri, responseHandler_t callback, int deflate, int heart_beat,
        bool suppressLog, const char* extra_headers, const char* tls_cafile, const char* tls_keyfile, 
        const char* tls_certfile, bool tls_disable_hostname_validation
    ) : m_sessionId(uuid), m_notify(callback), m_suppress_log(suppressLog), 
        m_extra_headers(extra_headers), m_playFile(0) {

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

        // Setup TLS options
        // NONE - disables validation
        // SYSTEM - uses the system CAs bundle
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
    }

    struct ProcessResult {
        switch_bool_t ok = SWITCH_FALSE;
        std::string rewrittenJsonData;
        std::vector<std::string> errors;
    };

    static inline void push_err(ProcessResult& out, const std::string& sid, const std::string& s) {
        out.errors.push_back("(" + sid + ") " + s);
    }

    void bindCallbacks(std::weak_ptr<AudioStreamer> wp) {
        client.setMessageCallback([wp](const std::string& message) {
            auto self = wp.lock();
            if (!self) return;
            if (self->isCleanedUp()) return;
            self->eventCallback(MESSAGE, message.c_str());
        });

        client.setOpenCallback([wp]() {
            auto self = wp.lock();
            if (!self) return;
            if (self->isCleanedUp()) return;

            cJSON* root = cJSON_CreateObject();
            cJSON_AddStringToObject(root, "status", "connected");
            char* json_str = cJSON_PrintUnformatted(root);

            self->eventCallback(CONNECT_SUCCESS, json_str);

            cJSON_Delete(root);
            switch_safe_free(json_str);
        });

        client.setErrorCallback([wp](int code, const std::string& msg) {
            auto self = wp.lock();
            if (!self) return;
            if (self->isCleanedUp()) return;

            cJSON* root = cJSON_CreateObject();
            cJSON_AddStringToObject(root, "status", "error");
            cJSON* message = cJSON_CreateObject();
            cJSON_AddNumberToObject(message, "code", code);
            cJSON_AddStringToObject(message, "error", msg.c_str());
            cJSON_AddItemToObject(root, "message", message);

            char* json_str = cJSON_PrintUnformatted(root);

            self->eventCallback(CONNECT_ERROR, json_str);

            cJSON_Delete(root);
            switch_safe_free(json_str);
        });

        client.setCloseCallback([wp](int code, const std::string& reason) {
            auto self = wp.lock();
            if (!self) return;
            if (self->isCleanedUp()) return;

            cJSON* root = cJSON_CreateObject();
            cJSON_AddStringToObject(root, "status", "disconnected");
            cJSON* message = cJSON_CreateObject();
            cJSON_AddNumberToObject(message, "code", code);
            cJSON_AddStringToObject(message, "reason", reason.c_str());
            cJSON_AddItemToObject(root, "message", message);

            char* json_str = cJSON_PrintUnformatted(root);

            self->eventCallback(CONNECTION_DROPPED, json_str);

            cJSON_Delete(root);
            switch_safe_free(json_str);
        });
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
        std::string msg = message ? message : "";

        switch_core_session_t* psession = switch_core_session_locate(m_sessionId.c_str());
        if (!psession) {
            return;
        }

        // processing without holding a session (but we now have access to tech_pvt)
        ProcessResult pr;
        if (event == MESSAGE) {
            pr = processMessage(psession, msg);
            if (pr.ok == SWITCH_TRUE) {
                msg = pr.rewrittenJsonData; // overwrite only on success
            }
        }

        for (const auto& e : pr.errors) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_ERROR, "%s\n", e.c_str());
        }

        switch (event) {
            case CONNECT_SUCCESS:
                send_initial_metadata(psession);
                m_notify(psession, EVENT_CONNECT, msg.c_str());
                break;

            case CONNECTION_DROPPED:
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO, "connection closed\n");
                m_notify(psession, EVENT_DISCONNECT, msg.c_str());
                break;

            case CONNECT_ERROR:
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO, "connection error\n");
                m_notify(psession, EVENT_ERROR, msg.c_str());
                media_bug_close(psession);
                break;

            case MESSAGE:
                if (pr.ok == SWITCH_TRUE) {
                    m_notify(psession, EVENT_PLAY, msg.c_str());
                } else {
                    // fall back to EVENT_JSON
                    m_notify(psession, EVENT_JSON, msg.c_str());
                }

                if (!m_suppress_log) {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG,
                                    "response: %s\n", msg.c_str());
                }
                break;
        }

        switch_core_session_rwunlock(psession);
    }


    static inline size_t pcm16_bytes_per_ms(int sampleRate, int channels) {
        if (sampleRate <= 0 || channels <= 0) return 0;
        // sampleRate samples/sec * 2 bytes/sample * channels / 1000
        return (size_t)sampleRate * 2u * (size_t)channels / 1000u;
    }

    static inline void drop_oldest_from_buffer(switch_buffer_t* buf, switch_size_t bytes) {
        if (!buf || bytes == 0) return;
        // Consume and discard bytes from the head.
        std::vector<uint8_t> tmp;
        tmp.resize((size_t)bytes);
        switch_buffer_read(buf, tmp.data(), bytes);
    }

    ProcessResult processMessage(switch_core_session_t* psession, const std::string& message) {
        ProcessResult out;

        // RAII
        using jsonPtr = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;
        jsonPtr root(cJSON_Parse(message.c_str()), &cJSON_Delete);
        if (!root) return out;

        const char* jsonType = cJSON_GetObjectCstr(root.get(), "type");
        if (!jsonType || std::strcmp(jsonType, "streamAudio") != 0) {
            return out; // not ours
        }

        cJSON* jsonData = cJSON_GetObjectItem(root.get(), "data");
        if (!jsonData) {
            push_err(out, m_sessionId, "processMessage - no data in streamAudio");
            return out;
        }

        const char* jsAudioDataType = cJSON_GetObjectCstr(jsonData, "audioDataType");
        if (!jsAudioDataType) jsAudioDataType = "";

        jsonPtr jsonAudio(cJSON_DetachItemFromObject(jsonData, "audioData"), &cJSON_Delete);

        if (!jsonAudio) {
            push_err(out, m_sessionId, "processMessage - streamAudio missing 'audioData' field");
            return out;
        }

        if (!cJSON_IsString(jsonAudio.get()) || !jsonAudio->valuestring) {
            push_err(out, m_sessionId, "processMessage - 'audioData' is not a string (expected base64 string)");
            return out;
        }

        // Basic size guard
        const size_t b64len = std::strlen(jsonAudio->valuestring);
        if (b64len == 0) {
            push_err(out, m_sessionId, "processMessage - 'audioData' is empty");
            return out;
        }
        if (b64len > MAX_AUDIO_BASE64_LEN) {
            push_err(out, m_sessionId, "processMessage - 'audioData' too large");
            return out;
        }

        // sampleRate (only meaningful for raw)
        int sampleRate = 0;
        if (cJSON* jsonSampleRate = cJSON_GetObjectItem(jsonData, "sampleRate")) {
            sampleRate = jsonSampleRate->valueint;
        }

        // We support true real-time injection only for raw PCM16.
        // (Other formats can be supported later via decode pipeline, but that adds latency.)
        if (std::strcmp(jsAudioDataType, "raw") != 0) {
            push_err(out, m_sessionId, "processMessage - unsupported audio type for realtime injection: " + std::string(jsAudioDataType));
            return out;
        }

        if (sampleRate <= 0) {
            push_err(out, m_sessionId, "processMessage - missing/invalid sampleRate for raw audio");
            return out;
        }

        // base64 decode
        std::string decoded;
        try {
            decoded = base64_decode(jsonAudio->valuestring);
        } catch (const std::exception& e) {
            push_err(out, m_sessionId, "processMessage - base64 decode error: " + std::string(e.what()));
            return out;
        }

        if (decoded.empty()) {
            push_err(out, m_sessionId, "processMessage - decoded audio is empty");
            return out;
        }

        // Ensure we have whole PCM16 samples.
        const size_t bytes_per_sample_frame = (size_t)2u * (size_t)1u; // PCM16, mono by default
        // If caller is stereo, we still inject mono unless you extend the protocol.
        // Frame replacement code in mod_audio_stream.c uses the call's channel count.
        // To avoid mismatched channel layouts, we default to mono injection.
        if (decoded.size() % 2u != 0u) {
            // Drop the trailing byte rather than rejecting; avoids popping on odd chunk boundaries.
            decoded.resize(decoded.size() - 1);
        }

        // Locate tech_pvt and push into inject_buffer
        switch_media_bug_t* bug = get_media_bug(psession);
        if (!bug) {
            push_err(out, m_sessionId, "processMessage - no media bug for injection");
            return out;
        }
        auto* tech_pvt = (private_t*) switch_core_media_bug_get_user_data(bug);
        if (!tech_pvt) {
            push_err(out, m_sessionId, "processMessage - missing tech_pvt for injection");
            return out;
        }
        if (!tech_pvt->inject_buffer) {
            push_err(out, m_sessionId, "processMessage - inject_buffer not initialized");
            return out;
        }

        // Optional: enforce sample rate match to avoid expensive resampling in this fast path.
        // We accept only PCM16 at the session sampling rate.
        if (tech_pvt->inject_sample_rate > 0 && sampleRate != tech_pvt->inject_sample_rate) {
            push_err(out, m_sessionId, "processMessage - sampleRate mismatch (got " + std::to_string(sampleRate) +
                                       ", expected " + std::to_string(tech_pvt->inject_sample_rate) + ")");
            return out;
        }

        // Bound buffer growth: keep at most N ms of audio.
        const int channels = 1; // injection currently mono
        const size_t max_bytes = pcm16_bytes_per_ms(tech_pvt->inject_sample_rate, channels) * (size_t)INJECT_BUFFER_MS_DEFAULT;
        const switch_size_t inuse = tech_pvt->inject_buffer ? switch_buffer_inuse(tech_pvt->inject_buffer) : 0;

        switch_mutex_lock(tech_pvt->mutex);

        if (max_bytes > 0) {
            const switch_size_t inuse_locked = switch_buffer_inuse(tech_pvt->inject_buffer);
            const size_t incoming = decoded.size();
            if ((size_t)inuse_locked + incoming > max_bytes) {
                const size_t over = ((size_t)inuse_locked + incoming) - max_bytes;
                drop_oldest_from_buffer(tech_pvt->inject_buffer, (switch_size_t)over);
            }
        }

        switch_buffer_write(tech_pvt->inject_buffer, decoded.data(), (switch_size_t)decoded.size());
        switch_mutex_unlock(tech_pvt->mutex);

        // Return a small ack payload (optional) so that upstream can correlate.
        cJSON_AddNumberToObject(jsonData, "bytes", (double)decoded.size());

        // NOTE: We no longer write temp files here. Audio is injected directly into the call.

        // return rewritten jsonData as string
        char* jsonString = cJSON_PrintUnformatted(jsonData);
        if (!jsonString) {
            push_err(out, m_sessionId, "processMessage - cJSON_PrintUnformatted failed");
            return out;
        }

        out.rewrittenJsonData.assign(jsonString);
        std::free(jsonString);
        out.ok = SWITCH_TRUE;
        return out;
    }

private:
    std::string m_sessionId;
    responseHandler_t m_notify;
    WebSocketClient client;
    bool m_suppress_log;
    const char* m_extra_headers;
    int m_playFile;
    std::unordered_set<std::string> m_Files;
    std::atomic<bool> m_cleanedUp{false};
    std::mutex m_stateMutex;
};


namespace {

    switch_status_t stream_data_init(private_t *tech_pvt, switch_core_session_t *session, char *wsUri,
                                     uint32_t sampling, int desiredSampling, int channels, char *metadata, responseHandler_t responseHandler,
                                     int deflate, int heart_beat, bool suppressLog, int rtp_packets, const char* extra_headers,
                                     const char *tls_cafile, const char *tls_keyfile, const char *tls_certfile, 
                                     bool tls_disable_hostname_validation)
    {
        int err; //speex

        switch_memory_pool_t *pool = switch_core_session_get_pool(session);

        memset(tech_pvt, 0, sizeof(private_t));

    strncpy(tech_pvt->sessionId, switch_core_session_get_uuid(session), MAX_SESSION_ID);
    tech_pvt->sessionId[MAX_SESSION_ID - 1] = '\0';
    strncpy(tech_pvt->ws_uri, wsUri, MAX_WS_URI);
    tech_pvt->ws_uri[MAX_WS_URI - 1] = '\0';
        tech_pvt->sampling = desiredSampling;
        tech_pvt->responseHandler = responseHandler;
        tech_pvt->rtp_packets = rtp_packets;
        tech_pvt->channels = channels;
        tech_pvt->audio_paused = 0;

        if (metadata) {
            strncpy(tech_pvt->initialMetadata, metadata, MAX_METADATA_LEN);
            tech_pvt->initialMetadata[MAX_METADATA_LEN - 1] = '\0';
        }

        //size_t buflen = (FRAME_SIZE_8000 * desiredSampling / 8000 * channels * 1000 / RTP_PERIOD * BUFFERED_SEC);
        const size_t buflen = (FRAME_SIZE_8000 * desiredSampling / 8000 * channels * rtp_packets);
        
        auto sp = AudioStreamer::create(tech_pvt->sessionId, wsUri, responseHandler, deflate, heart_beat,
                                        suppressLog, extra_headers, tls_cafile, tls_keyfile,
                                        tls_certfile, tls_disable_hostname_validation);

        tech_pvt->pAudioStreamer = new std::shared_ptr<AudioStreamer>(sp);

        switch_mutex_init(&tech_pvt->mutex, SWITCH_MUTEX_NESTED, pool);
        
        if (switch_buffer_create(pool, &tech_pvt->sbuffer, buflen) != SWITCH_STATUS_SUCCESS) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                "%s: Error creating switch buffer.\n", tech_pvt->sessionId);
            return SWITCH_STATUS_FALSE;
        }

        // Inject buffer for AI->FS audio (used by mod_audio_stream.c in SWITCH_ABC_TYPE_WRITE).
        // Size it to ~INJECT_BUFFER_MS_DEFAULT of PCM16 at desiredSampling.
        tech_pvt->inject_sample_rate = desiredSampling;
        tech_pvt->inject_bytes_per_sample = 2; // PCM16
        const size_t inject_bytes_per_ms = pcm16_bytes_per_ms(desiredSampling, 1);
        const size_t inject_buflen = std::max<size_t>(inject_bytes_per_ms * (size_t)INJECT_BUFFER_MS_DEFAULT, 3200u);
        if (switch_buffer_create(pool, &tech_pvt->inject_buffer, inject_buflen) != SWITCH_STATUS_SUCCESS) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                "%s: Error creating inject buffer.\n", tech_pvt->sessionId);
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
        tech_pvt->inject_buffer = nullptr;
        tech_pvt->sbuffer = nullptr;
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
            const unsigned char ch = (unsigned char)*hostEnd;
            if (!std::isalnum(ch) && *hostEnd != '-' && *hostEnd != '.') {
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
                const unsigned char ch = (unsigned char)*portStart;
                if (!std::isdigit(ch)) {
                    return 0;
                }
                ++portStart;
            }
        }

        // Copy valid URI to wsUri
        std::strncpy(wsUri, url, MAX_WS_URI);
        wsUri[MAX_WS_URI - 1] = '\0';
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

        std::shared_ptr<AudioStreamer> streamer;

        switch_mutex_lock(tech_pvt->mutex);

        if (tech_pvt->pAudioStreamer) {
            auto sp_wrap = static_cast<std::shared_ptr<AudioStreamer>*>(tech_pvt->pAudioStreamer);
            if (sp_wrap && *sp_wrap) {
                streamer = *sp_wrap; // copy shared_ptr
            }
        }

        switch_mutex_unlock(tech_pvt->mutex);

        if (streamer) {
            streamer->writeText(text);
            return SWITCH_STATUS_SUCCESS;
        }

        return SWITCH_STATUS_FALSE;
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
    int deflate = 0;
    int heart_beat = 0;
        bool suppressLog = false;
        const char* buffer_size;
        const char* extra_headers;
        int rtp_packets = 1; //20ms burst
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
        if (SWITCH_STATUS_SUCCESS != stream_data_init(tech_pvt, session, wsUri, samples_per_second, sampling, channels, 
                                                        metadata, responseHandler, deflate, heart_beat, suppressLog, rtp_packets, 
                                                        extra_headers, tls_cafile, tls_keyfile, tls_certfile, tls_disable_hostname_validation)) {
            destroy_tech_pvt(tech_pvt);
            return SWITCH_STATUS_FALSE;
        }

        *ppUserData = tech_pvt;

        return SWITCH_STATUS_SUCCESS;
    }

    switch_bool_t stream_frame(switch_media_bug_t *bug) {
        auto *tech_pvt = (private_t *)switch_core_media_bug_get_user_data(bug);
        if (!tech_pvt) return SWITCH_TRUE;
        if (tech_pvt->audio_paused || tech_pvt->cleanup_started) return SWITCH_TRUE;
        
        std::shared_ptr<AudioStreamer> streamer;
        std::vector<std::vector<uint8_t>> pending_send;

        if (switch_mutex_trylock(tech_pvt->mutex) != SWITCH_STATUS_SUCCESS) {
            return SWITCH_TRUE;
        }

        if (!tech_pvt->pAudioStreamer) {
            switch_mutex_unlock(tech_pvt->mutex);
            return SWITCH_TRUE;
        }

        auto sp_ptr = static_cast<std::shared_ptr<AudioStreamer>*>(tech_pvt->pAudioStreamer);
        if (!sp_ptr || !(*sp_ptr)) {
            switch_mutex_unlock(tech_pvt->mutex);
            return SWITCH_TRUE;
        }

        streamer = *sp_ptr;

        auto *resampler = tech_pvt->resampler;
        const int channels = tech_pvt->channels;
        const int rtp_packets = tech_pvt->rtp_packets;

        if (nullptr == resampler) {
            
            uint8_t data_buf[SWITCH_RECOMMENDED_BUFFER_SIZE];
            switch_frame_t frame = {};
            frame.data = data_buf;
            frame.buflen = SWITCH_RECOMMENDED_BUFFER_SIZE;

            while (switch_core_media_bug_read(bug, &frame, SWITCH_TRUE) == SWITCH_STATUS_SUCCESS) {
                if (!frame.datalen) {
                    continue;
                }

                if (rtp_packets == 1) {
                    pending_send.emplace_back((uint8_t*)frame.data, (uint8_t*)frame.data + frame.datalen);
                    continue;
                }

                size_t freespace = switch_buffer_freespace(tech_pvt->sbuffer);
                
                if (freespace >= frame.datalen) {
                    switch_buffer_write(tech_pvt->sbuffer, static_cast<uint8_t *>(frame.data), frame.datalen);
                }

                if (switch_buffer_freespace(tech_pvt->sbuffer) == 0) {
                    switch_size_t inuse = switch_buffer_inuse(tech_pvt->sbuffer);
                    if (inuse > 0) {
                        std::vector<uint8_t> tmp(inuse);
                        switch_buffer_read(tech_pvt->sbuffer, tmp.data(), inuse);
                        switch_buffer_zero(tech_pvt->sbuffer);
                        pending_send.emplace_back(std::move(tmp));
                    }
                }
            }
            
        } else {

            uint8_t data[SWITCH_RECOMMENDED_BUFFER_SIZE];
            switch_frame_t frame = {};
            frame.data = data;
            frame.buflen = SWITCH_RECOMMENDED_BUFFER_SIZE;

            while (switch_core_media_bug_read(bug, &frame, SWITCH_TRUE) == SWITCH_STATUS_SUCCESS) {
                if(!frame.datalen) {
                    continue;
                }

                const size_t freespace = switch_buffer_freespace(tech_pvt->sbuffer);
                spx_uint32_t in_len = frame.samples;
                spx_uint32_t out_len = (freespace / (tech_pvt->channels * sizeof(spx_int16_t)));
                
                if(out_len == 0) {
                    if(freespace == 0) {
                        switch_size_t inuse = switch_buffer_inuse(tech_pvt->sbuffer);
                        if (inuse > 0) {
                            std::vector<uint8_t> tmp(inuse);
                            switch_buffer_read(tech_pvt->sbuffer, tmp.data(), inuse);
                            switch_buffer_zero(tech_pvt->sbuffer);
                            pending_send.emplace_back(std::move(tmp));
                        }
                    }
                    continue;
                }

                std::vector<spx_int16_t> out;
                out.resize((size_t)out_len * (size_t)channels);

                if(channels == 1) {
                    speex_resampler_process_int(resampler,
                                    0,
                                    (const spx_int16_t *)frame.data,
                                    &in_len,
                                    out.data(),
                                    &out_len);
                } else {
                    speex_resampler_process_interleaved_int(resampler,
                                    (const spx_int16_t *)frame.data,
                                    &in_len,
                                    out.data(),
                                    &out_len);
                }

                if(out_len > 0) {
                    const size_t bytes_written = (size_t)out_len * (size_t)channels * sizeof(spx_int16_t);

                    if (rtp_packets == 1) { //20ms packet
                        const uint8_t* p = (const uint8_t*)out.data();
                        pending_send.emplace_back(p, p + bytes_written);
                        continue;
                    }

                    if (bytes_written <= switch_buffer_freespace(tech_pvt->sbuffer)) {
                        switch_buffer_write(tech_pvt->sbuffer, (const uint8_t *)out.data(), bytes_written);
                    }
                }

                if (switch_buffer_freespace(tech_pvt->sbuffer) == 0) {
                    switch_size_t inuse = switch_buffer_inuse(tech_pvt->sbuffer);
                    if (inuse > 0) {
                        std::vector<uint8_t> tmp(inuse);
                        switch_buffer_read(tech_pvt->sbuffer, tmp.data(), inuse);
                        switch_buffer_zero(tech_pvt->sbuffer);
                        pending_send.emplace_back(std::move(tmp));
                    }
                }
            }
        }
        
        switch_mutex_unlock(tech_pvt->mutex);
    
        if (!streamer || !streamer->isConnected()) return SWITCH_TRUE;

        for (auto &chunk : pending_send) {
            if (!chunk.empty()) {
                streamer->writeBinary(chunk.data(), chunk.size());
            }
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

            std::shared_ptr<AudioStreamer>* sp_wrap = nullptr;
            std::shared_ptr<AudioStreamer> streamer;

            switch_mutex_lock(tech_pvt->mutex);

            if (tech_pvt->cleanup_started) {
                switch_mutex_unlock(tech_pvt->mutex);
                return SWITCH_STATUS_SUCCESS;
            }
            tech_pvt->cleanup_started = 1;

            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) stream_session_cleanup\n", sessionId);

            switch_channel_set_private(channel, MY_BUG_NAME, nullptr);

            sp_wrap = static_cast<std::shared_ptr<AudioStreamer>*>(tech_pvt->pAudioStreamer);
            tech_pvt->pAudioStreamer = nullptr;

            if (sp_wrap && *sp_wrap) {
                streamer = *sp_wrap;
            }

            switch_mutex_unlock(tech_pvt->mutex);

            if (!channelIsClosing) {
                switch_core_media_bug_remove(session, &bug);
            }

            if (sp_wrap) {
                delete sp_wrap;
                sp_wrap = nullptr;
            }

            if(streamer) {
                streamer->deleteFiles();
                if (text) streamer->writeText(text);
                
                streamer->markCleanedUp();
                streamer->disconnect();
            }

            destroy_tech_pvt(tech_pvt);

            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO, "(%s) stream_session_cleanup: connection closed\n", sessionId);
            return SWITCH_STATUS_SUCCESS;
        }

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "stream_session_cleanup: no bug - websocket connection already closed\n");
        return SWITCH_STATUS_FALSE;
    }
}