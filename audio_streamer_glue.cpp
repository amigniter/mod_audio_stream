#include <string>
#include <cstring>
#include "mod_audio_stream.h"
#include "WebSocketClient.h"
#include <switch_json.h>
#include <fstream>
#include <sstream>
#include <switch_buffer.h>
#include <unordered_set>
#include <atomic>
#include <vector>
#include <memory>
#include <mutex>
#include "base64.h"
#include <algorithm>
#include <climits>
#include <cstdint>

#define MOD_AUDIO_STREAM_VERSION "1.1.0"

#define FRAME_SIZE_8000  320 
#define INJECT_BUFFER_MS_DEFAULT 500
#define MAX_AUDIO_BASE64_LEN (4 * 1024 * 1024) 

class AudioStreamer {
public:
   
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

        if(heart_beat)
            client.setPingInterval(heart_beat);

        if(deflate)
            client.enableCompression(false);

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

        if (event == MESSAGE) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG,
                              "(%s) eventCallback: incoming MESSAGE size=%zu\n",
                              m_sessionId.c_str(), msg.size());
            if (msg.size() > 512) {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG,
                                  "(%s) eventCallback: message snippet=%.512s...\n",
                                  m_sessionId.c_str(), msg.c_str());
            }
        }

        ProcessResult pr;
        if (event == MESSAGE) {
            pr = processMessage(psession, msg);
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO,
                              "(%s) processMessage result: ok=%d errors=%zu\n",
                              m_sessionId.c_str(), pr.ok == SWITCH_TRUE ? 1 : 0, pr.errors.size());
            if (pr.ok == SWITCH_TRUE) {
                msg = pr.rewrittenJsonData; 
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
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO,
                                      "(%s) PUSHBACK accepted -> EVENT_PLAY\n", m_sessionId.c_str());
                    m_notify(psession, EVENT_PLAY, msg.c_str());
                } else {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG,
                                      "(%s) MESSAGE treated as generic JSON\n", m_sessionId.c_str());
                    m_notify(psession, EVENT_JSON, msg.c_str());
                }

                if (!m_suppress_log) {
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG,
                                    "response (v%s): %s\n", MOD_AUDIO_STREAM_VERSION, msg.c_str());
                }
                break;
        }

        switch_core_session_rwunlock(psession);
    }


    static inline size_t pcm16_bytes_per_ms(int sampleRate, int channels) {
        if (sampleRate <= 0 || channels <= 0) return 0;
        return (size_t)sampleRate * 2u * (size_t)channels / 1000u;
    }

    static inline bool host_is_little_endian() {
        const uint16_t x = 1;
        return *((const uint8_t*)&x) == 1;
    }

    static inline void byteswap_inplace_16(std::string& s) {
        const size_t n = s.size() & ~size_t(1);
        for (size_t i = 0; i < n; i += 2) {
            std::swap(s[i], s[i + 1]);
        }
    }

    static inline std::string downmix_stereo_to_mono_pcm16le(const uint8_t* in, size_t in_bytes) {
       
        const size_t frames = (in_bytes / 4);
        std::string out;
        out.resize(frames * 2);
        for (size_t i = 0; i < frames; ++i) {
            int16_t l;
            int16_t r;
            std::memcpy(&l, in + i * 4, 2);
            std::memcpy(&r, in + i * 4 + 2, 2);
            const int32_t m = ((int32_t)l + (int32_t)r) / 2;
            const int16_t mono = (int16_t)std::max<int32_t>(-32768, std::min<int32_t>(32767, m));
            std::memcpy(&out[i * 2], &mono, 2);
        }
        return out;
    }

    static inline std::string upmix_mono_to_stereo_pcm16le(const uint8_t* in, size_t in_bytes) {
        const size_t frames = in_bytes / 2;
        std::string out;
        out.resize(frames * 4);
        for (size_t i = 0; i < frames; ++i) {
            int16_t m;
            std::memcpy(&m, in + i * 2, 2);
            std::memcpy(&out[i * 4], &m, 2);
            std::memcpy(&out[i * 4 + 2], &m, 2);
        }
        return out;
    }

    static inline std::string resample_pcm16le_speex(const uint8_t* in, size_t in_bytes, int channels,
                                                     int in_sr, int out_sr, SpeexResamplerState* resampler) {
       
        if (!resampler || in_sr == out_sr) {
            return std::string((const char*)in, (const char*)in + in_bytes);
        }
        const spx_uint32_t in_frames = (spx_uint32_t)(in_bytes / (size_t)(channels * 2));
        if (in_frames == 0) return {};

        const uint64_t nom = (uint64_t)in_frames * (uint64_t)out_sr;
        const uint32_t out_frames_cap = (uint32_t)(nom / (uint64_t)in_sr + 16);
        std::vector<spx_int16_t> out;
        out.resize((size_t)out_frames_cap * (size_t)channels);

        spx_uint32_t in_len = in_frames;
        spx_uint32_t out_len = out_frames_cap;
        int err = 0;

        if (channels == 1) {
            err = speex_resampler_process_int(resampler, 0,
                                              (const spx_int16_t*)in, &in_len,
                                              out.data(), &out_len);
        } else {
            err = speex_resampler_process_interleaved_int(resampler,
                                                          (const spx_int16_t*)in, &in_len,
                                                          out.data(), &out_len);
        }
        if (err != RESAMPLER_ERR_SUCCESS) {
            return std::string((const char*)in, (const char*)in + in_bytes);
        }
        const size_t out_bytes = (size_t)out_len * (size_t)channels * 2;
        return std::string((const char*)out.data(), (const char*)out.data() + out_bytes);
    }

    static inline void drop_oldest_from_buffer(switch_buffer_t* buf, switch_size_t bytes) {
        if (!buf || bytes == 0) return;
        std::vector<uint8_t> tmp;
        tmp.resize((size_t)bytes);
        switch_buffer_read(buf, tmp.data(), bytes);
    }

    ProcessResult processMessage(switch_core_session_t* psession, const std::string& message) {
        ProcessResult out;

        using jsonPtr = std::unique_ptr<cJSON, decltype(&cJSON_Delete)>;
        jsonPtr root(cJSON_Parse(message.c_str()), &cJSON_Delete);
        if (!root) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG,
                              "(%s) processMessage: invalid JSON or empty message\n", m_sessionId.c_str());
            return out;
        }

        const char* jsonType = cJSON_GetObjectCstr(root.get(), "type");
        cJSON* jsonData = nullptr;

        if (!jsonType || std::strcmp(jsonType, "streamAudio") != 0) {
            
            if (cJSON_GetObjectItem(root.get(), "audioData") || cJSON_GetObjectItem(root.get(), "file") || cJSON_GetObjectItem(root.get(), "audioDataType")) {
                jsonData = root.get();
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO,
                                  "(%s) processMessage: treating root as data shorthand\n", m_sessionId.c_str());
            } else {
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG,
                                  "(%s) processMessage ignored (expected type=streamAudio). raw=%s\n",
                                  m_sessionId.c_str(), message.c_str());
                return out; 
            }
        } else {
            jsonData = cJSON_GetObjectItem(root.get(), "data");
            if (!jsonData) {
                push_err(out, m_sessionId, "processMessage - no data in streamAudio");
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_ERROR,
                                  "(%s) processMessage - no data object in streamAudio\n", m_sessionId.c_str());
                return out;
            }
        }

        const char* jsAudioDataType = cJSON_GetObjectCstr(jsonData, "audioDataType");
        if (!jsAudioDataType) jsAudioDataType = "";

        jsonPtr jsonAudio(cJSON_DetachItemFromObject(jsonData, "audioData"), &cJSON_Delete);

        std::string decoded;
        bool decoded_from_file = false;

        if (jsonAudio && cJSON_IsString(jsonAudio.get()) && jsonAudio->valuestring) {
           
            const size_t b64len = std::strlen(jsonAudio->valuestring);
            if (b64len == 0) {
                push_err(out, m_sessionId, "processMessage - 'audioData' is empty");
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_ERROR,
                                  "(%s) processMessage - audioData empty\n", m_sessionId.c_str());
                return out;
            }
            if (b64len > MAX_AUDIO_BASE64_LEN) {
                push_err(out, m_sessionId, "processMessage - 'audioData' too large");
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_ERROR,
                                  "(%s) processMessage - audioData too large: %zu\n", m_sessionId.c_str(), b64len);
                return out;
            }

            try {
                decoded = base64_decode(jsonAudio->valuestring);
            } catch (const std::exception& e) {
                push_err(out, m_sessionId, "processMessage - base64 decode error: " + std::string(e.what()));
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_ERROR,
                                  "(%s) processMessage - base64 decode exception: %s\n", m_sessionId.c_str(), e.what());
                return out;
            }
        } else {
            cJSON* jsonFile = cJSON_GetObjectItem(jsonData, "file");
            if (jsonFile && cJSON_IsString(jsonFile) && jsonFile->valuestring) {
                const char* filepath = jsonFile->valuestring;
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO,
                                  "(%s) processMessage: attempting to read file payload %s\n",
                                  m_sessionId.c_str(), filepath);

                std::ifstream ifs(filepath, std::ios::binary);
                if (!ifs) {
                    push_err(out, m_sessionId, "processMessage - cannot open file: " + std::string(filepath));
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_ERROR,
                                      "(%s) processMessage - cannot open file: %s\n", m_sessionId.c_str(), filepath);
                    return out;
                }
                std::ostringstream ss;
                ss << ifs.rdbuf();
                decoded = ss.str();
                decoded_from_file = true;
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO,
                                  "(%s) processMessage: read file %s size=%zu\n",
                                  m_sessionId.c_str(), filepath, decoded.size());
            } else {
                push_err(out, m_sessionId, "processMessage - streamAudio missing 'audioData' field and no 'file' provided");
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_ERROR,
                                  "(%s) processMessage - missing audioData field and no file\n", m_sessionId.c_str());
                return out;
            }
        }

        int sampleRate = 0;
        if (cJSON* jsonSampleRate = cJSON_GetObjectItem(jsonData, "sampleRate")) {
            sampleRate = jsonSampleRate->valueint;
        }

        if (std::strcmp(jsAudioDataType, "raw") != 0) {
            push_err(out, m_sessionId, "processMessage - unsupported audio type for realtime injection: " + std::string(jsAudioDataType));
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_ERROR,
                              "(%s) processMessage - unsupported audioDataType=%s\n", m_sessionId.c_str(), jsAudioDataType);
            return out;
        }

        if (sampleRate <= 0) {
            push_err(out, m_sessionId, "processMessage - missing/invalid sampleRate for raw audio");
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_ERROR,
                              "(%s) processMessage - missing/invalid sampleRate\n", m_sessionId.c_str());
            return out;
        }

        if (decoded.empty()) {
            push_err(out, m_sessionId, "processMessage - decoded audio is empty");
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_ERROR,
                              "(%s) processMessage - decoded audio empty after base64/file read\n", m_sessionId.c_str());
            return out;
        }

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG,
                          "(%s) processMessage: decoded_bytes=%zu sampleRate=%d\n",
                          m_sessionId.c_str(), decoded.size(), sampleRate);

        if (decoded.size() % 2u != 0u) {
            decoded.resize(decoded.size() - 1);
        }

        switch_media_bug_t* bug = get_media_bug(psession);
        if (!bug) {
            push_err(out, m_sessionId, "processMessage - no media bug for injection");
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_ERROR,
                              "(%s) processMessage - no media bug (can't inject)\n", m_sessionId.c_str());
            return out;
        }
        auto* tech_pvt = (private_t*) switch_core_media_bug_get_user_data(bug);
        if (!tech_pvt) {
            push_err(out, m_sessionId, "processMessage - missing tech_pvt for injection");
            return out;
        }
        if (!tech_pvt->inject_buffer) {
            push_err(out, m_sessionId, "processMessage - inject_buffer not initialized");
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_ERROR,
                              "(%s) processMessage - inject_buffer not initialized\n", m_sessionId.c_str());
            return out;
        }

        if (!host_is_little_endian()) {
            byteswap_inplace_16(decoded);
        }

        int in_channels = 1;
        if (cJSON* jsonCh = cJSON_GetObjectItem(jsonData, "channels")) {
            if (cJSON_IsNumber(jsonCh) && jsonCh->valueint > 0) {
                in_channels = jsonCh->valueint;
            }
        }
        if (in_channels != 1 && in_channels != 2) {
            push_err(out, m_sessionId, "processMessage - unsupported channels (must be 1 or 2)");
            return out;
        }

        const int out_channels = (tech_pvt->channels == 2) ? 2 : 1;

        if (in_channels == 2 && out_channels == 1) {
            decoded = downmix_stereo_to_mono_pcm16le((const uint8_t*)decoded.data(), decoded.size());
        } else if (in_channels == 1 && out_channels == 2) {
            decoded = upmix_mono_to_stereo_pcm16le((const uint8_t*)decoded.data(), decoded.size());
        }

        const int out_sr = tech_pvt->sampling > 0 ? tech_pvt->sampling : tech_pvt->inject_sample_rate;
        if (out_sr <= 0) {
            push_err(out, m_sessionId, "processMessage - invalid output sample rate (session) for injection");
            return out;
        }

        if (sampleRate != out_sr) {
            switch_mutex_lock(tech_pvt->mutex);
            if (!tech_pvt->inject_resampler) {
                int err = 0;
                tech_pvt->inject_resampler = speex_resampler_init(out_channels, sampleRate, out_sr,
                                                                  SWITCH_RESAMPLE_QUALITY, &err);
                if (err != 0 || !tech_pvt->inject_resampler) {
                    switch_mutex_unlock(tech_pvt->mutex);
                    push_err(out, m_sessionId, "processMessage - failed to init inject resampler");
                    return out;
                }
                switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO,
                                  "(%s) processMessage: created inject resampler %d -> %d ch=%d\n",
                                  m_sessionId.c_str(), sampleRate, out_sr, out_channels);
            } else {
                spx_uint32_t in_r = 0, out_r = 0;
                speex_resampler_get_rate(tech_pvt->inject_resampler, &in_r, &out_r);
                if ((int)in_r != sampleRate || (int)out_r != out_sr) {
                    speex_resampler_destroy(tech_pvt->inject_resampler);
                    tech_pvt->inject_resampler = nullptr;
                    int err = 0;
                    tech_pvt->inject_resampler = speex_resampler_init(out_channels, sampleRate, out_sr,
                                                                      SWITCH_RESAMPLE_QUALITY, &err);
                    if (err != 0 || !tech_pvt->inject_resampler) {
                        switch_mutex_unlock(tech_pvt->mutex);
                        push_err(out, m_sessionId, "processMessage - failed to reinit inject resampler");
                        return out;
                    }
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO,
                                      "(%s) processMessage: re-created inject resampler %d -> %d ch=%d\n",
                                      m_sessionId.c_str(), sampleRate, out_sr, out_channels);
                }
            }

            /* IMPORTANT: keep the mutex held while using inject_resampler to avoid a
               use-after-free if cleanup destroys it concurrently. */
            decoded = resample_pcm16le_speex((const uint8_t*)decoded.data(), decoded.size(), out_channels,
                                            sampleRate, out_sr, tech_pvt->inject_resampler);

            switch_mutex_unlock(tech_pvt->mutex);
        
            sampleRate = out_sr;

            if (decoded.empty()) {
                push_err(out, m_sessionId, "processMessage - resample produced empty output");
                return out;
            }
        }

        const size_t frame_align = (size_t)out_channels * 2u;
        if (frame_align > 0 && (decoded.size() % frame_align) != 0) {
            decoded.resize(decoded.size() - (decoded.size() % frame_align));
        }

        const size_t frame_bytes_20ms = pcm16_bytes_per_ms(out_sr, out_channels) * 20u;
        if (frame_bytes_20ms > 0 && decoded.size() >= frame_bytes_20ms) {
            decoded.resize(decoded.size() - (decoded.size() % frame_bytes_20ms));
        }

        switch_mutex_lock(tech_pvt->mutex);

        tech_pvt->inject_sample_rate = out_sr;
        tech_pvt->inject_bytes_per_sample = 2;
        const int inject_sr = tech_pvt->inject_sample_rate;

        const switch_size_t inuse_before = switch_buffer_inuse(tech_pvt->inject_buffer);
        const switch_size_t free_before = switch_buffer_freespace(tech_pvt->inject_buffer);
        const size_t max_bytes = (size_t)inuse_before + (size_t)free_before;

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG,
                          "(%s) processMessage: inject_sample_rate=%d max_bytes=%zu (buffer_len)\n",
                          m_sessionId.c_str(), inject_sr, max_bytes);

        if (max_bytes > 0) {
            const switch_size_t inuse_locked = switch_buffer_inuse(tech_pvt->inject_buffer);
            const size_t incoming = decoded.size();
            if ((size_t)inuse_locked + incoming > max_bytes) {
                const size_t over = ((size_t)inuse_locked + incoming) - max_bytes;
                const size_t frame_bytes_20ms = pcm16_bytes_per_ms(out_sr, out_channels) * 20u;
                size_t drop = over;
                if (frame_bytes_20ms > 0) {
                    drop = ((over + frame_bytes_20ms - 1) / frame_bytes_20ms) * frame_bytes_20ms;
                } else {
                    const size_t sample_align = (size_t)out_channels * 2u;
                    if (sample_align > 0) {
                        drop = ((over + sample_align - 1) / sample_align) * sample_align;
                    }
                }

                const size_t inuse_sz = (size_t)inuse_locked;
                if (drop > inuse_sz) drop = inuse_sz;

                if (drop > 0) {
                    drop_oldest_from_buffer(tech_pvt->inject_buffer, (switch_size_t)drop);
                    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_DEBUG,
                                      "(%s) processMessage: inject buffer overflow, dropped %zu bytes (aligned)\n",
                                      m_sessionId.c_str(), drop);
                }
            }
        }

        switch_buffer_write(tech_pvt->inject_buffer, decoded.data(), (switch_size_t)decoded.size());
        const switch_size_t inuse_after = switch_buffer_inuse(tech_pvt->inject_buffer);
        switch_mutex_unlock(tech_pvt->mutex);

    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(psession), SWITCH_LOG_INFO,
              "(%s) PUSHBACK queued: decoded_bytes=%zu in_sr=%d out_sr=%d in_ch=%d out_ch=%d inject_inuse_after=%u\n",
              m_sessionId.c_str(), decoded.size(), sampleRate, out_sr, in_channels, out_channels, (unsigned)inuse_after);

        cJSON_AddNumberToObject(jsonData, "bytes", (double)decoded.size());

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

    static inline size_t pcm16_bytes_per_ms(int sampleRate, int channels) {
        if (sampleRate <= 0 || channels <= 0) return 0;
        return (size_t)sampleRate * 2u * (size_t)channels / 1000u;
    }

    static inline void drop_oldest_from_buffer(switch_buffer_t* buf, switch_size_t bytes) {
        if (!buf || bytes == 0) return;
        std::vector<uint8_t> tmp;
        tmp.resize((size_t)bytes);
        switch_buffer_read(buf, tmp.data(), bytes);
    }

    switch_status_t stream_data_init(private_t *tech_pvt, switch_core_session_t *session, char *wsUri,
                                     uint32_t sampling, int desiredSampling, int channels, char *metadata, responseHandler_t responseHandler,
                                     int deflate, int heart_beat, bool suppressLog, int rtp_packets, const char* extra_headers,
                                     const char *tls_cafile, const char *tls_keyfile, const char *tls_certfile, 
                                     bool tls_disable_hostname_validation)
    {
        int err; 

        switch_memory_pool_t *pool = switch_core_session_get_pool(session);

        const char* _uuid_log = switch_core_session_get_uuid(session);
    switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_INFO,
              "(%s) mod_audio_stream build version %s running\n",
              _uuid_log, MOD_AUDIO_STREAM_VERSION);

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

    /* per-session pushback counters */
    tech_pvt->inject_write_calls = 0;
    tech_pvt->inject_bytes = 0;
    tech_pvt->inject_underruns = 0;
    tech_pvt->inject_last_report = 0;

        if (metadata) {
            strncpy(tech_pvt->initialMetadata, metadata, MAX_METADATA_LEN);
            tech_pvt->initialMetadata[MAX_METADATA_LEN - 1] = '\0';
        }

        const size_t buflen = (FRAME_SIZE_8000 * desiredSampling / 8000 * channels * rtp_packets);

        
        switch_mutex_init(&tech_pvt->mutex, SWITCH_MUTEX_NESTED, pool);
        
        if (switch_buffer_create(pool, &tech_pvt->sbuffer, buflen) != SWITCH_STATUS_SUCCESS) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                "%s: Error creating switch buffer.\n", tech_pvt->sessionId);
            return SWITCH_STATUS_FALSE;
        }

        tech_pvt->inject_sample_rate = desiredSampling;
        tech_pvt->inject_bytes_per_sample = 2; 
        const size_t inject_bytes_per_ms = pcm16_bytes_per_ms(desiredSampling, channels);
        const size_t inject_buflen = std::max<size_t>(inject_bytes_per_ms * (size_t)INJECT_BUFFER_MS_DEFAULT, 3200u);
        if (switch_buffer_create(pool, &tech_pvt->inject_buffer, inject_buflen) != SWITCH_STATUS_SUCCESS) {
            switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_ERROR,
                "%s: Error creating inject buffer.\n", tech_pvt->sessionId);
            return SWITCH_STATUS_FALSE;
        }

        auto sp = AudioStreamer::create(tech_pvt->sessionId, wsUri, responseHandler, deflate, heart_beat,
                                        suppressLog, extra_headers, tls_cafile, tls_keyfile,
                                        tls_certfile, tls_disable_hostname_validation);

        tech_pvt->pAudioStreamer = new std::shared_ptr<AudioStreamer>(sp);

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

        tech_pvt->inject_resampler = nullptr;

        switch_log_printf(SWITCH_CHANNEL_SESSION_LOG(session), SWITCH_LOG_DEBUG, "(%s) stream_data_init\n", tech_pvt->sessionId);

        return SWITCH_STATUS_SUCCESS;
    }

    void destroy_tech_pvt(private_t* tech_pvt) {
        switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, "%s destroy_tech_pvt\n", tech_pvt->sessionId);
        if (tech_pvt->resampler) {
            speex_resampler_destroy(tech_pvt->resampler);
            tech_pvt->resampler = nullptr;
        }
        if (tech_pvt->inject_resampler) {
            speex_resampler_destroy(tech_pvt->inject_resampler);
            tech_pvt->inject_resampler = nullptr;
        }
        /* tech_pvt->mutex comes from the session pool; avoid destroying it explicitly.
           The pool cleanup will reclaim it safely after all media threads have stopped. */
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

        if (strncmp(url, "ws://", 5) == 0) {
            scheme = "ws";
            hostStart = url + 5;
        } else if (strncmp(url, "wss://", 6) == 0) {
            scheme = "wss";
            hostStart = url + 6;
        } else {
            return 0;
        }

        hostEnd = hostStart;
        while (*hostEnd && *hostEnd != ':' && *hostEnd != '/') {
            const unsigned char ch = (unsigned char)*hostEnd;
            if (!std::isalnum(ch) && *hostEnd != '-' && *hostEnd != '.') {
                return 0;
            }
            ++hostEnd;
        }

        if (hostStart == hostEnd) {
            return 0;
        }

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

        std::strncpy(wsUri, url, MAX_WS_URI);
        wsUri[MAX_WS_URI - 1] = '\0';
        return 1;
    }

    switch_status_t is_valid_utf8(const char *str) {
        switch_status_t status = SWITCH_STATUS_FALSE;
        while (*str) {
            if ((*str & 0x80) == 0x00) {
                str++;
            } else if ((*str & 0xE0) == 0xC0) {
                if ((str[1] & 0xC0) != 0x80) {
                    return status;
                }
                str += 2;
            } else if ((*str & 0xF0) == 0xE0) {
                if ((str[1] & 0xC0) != 0x80 || (str[2] & 0xC0) != 0x80) {
                    return status;
                }
                str += 3;
            } else if ((*str & 0xF8) == 0xF0) {
                if ((str[1] & 0xC0) != 0x80 || (str[2] & 0xC0) != 0x80 || (str[3] & 0xC0) != 0x80) {
                    return status;
                }
                str += 4;
            } else {
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
                streamer = *sp_wrap; 
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

        SpeexResamplerState *resampler = nullptr;
        int channels = 1;
        int rtp_packets = 1;
        switch_buffer_t *sbuffer = nullptr;

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

        resampler = tech_pvt->resampler;
        channels = tech_pvt->channels;
        rtp_packets = tech_pvt->rtp_packets;
        sbuffer = tech_pvt->sbuffer;

        switch_mutex_unlock(tech_pvt->mutex);

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

                switch_mutex_lock(tech_pvt->mutex);
                size_t freespace = switch_buffer_freespace(sbuffer);
                
                if (freespace >= frame.datalen) {
                    switch_buffer_write(sbuffer, static_cast<uint8_t *>(frame.data), frame.datalen);
                }

                if (switch_buffer_freespace(sbuffer) == 0) {
                    switch_size_t inuse = switch_buffer_inuse(sbuffer);
                    if (inuse > 0) {
                        std::vector<uint8_t> tmp(inuse);
                        switch_buffer_read(sbuffer, tmp.data(), inuse);
                        switch_buffer_zero(sbuffer);
                        pending_send.emplace_back(std::move(tmp));
                    }
                }

                switch_mutex_unlock(tech_pvt->mutex);
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

                const size_t freespace = switch_buffer_freespace(sbuffer);
                spx_uint32_t in_len = frame.samples;
                spx_uint32_t out_len = (freespace / (channels * sizeof(spx_int16_t)));
                
                if(out_len == 0) {
                    if(freespace == 0) {
                        switch_size_t inuse = switch_buffer_inuse(sbuffer);
                        if (inuse > 0) {
                            std::vector<uint8_t> tmp(inuse);
                            switch_buffer_read(sbuffer, tmp.data(), inuse);
                            switch_buffer_zero(sbuffer);
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

                    if (rtp_packets == 1) { 
                        const uint8_t* p = (const uint8_t*)out.data();
                        pending_send.emplace_back(p, p + bytes_written);
                        continue;
                    }

                    if (bytes_written <= switch_buffer_freespace(tech_pvt->sbuffer)) {
                        switch_buffer_write(sbuffer, (const uint8_t *)out.data(), bytes_written);
                    }
                }

                if (switch_buffer_freespace(sbuffer) == 0) {
                    switch_size_t inuse = switch_buffer_inuse(sbuffer);
                    if (inuse > 0) {
                        std::vector<uint8_t> tmp(inuse);
                        switch_buffer_read(sbuffer, tmp.data(), inuse);
                        switch_buffer_zero(sbuffer);
                        pending_send.emplace_back(std::move(tmp));
                    }
                }
            }
        }
    
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