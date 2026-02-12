#include "openai_realtime.h"
#include "WebSocketClient.h"
#include <sstream>
#include <cstring>
#include <algorithm>

extern "C" {
}

#include "../base64.h"

#ifdef HAVE_SWITCH_H
#include <switch.h>
#define OAI_LOG_INFO(fmt, ...)  switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_INFO, fmt, ##__VA_ARGS__)
#define OAI_LOG_ERROR(fmt, ...) switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_ERROR, fmt, ##__VA_ARGS__)
#define OAI_LOG_DEBUG(fmt, ...) switch_log_printf(SWITCH_CHANNEL_LOG, SWITCH_LOG_DEBUG, fmt, ##__VA_ARGS__)
#else
#include <cstdio>
#define OAI_LOG_INFO(fmt, ...)  fprintf(stderr, "[OAI-INFO] " fmt, ##__VA_ARGS__)
#define OAI_LOG_ERROR(fmt, ...) fprintf(stderr, "[OAI-ERROR] " fmt, ##__VA_ARGS__)
#define OAI_LOG_DEBUG(fmt, ...) fprintf(stderr, "[OAI-DEBUG] " fmt, ##__VA_ARGS__)
#endif

namespace ai_engine {

/* ---------- Robust JSON helpers ---------- */

static size_t skip_ws(const std::string& s, size_t pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                              s[pos] == '\n' || s[pos] == '\r'))
        ++pos;
    return pos;
}

static size_t find_key(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while (pos < json.size()) {
        pos = json.find(needle, pos);
        if (pos == std::string::npos) return std::string::npos;
        pos += needle.size();
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ':') {
            return pos + 1;
        }
    }
    return std::string::npos;
}

std::string OpenAIRealtimeClient::json_get_string(const std::string& json,
                                                   const std::string& key) {
    size_t pos = find_key(json, key);
    if (pos == std::string::npos) return "";
    pos = skip_ws(json, pos);
    if (pos >= json.size() || json[pos] != '"') return "";
    ++pos;

    std::string result;
    result.reserve(128);
    while (pos < json.size()) {
        char c = json[pos];
        if (c == '\\' && pos + 1 < json.size()) {
            char next = json[pos + 1];
            switch (next) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case '/':  result += '/';  break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                case 'b':  result += '\b'; break;
                case 'f':  result += '\f'; break;
                default:   result += '\\'; result += next; break;
            }
            pos += 2;
            continue;
        }
        if (c == '"') break;
        result += c;
        ++pos;
    }
    return result;
}

std::string OpenAIRealtimeClient::json_get_object(const std::string& json,
                                                   const std::string& key) {
    size_t pos = find_key(json, key);
    if (pos == std::string::npos) return "";
    pos = skip_ws(json, pos);
    if (pos >= json.size() || json[pos] != '{') return "";

    int depth = 0;
    size_t start = pos;
    bool in_string = false;
    for (size_t i = pos; i < json.size(); ++i) {
        char c = json[i];
        if (c == '\\' && in_string) { ++i; continue; }
        if (c == '"') { in_string = !in_string; continue; }
        if (in_string) continue;
        if (c == '{') ++depth;
        else if (c == '}') {
            --depth;
            if (depth == 0) return json.substr(start, i - start + 1);
        }
    }
    return "";
}

bool OpenAIRealtimeClient::json_has_key(const std::string& json,
                                         const std::string& key) {
    return find_key(json, key) != std::string::npos;
}

std::string OpenAIRealtimeClient::base64_encode_pcm(const int16_t* samples,
                                                     size_t num_samples) {
    const unsigned char* data = reinterpret_cast<const unsigned char*>(samples);
    size_t bytes = num_samples * sizeof(int16_t);
    return base64_encode(data, bytes);
}

/* ---------- Lifecycle ---------- */

OpenAIRealtimeClient::OpenAIRealtimeClient()
    : ws_(std::make_unique<WebSocketClient>())
{
}

OpenAIRealtimeClient::~OpenAIRealtimeClient() {
    disconnect();
}

void OpenAIRealtimeClient::connect(const OpenAIRealtimeConfig& cfg) {
    cfg_ = cfg;

    if (connected_.load(std::memory_order_acquire)) {
        disconnect();
    }

    std::string url = "wss://api.openai.com/v1/realtime?model=" + cfg_.model;
    OAI_LOG_INFO("OpenAI Realtime: connecting to %s\n", url.c_str());

    ws_->setUrl(url);

    WebSocketHeaders headers;
    headers.set("Authorization", "Bearer " + cfg_.api_key);
    headers.set("OpenAI-Beta", "realtime=v1");
    ws_->setHeaders(headers);

    ws_->setConnectionTimeout(cfg_.connect_timeout_ms / 1000);
    ws_->setPingInterval(cfg_.ping_interval_s);
    ws_->enableCompression(false);

    WebSocketTLSOptions tls;
    tls.disableHostnameValidation = false;
    ws_->setTLSOptions(tls);

    ws_->setOpenCallback([this]() {
        OAI_LOG_INFO("OpenAI Realtime: WebSocket connected\n");
        connected_.store(true, std::memory_order_release);
        if (cb_connection_) cb_connection_(true);
        send_session_update();
    });

    ws_->setCloseCallback([this](int code, const std::string& reason) {
        OAI_LOG_INFO("OpenAI Realtime: closed (code=%d reason=%s)\n", code, reason.c_str());
        connected_.store(false, std::memory_order_release);
        session_configured_.store(false, std::memory_order_release);
        is_responding_.store(false, std::memory_order_release);
        if (cb_connection_) cb_connection_(false);
    });

    ws_->setErrorCallback([this](int code, const std::string& msg) {
        OAI_LOG_ERROR("OpenAI Realtime: error (code=%d msg=%s)\n", code, msg.c_str());
        if (cb_error_) cb_error_(msg, std::to_string(code));
        connected_.store(false, std::memory_order_release);
        if (cb_connection_) cb_connection_(false);
    });

    ws_->setMessageCallback([this](const std::string& message) {
        handle_message(message);
    });

    ws_->connect();
}

void OpenAIRealtimeClient::disconnect() {
    if (ws_) {
        ws_->setMessageCallback({});
        ws_->setOpenCallback({});
        ws_->setCloseCallback({});
        ws_->setErrorCallback({});
        ws_->disconnect();
    }
    connected_.store(false, std::memory_order_release);
    session_configured_.store(false, std::memory_order_release);
    is_responding_.store(false, std::memory_order_release);
}

bool OpenAIRealtimeClient::is_connected() const {
    return connected_.load(std::memory_order_acquire);
}

/* ---------- Send methods ---------- */

void OpenAIRealtimeClient::send_audio(const int16_t* samples, size_t num_samples) {
    if (!is_connected() || !samples || num_samples == 0) return;
    std::string b64 = base64_encode_pcm(samples, num_samples);
    std::string msg;
    msg.reserve(b64.size() + 64);
    msg = "{\"type\":\"input_audio_buffer.append\",\"audio\":\"";
    msg += b64;
    msg += "\"}";
    ws_->sendMessage(msg);
}

void OpenAIRealtimeClient::commit_audio() {
    if (!is_connected()) return;
    ws_->sendMessage("{\"type\":\"input_audio_buffer.commit\"}");
}

void OpenAIRealtimeClient::cancel_response() {
    if (!is_connected()) return;
    OAI_LOG_INFO("OpenAI Realtime: cancelling response\n");
    ws_->sendMessage("{\"type\":\"response.cancel\"}");
    ws_->sendMessage("{\"type\":\"input_audio_buffer.clear\"}");
    is_responding_.store(false, std::memory_order_release);
}

void OpenAIRealtimeClient::send_text_message(const std::string& text) {
    if (!is_connected() || text.empty()) return;
    std::ostringstream oss;
    oss << "{\"type\":\"conversation.item.create\",\"item\":{\"type\":\"message\","
        << "\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"";
    for (char c : text) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            default:   oss << c;      break;
        }
    }
    oss << "\"}]}}";
    ws_->sendMessage(oss.str());
    ws_->sendMessage("{\"type\":\"response.create\"}");
}

/* ---------- Session update ---------- */

void OpenAIRealtimeClient::send_session_update() {
    std::string config = build_session_config();
    OAI_LOG_DEBUG("OpenAI Realtime: sending session.update\n");
    ws_->sendMessage(config);
}

std::string OpenAIRealtimeClient::build_session_config() const {
    std::ostringstream oss;
    oss << "{\"type\":\"session.update\",\"session\":{"
        << "\"modalities\":[\"text\"],"
        << "\"instructions\":\"";

    for (char c : cfg_.system_prompt) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            case '\t': oss << "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    oss << buf;
                } else {
                    oss << c;
                }
                break;
        }
    }

    oss << "\","
        << "\"input_audio_format\":\"" << cfg_.input_audio_format << "\","
        << "\"input_audio_transcription\":{\"model\":\"whisper-1\"},";

    if (cfg_.vad_type == "server_vad") {
        oss << "\"turn_detection\":{"
            << "\"type\":\"server_vad\","
            << "\"threshold\":" << cfg_.vad_threshold << ","
            << "\"prefix_padding_ms\":" << cfg_.vad_prefix_padding_ms << ","
            << "\"silence_duration_ms\":" << cfg_.vad_silence_duration_ms
            << "},";
    } else {
        oss << "\"turn_detection\":null,";
    }

    oss << "\"temperature\":" << cfg_.temperature << ","
        << "\"max_response_output_tokens\":" << cfg_.max_response_output_tokens
        << "}}";

    return oss.str();
}

/* ---------- Message handler ---------- */

void OpenAIRealtimeClient::handle_message(const std::string& message) {
    std::string type = json_get_string(message, "type");
    if (type.empty()) return;

    /* session.created */
    if (type == "session.created") {
        std::string session_obj = json_get_object(message, "session");
        if (!session_obj.empty()) {
            session_id_ = json_get_string(session_obj, "id");
        }
        if (session_id_.empty()) {
            session_id_ = json_get_string(message, "id");
        }
        OAI_LOG_INFO("OpenAI Realtime: session created (id=%s)\n", session_id_.c_str());
        if (cb_session_created_) cb_session_created_(session_id_);
        return;
    }

    /* session.updated */
    if (type == "session.updated") {
        session_configured_.store(true, std::memory_order_release);
        OAI_LOG_INFO("OpenAI Realtime: session configured\n");
        return;
    }

    /* VAD events */
    if (type == "input_audio_buffer.speech_started") {
        is_speech_active_.store(true, std::memory_order_release);
        if (cb_speech_started_) cb_speech_started_();
        return;
    }
    if (type == "input_audio_buffer.speech_stopped") {
        is_speech_active_.store(false, std::memory_order_release);
        if (cb_speech_stopped_) cb_speech_stopped_();
        return;
    }

    /* Input transcription */
    if (type == "conversation.item.input_audio_transcription.completed") {
        std::string transcript = json_get_string(message, "transcript");
        if (cb_input_transcript_ && !transcript.empty())
            cb_input_transcript_(transcript);
        return;
    }
    if (type == "conversation.item.input_audio_transcription.failed") {
        std::string err_obj = json_get_object(message, "error");
        std::string err_msg = err_obj.empty() ? "transcription failed" :
                              json_get_string(err_obj, "message");
        OAI_LOG_ERROR("OpenAI Realtime: transcription failed: %s\n", err_msg.c_str());
        return;
    }

    /* Response lifecycle */
    if (type == "response.created") {
        std::string resp = json_get_object(message, "response");
        if (!resp.empty()) current_response_id_ = json_get_string(resp, "id");
        if (current_response_id_.empty()) current_response_id_ = json_get_string(message, "id");
        current_response_text_.clear();
        is_responding_.store(true, std::memory_order_release);
        return;
    }

    /* Text deltas */
    if (type == "response.text.delta" || type == "response.audio_transcript.delta") {
        std::string delta = json_get_string(message, "delta");
        if (!delta.empty()) {
            current_response_text_ += delta;
            if (cb_text_delta_) cb_text_delta_(delta, current_response_id_);
        }
        return;
    }

    if (type == "response.text.done" || type == "response.audio_transcript.done") {
        return;
    }

    /* Informational response events */
    if (type == "response.output_item.added" || type == "response.content_part.added" ||
        type == "response.output_item.done" || type == "response.content_part.done") {
        return;
    }

    /* Response done */
    if (type == "response.done") {
        is_responding_.store(false, std::memory_order_release);
        if (cb_response_done_) cb_response_done_(current_response_text_, current_response_id_);
        current_response_text_.clear();
        current_response_id_.clear();
        return;
    }

    /* Response cancelled */
    if (type == "response.cancelled" || type == "response.interrupted") {
        is_responding_.store(false, std::memory_order_release);
        if (cb_response_interrupted_) cb_response_interrupted_();
        current_response_text_.clear();
        return;
    }

    /* Errors */
    if (type == "error") {
        std::string error_msg, error_code;
        std::string err_obj = json_get_object(message, "error");
        if (!err_obj.empty()) {
            error_msg = json_get_string(err_obj, "message");
            error_code = json_get_string(err_obj, "code");
            if (error_code.empty()) error_code = json_get_string(err_obj, "type");
        }
        if (error_msg.empty()) {
            error_msg = json_get_string(message, "message");
            error_code = json_get_string(message, "code");
        }
        OAI_LOG_ERROR("OpenAI Realtime: error: %s (code=%s)\n", error_msg.c_str(), error_code.c_str());
        if (cb_error_) cb_error_(error_msg, error_code);
        return;
    }

    /* Rate limits (ignore) */
    if (type == "rate_limits.updated") return;

    OAI_LOG_DEBUG("OpenAI Realtime: unhandled event: %s\n", type.c_str());
}

} /* namespace ai_engine */
