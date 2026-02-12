#include "openai_realtime.h"
#include "WebSocketClient.h"
#include <sstream>
#include <cstring>
#include <algorithm>
extern "C" {
}
#include "../base64.h"

namespace ai_engine {

std::string OpenAIRealtimeClient::json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        search = "\"" + key + "\": \"";
        pos = json.find(search);
        if (pos == std::string::npos) {
            search = "\"" + key + "\":  \"";
            pos = json.find(search);
            if (pos == std::string::npos) return "";
        }
    }
    pos += search.size();
    size_t end = json.find('"', pos);
    if (end == std::string::npos) return "";
    while (end > 0 && json[end - 1] == '\\') {
        end = json.find('"', end + 1);
        if (end == std::string::npos) return "";
    }
    return json.substr(pos, end - pos);
}

std::string OpenAIRealtimeClient::json_get_object(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":{";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        search = "\"" + key + "\": {";
        pos = json.find(search);
        if (pos == std::string::npos) return "";
    }
    pos = json.find('{', pos);
    if (pos == std::string::npos) return "";
    int depth = 0;
    size_t start = pos;
    for (size_t i = pos; i < json.size(); ++i) {
        if (json[i] == '{') depth++;
        else if (json[i] == '}') {
            depth--;
            if (depth == 0) return json.substr(start, i - start + 1);
        }
    }
    return "";
}

bool OpenAIRealtimeClient::json_has_key(const std::string& json, const std::string& key) {
    return json.find("\"" + key + "\"") != std::string::npos;
}

std::string OpenAIRealtimeClient::base64_encode_pcm(const int16_t* samples, size_t num_samples) {
    const unsigned char* data = reinterpret_cast<const unsigned char*>(samples);
    size_t bytes = num_samples * sizeof(int16_t);
    return base64_encode(data, bytes);
}

OpenAIRealtimeClient::OpenAIRealtimeClient()
    : ws_(std::make_unique<WebSocketClient>())
{
}

OpenAIRealtimeClient::~OpenAIRealtimeClient() {
    disconnect();
}

void OpenAIRealtimeClient::connect(const OpenAIRealtimeConfig& cfg) {
    cfg_ = cfg;
    std::string url = "wss://api.openai.com/v1/realtime?model=" + cfg_.model;
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
        connected_.store(true, std::memory_order_release);
        if (cb_connection_) cb_connection_(true);
        send_session_update();
    });
    ws_->setCloseCallback([this](int code, const std::string& reason) {
        connected_.store(false, std::memory_order_release);
        session_configured_.store(false, std::memory_order_release);
        is_responding_.store(false, std::memory_order_release);
        if (cb_connection_) cb_connection_(false);
    });
    ws_->setErrorCallback([this](int code, const std::string& msg) {
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
    ws_->sendMessage("{\"type\":\"response.cancel\"}");
    ws_->sendMessage("{\"type\":\"input_audio_buffer.clear\"}");
    is_responding_.store(false, std::memory_order_release);
}

void OpenAIRealtimeClient::send_text_message(const std::string& text) {
    if (!is_connected() || text.empty()) return;
    std::ostringstream oss;
    oss << "{\"type\":\"conversation.item.create\",\"item\":{\"type\":\"message\"," << "\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"";
    for (char c : text) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n"; break;
            default: oss << c; break;
        }
    }
    oss << "\"}]}}";
    ws_->sendMessage(oss.str());
    ws_->sendMessage("{\"type\":\"response.create\"}");
}

void OpenAIRealtimeClient::send_session_update() {
    std::string config = build_session_config();
    ws_->sendMessage(config);
}

std::string OpenAIRealtimeClient::build_session_config() const {
    std::ostringstream oss;
    oss << "{\"type\":\"session.update\",\"session\":{" << "\"modalities\":[\"text\"]," << "\"instructions\":\"";
    for (char c : cfg_.system_prompt) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n"; break;
            default: oss << c; break;
        }
    }
    oss << "\"," << "\"input_audio_format\":\"" << cfg_.input_audio_format << "\"," << "\"input_audio_transcription\":{\"model\":\"whisper-1\"},";
    if (cfg_.vad_type == "server_vad") {
        oss << "\"turn_detection\":{" << "\"type\":\"server_vad\"," << "\"threshold\":" << cfg_.vad_threshold << "," << "\"prefix_padding_ms\":" << cfg_.vad_prefix_padding_ms << "," << "\"silence_duration_ms\":" << cfg_.vad_silence_duration_ms << "},";
    } else {
        oss << "\"turn_detection\":null,";
    }
    oss << "\"temperature\":" << cfg_.temperature << "," << "\"max_response_output_tokens\":" << cfg_.max_response_output_tokens << "}}";
    return oss.str();
}

void OpenAIRealtimeClient::handle_message(const std::string& message) {
    std::string type = json_get_string(message, "type");
    if (type.empty()) return;
    if (type == "session.created") {
        session_id_ = json_get_string(message, "id");
        if (cb_session_created_) {
            cb_session_created_(session_id_);
        }
        return;
    }
    if (type == "session.updated") {
        session_configured_.store(true, std::memory_order_release);
        return;
    }
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
    if (type == "conversation.item.input_audio_transcription.completed") {
        std::string transcript = json_get_string(message, "transcript");
        if (cb_input_transcript_ && !transcript.empty()) {
            cb_input_transcript_(transcript);
        }
        return;
    }
    if (type == "response.created") {
        current_response_id_ = json_get_string(message, "id");
        if (current_response_id_.empty()) {
            std::string resp = json_get_object(message, "response");
            if (!resp.empty()) {
                current_response_id_ = json_get_string(resp, "id");
            }
        }
        current_response_text_.clear();
        is_responding_.store(true, std::memory_order_release);
        return;
    }
    if (type == "response.text.delta") {
        std::string delta = json_get_string(message, "delta");
        if (!delta.empty()) {
            std::string unescaped;
            unescaped.reserve(delta.size());
            for (size_t i = 0; i < delta.size(); ++i) {
                if (delta[i] == '\\' && i + 1 < delta.size()) {
                    char next = delta[i + 1];
                    if (next == 'n') { unescaped += '\n'; i++; }
                    else if (next == 't') { unescaped += '\t'; i++; }
                    else if (next == '"') { unescaped += '"'; i++; }
                    else if (next == '\\') { unescaped += '\\'; i++; }
                    else { unescaped += delta[i]; }
                } else {
                    unescaped += delta[i];
                }
            }
            current_response_text_ += unescaped;
            if (cb_text_delta_) {
                cb_text_delta_(unescaped, current_response_id_);
            }
        }
        return;
    }
    if (type == "response.text.done") {
        return;
    }
    if (type == "response.done") {
        is_responding_.store(false, std::memory_order_release);
        if (cb_response_done_) {
            cb_response_done_(current_response_text_, current_response_id_);
        }
        current_response_text_.clear();
        current_response_id_.clear();
        return;
    }
    if (type == "response.cancelled" || type == "response.interrupted") {
        is_responding_.store(false, std::memory_order_release);
        if (cb_response_interrupted_) cb_response_interrupted_();
        current_response_text_.clear();
        return;
    }
    if (type == "error") {
        std::string error_msg = json_get_string(message, "message");
        std::string error_code = json_get_string(message, "code");
        if (error_msg.empty()) {
            std::string err_obj = json_get_object(message, "error");
            if (!err_obj.empty()) {
                error_msg = json_get_string(err_obj, "message");
                error_code = json_get_string(err_obj, "code");
            }
        }
        if (cb_error_) cb_error_(error_msg, error_code);
        return;
    }
    if (type == "rate_limits.updated") {
        return;
    }
}

}
