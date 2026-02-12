#ifndef AI_ENGINE_OPENAI_REALTIME_H
#define AI_ENGINE_OPENAI_REALTIME_H

#include <string>
#include <functional>
#include <memory>
#include <atomic>
#include <mutex>
#include <cstdint>
#include <vector>

class WebSocketClient;

namespace ai_engine {

struct OpenAIRealtimeConfig {
    
    std::string api_key;
    std::string model          = "gpt-4o-realtime-preview";
    std::string system_prompt  = "You are a helpful AI assistant on a phone call. "
                                  "Keep responses brief and conversational. "
                                  "Respond in the same language the caller uses.";
    std::string vad_type       = "server_vad"; 
    float       vad_threshold  = 0.5f;
    int         vad_prefix_padding_ms  = 300;
    int         vad_silence_duration_ms = 500;
    int         input_sample_rate  = 16000;   
    std::string input_audio_format = "pcm16"; 
    bool        eagerly_emit_text  = true;    
    float       temperature        = 0.8f;
    int         max_response_output_tokens = 4096;
    int         connect_timeout_ms = 5000;
    int         ping_interval_s    = 25;
    std::vector<std::string> tools;
};

using OnSessionCreated = std::function<void(const std::string& session_id)>;
using OnResponseTextDelta = std::function<void(const std::string& delta,
                                                 const std::string& response_id)>;
using OnResponseDone = std::function<void(const std::string& full_text,
                                           const std::string& response_id)>;
using OnTranscript = std::function<void(const std::string& transcript,
                                         bool is_final)>;
using OnSpeechStarted = std::function<void()>;
using OnSpeechStopped = std::function<void()>;
using OnInputTranscriptDone = std::function<void(const std::string& transcript)>;
using OnResponseInterrupted = std::function<void()>;
using OnError = std::function<void(const std::string& error, const std::string& code)>;
using OnConnectionChange = std::function<void(bool connected)>;

class OpenAIRealtimeClient {
public:
    OpenAIRealtimeClient();
    ~OpenAIRealtimeClient();
    OpenAIRealtimeClient(const OpenAIRealtimeClient&) = delete;
    OpenAIRealtimeClient& operator=(const OpenAIRealtimeClient&) = delete;

    void connect(const OpenAIRealtimeConfig& cfg);
    void disconnect();
    bool is_connected() const;
    void send_audio(const int16_t* samples, size_t num_samples);
    void commit_audio();
    void cancel_response();
    void send_text_message(const std::string& text);
    void on_session_created(OnSessionCreated cb)         { cb_session_created_ = std::move(cb); }
    void on_response_text_delta(OnResponseTextDelta cb)  { cb_text_delta_ = std::move(cb); }
    void on_response_done(OnResponseDone cb)             { cb_response_done_ = std::move(cb); }
    void on_transcript(OnTranscript cb)                  { cb_transcript_ = std::move(cb); }
    void on_speech_started(OnSpeechStarted cb)           { cb_speech_started_ = std::move(cb); }
    void on_speech_stopped(OnSpeechStopped cb)           { cb_speech_stopped_ = std::move(cb); }
    void on_input_transcript_done(OnInputTranscriptDone cb) { cb_input_transcript_ = std::move(cb); }
    void on_response_interrupted(OnResponseInterrupted cb)  { cb_response_interrupted_ = std::move(cb); }
    void on_error(OnError cb)                            { cb_error_ = std::move(cb); }
    void on_connection_change(OnConnectionChange cb)     { cb_connection_ = std::move(cb); }
    bool is_responding() const { return is_responding_.load(std::memory_order_relaxed); }
    bool is_speech_active() const { return is_speech_active_.load(std::memory_order_relaxed); }

private:

    std::unique_ptr<WebSocketClient> ws_;
    OpenAIRealtimeConfig cfg_;
    std::atomic<bool> connected_{false};
    std::atomic<bool> session_configured_{false};
    std::atomic<bool> is_responding_{false};
    std::atomic<bool> is_speech_active_{false};
    std::string       session_id_;
    std::string       current_response_id_;
    std::string       current_response_text_;
    OnSessionCreated       cb_session_created_;
    OnResponseTextDelta    cb_text_delta_;
    OnResponseDone         cb_response_done_;
    OnTranscript           cb_transcript_;
    OnSpeechStarted        cb_speech_started_;
    OnSpeechStopped        cb_speech_stopped_;
    OnInputTranscriptDone  cb_input_transcript_;
    OnResponseInterrupted  cb_response_interrupted_;
    OnError                cb_error_;
    OnConnectionChange     cb_connection_;

    void handle_message(const std::string& message);
    void send_session_update();
    std::string build_session_config() const;
    static std::string json_get_string(const std::string& json, const std::string& key);
    static std::string json_get_object(const std::string& json, const std::string& key);
    static bool json_has_key(const std::string& json, const std::string& key);
    static std::string base64_encode_pcm(const int16_t* samples, size_t num_samples);
};

}

#endif
