#ifndef AI_ENGINE_AI_ENGINE_H
#define AI_ENGINE_AI_ENGINE_H
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <queue>
#include <condition_variable>
#include <functional>
#include <cstdint>
#include <cstdio>
#include "openai_realtime.h"
#include "sentence_buffer.h"
#include "tts_engine.h"
#include "tts_cache.h"
#include "dsp_pipeline.h"
#include "ring_buffer.h"

struct switch_core_session;
typedef struct switch_core_session switch_core_session_t;

struct SpeexResamplerState_;
typedef struct SpeexResamplerState_ SpeexResamplerState;

namespace ai_engine {

/* ---- Shared JSON string escaping utility ---- */
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
                break;
        }
    }
    return out;
}

struct AIEngineConfig {
    
    OpenAIRealtimeConfig openai;
    TTSConfig tts;
    DSPConfig dsp;

    SentenceBufferConfig sentence;
    TTSCacheConfig cache;

    int  freeswitch_sample_rate = 8000;   
    int  openai_send_rate       = 24000;  
    int  tts_worker_threads     = 1;       
    int  inject_buffer_ms       = 5000;   
    bool enable_barge_in        = true;    
    int  barge_in_min_ms        = 200;    
    bool debug_logging          = false;
    std::string session_uuid;              
};

enum class AIEngineState {
    IDLE,
    CONNECTING,
    LISTENING,
    PROCESSING,
    SPEAKING,
    ERROR,
    SHUTDOWN
};

using OnStateChange = std::function<void(AIEngineState state, const std::string& detail)>;
using OnAudioReady = std::function<void(const int16_t* samples, size_t count, int sample_rate)>;
using OnAIEvent = std::function<void(const std::string& event_name, const std::string& json)>;

struct TTSWorkItem {
    std::string sentence;
    bool        is_final = false;
    uint64_t    sequence = 0; 
};

class AIEngine {
public:
    AIEngine();
    ~AIEngine();

    AIEngine(const AIEngine&) = delete;
    AIEngine& operator=(const AIEngine&) = delete;
    bool start(const AIEngineConfig& cfg, switch_core_session_t* session);
    void stop();
    bool is_running() const;
    void feed_audio(const int16_t* samples, size_t num_samples);
    size_t read_audio(int16_t* dest, size_t num_samples);
    double available_audio_ms() const;
    void handle_barge_in();
    void set_state_callback(OnStateChange cb)  { cb_state_ = std::move(cb); }
    void set_audio_callback(OnAudioReady cb)   { cb_audio_ = std::move(cb); }
    void set_event_callback(OnAIEvent cb)      { cb_event_ = std::move(cb); }
    AIEngineState state() const { return state_.load(std::memory_order_acquire); }
    bool is_speaking() const { return state() == AIEngineState::SPEAKING; }
    bool is_listening() const { return state() == AIEngineState::LISTENING; }

    struct Stats {
        uint64_t audio_frames_sent    = 0;  
        uint64_t text_tokens_received = 0;  
        uint64_t sentences_synthesized = 0;
        uint64_t tts_cache_hits       = 0;  
        uint64_t barge_ins            = 0;  
        double   avg_tts_latency_ms   = 0;  
    };

    Stats get_stats() const;

private:

    AIEngineConfig                      cfg_;
    std::unique_ptr<OpenAIRealtimeClient> openai_;
    std::mutex                          openai_mutex_;
    std::unique_ptr<ITTSEngine>         tts_engine_;
    std::unique_ptr<TTSCache>           tts_cache_;
    SentenceBuffer                      sentence_buffer_;
    DSPPipeline                         dsp_;
    std::unique_ptr<SPSCRingBuffer>     ring_buffer_;
    SpeexResamplerState*                upsample_resampler_ = nullptr;
    SpeexResamplerState*                downsample_resampler_ = nullptr;
    std::mutex                          resampler_mutex_;
    std::atomic<AIEngineState>  state_{AIEngineState::IDLE};
    std::atomic<bool>           running_{false};
    std::atomic<bool>           tts_abort_{false};
    std::atomic<int>            reconnect_attempts_{0};
    static constexpr int        kMaxReconnectAttempts = 5;
    std::thread                 reconnect_thread_;
    std::mutex                  reconnect_mutex_;
    std::mutex                  sentence_mutex_;
    std::thread                         tts_thread_;
    std::queue<TTSWorkItem>             tts_queue_;
    std::mutex                          tts_queue_mutex_;
    std::condition_variable             tts_queue_cv_;
    std::atomic<uint64_t>               tts_sequence_{0};

    void tts_worker_loop();
    void process_tts_item(const TTSWorkItem& item);
    void on_openai_connected(const std::string& session_id);
    void on_openai_text_delta(const std::string& delta, const std::string& response_id);
    void on_openai_response_done(const std::string& full_text, const std::string& response_id);
    void on_openai_speech_started();
    void on_openai_speech_stopped();
    void on_openai_input_transcript(const std::string& transcript);
    void on_openai_error(const std::string& error, const std::string& code);
    void on_openai_connection_change(bool connected);
    void on_tts_audio(const int16_t* samples, size_t count,
                      bool is_final, const std::string& sentence,
                      int tts_sr);
    void set_state(AIEngineState new_state, const std::string& detail = "");
    void flush_tts_queue();
    void resample_up(const int16_t* in, size_t in_samples,
                     std::vector<int16_t>& out);
    void resample_down(const int16_t* in, size_t in_samples,
                       int in_rate, std::vector<int16_t>& out);

    OnStateChange  cb_state_;
    OnAudioReady   cb_audio_;
    OnAIEvent      cb_event_;
    mutable std::mutex stats_mutex_;
    Stats stats_;
};

} 

#endif