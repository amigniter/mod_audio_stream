#ifndef AI_ENGINE_TTS_ENGINE_H
#define AI_ENGINE_TTS_ENGINE_H

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

typedef void CURL;
typedef void CURLM;

namespace ai_engine {

struct TTSConfig {

    std::string provider           = "elevenlabs";  
    std::string elevenlabs_api_key;
    std::string elevenlabs_voice_id;
    std::string elevenlabs_model_id = "eleven_turbo_v2_5";
    float       elevenlabs_stability        = 0.5f;
    float       elevenlabs_similarity_boost = 0.75f;
    float       elevenlabs_style            = 0.0f;
    bool        elevenlabs_use_speaker_boost = true;
    int         elevenlabs_output_sample_rate = 8000;  
    std::string elevenlabs_output_format;  
    std::string openai_api_key;
    std::string openai_voice  = "alloy";
    std::string openai_model  = "tts-1";
    float       openai_speed  = 1.0f;
    int    connect_timeout_ms  = 3000;
    int    request_timeout_ms  = 30000;
    bool   enable_cache        = true;
    size_t cache_max_entries   = 200;
    int    cache_ttl_seconds   = 3600;
};

using TTSAudioCallback = std::function<void(
    const int16_t* samples,
    size_t count,
    bool is_final,
    const std::string& sentence
)>;

using TTSErrorCallback = std::function<void(
    const std::string& error_message,
    int error_code
)>;

class ITTSEngine {
public:

    virtual ~ITTSEngine() = default;
    virtual bool synthesize(
        const std::string& text,
        TTSAudioCallback audio_cb,
        TTSErrorCallback error_cb,
        std::atomic<bool>& abort_flag
    ) = 0;
    virtual void configure(const TTSConfig& cfg) = 0;
    virtual const char* name() const = 0;
    virtual int output_sample_rate() const = 0;
};

class ElevenLabsTTS : public ITTSEngine {
public:
    ElevenLabsTTS();
    ~ElevenLabsTTS() override;

    bool synthesize(
        const std::string& text,
        TTSAudioCallback audio_cb,
        TTSErrorCallback error_cb,
        std::atomic<bool>& abort_flag
    ) override;

    void configure(const TTSConfig& cfg) override;

    const char* name() const override { return "ElevenLabs"; }
    int output_sample_rate() const override { return output_sr_; }

private:
    TTSConfig   cfg_;
    int         output_sr_ = 8000;
    std::string api_url_;

    void build_url();
    std::string build_request_body(const std::string& text) const;
    static std::string compute_output_format(int sample_rate);
    static int actual_api_sample_rate(const std::string& format_str);
    struct CurlWriteContext {
        TTSAudioCallback    audio_cb;
        TTSErrorCallback    error_cb;
        std::atomic<bool>*  abort_flag;
        std::string         sentence;
        int                 sample_rate;
        bool                had_error;
        int                 http_status;
        std::vector<uint8_t> pcm_buffer;
        size_t               pcm_buffer_pos;
        bool                 header_parsed;
    };

    static size_t curl_write_callback(char* ptr, size_t size, size_t nmemb, void* userdata);
    static size_t curl_header_callback(char* buffer, size_t size, size_t nitems, void* userdata);
    static int    curl_progress_callback(void* clientp, double dltotal, double dlnow,
                                          double ultotal, double ulnow);
};

class OpenAITTS : public ITTSEngine {
public:
    OpenAITTS();
    ~OpenAITTS() override;

    bool synthesize(
        const std::string& text,
        TTSAudioCallback audio_cb,
        TTSErrorCallback error_cb,
        std::atomic<bool>& abort_flag
    ) override;

    void configure(const TTSConfig& cfg) override;

    const char* name() const override { return "OpenAI-TTS"; }
    int output_sample_rate() const override { return 24000; }

private:
    TTSConfig   cfg_;
    std::string api_url_;

    std::string build_request_body(const std::string& text) const;

    struct CurlWriteContext {
        TTSAudioCallback    audio_cb;
        TTSErrorCallback    error_cb;
        std::atomic<bool>*  abort_flag;
        std::string         sentence;
        int                 sample_rate;
        bool                had_error;
        int                 http_status;
        std::vector<uint8_t> pcm_buffer;
        size_t               pcm_buffer_pos;
    };

    static size_t curl_write_callback(char* ptr, size_t size, size_t nmemb, void* userdata);
    static size_t curl_header_callback(char* buffer, size_t size, size_t nitems, void* userdata);
    static int    curl_progress_callback(void* clientp, double dltotal, double dlnow,
                                          double ultotal, double ulnow);
};

class FailoverTTS : public ITTSEngine {
public:
    FailoverTTS(std::unique_ptr<ITTSEngine> primary,
                std::unique_ptr<ITTSEngine> fallback);
    ~FailoverTTS() override;

    bool synthesize(
        const std::string& text,
        TTSAudioCallback audio_cb,
        TTSErrorCallback error_cb,
        std::atomic<bool>& abort_flag
    ) override;

    void configure(const TTSConfig& cfg) override;

    const char* name() const override { return "Failover"; }
    int output_sample_rate() const override;

private:
    std::unique_ptr<ITTSEngine> primary_;
    std::unique_ptr<ITTSEngine> fallback_;
    std::atomic<int>  consecutive_failures_{0};
    static constexpr int kFailoverThreshold = 3;
};

std::unique_ptr<ITTSEngine> create_tts_engine(const TTSConfig& cfg);

} 

#endif 
