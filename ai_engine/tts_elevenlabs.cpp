#include "tts_engine.h"
#include <curl/curl.h>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <mutex>

namespace ai_engine {

static std::once_flag g_curl_init_flag;
static void ensure_curl_init() {
    std::call_once(g_curl_init_flag, []() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    });
}

ElevenLabsTTS::ElevenLabsTTS() {
    ensure_curl_init();
}

ElevenLabsTTS::~ElevenLabsTTS() {
}

void ElevenLabsTTS::configure(const TTSConfig& cfg) {
    cfg_ = cfg;

    if (cfg_.elevenlabs_output_format.empty()) {
        cfg_.elevenlabs_output_format = compute_output_format(cfg.elevenlabs_output_sample_rate);
    }

    output_sr_ = actual_api_sample_rate(cfg_.elevenlabs_output_format);

    build_url();
}

void ElevenLabsTTS::build_url() {
    std::ostringstream oss;
    oss << "https://api.elevenlabs.io/v1/text-to-speech/"
        << cfg_.elevenlabs_voice_id
        << "/stream?output_format="
        << cfg_.elevenlabs_output_format;
    api_url_ = oss.str();
}

std::string ElevenLabsTTS::compute_output_format(int sample_rate) {
    switch (sample_rate) {
        case 8000:  return "pcm_16000"; 
        case 16000: return "pcm_16000";
        case 22050: return "pcm_22050";
        case 24000: return "pcm_24000";
        case 44100: return "pcm_44100";
        default:    return "pcm_16000";
    }
}

int ElevenLabsTTS::actual_api_sample_rate(const std::string& format_str) {
    if (format_str.find("44100") != std::string::npos) return 44100;
    if (format_str.find("24000") != std::string::npos) return 24000;
    if (format_str.find("22050") != std::string::npos) return 22050;
    if (format_str.find("16000") != std::string::npos) return 16000;
    return 16000;  
}

std::string ElevenLabsTTS::build_request_body(const std::string& text) const {

    std::ostringstream oss;
    oss << "{\"text\":\"";

    for (char c : text) {
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

    oss << "\",\"model_id\":\"" << cfg_.elevenlabs_model_id << "\""
        << ",\"voice_settings\":{"
        << "\"stability\":" << cfg_.elevenlabs_stability
        << ",\"similarity_boost\":" << cfg_.elevenlabs_similarity_boost
        << ",\"style\":" << cfg_.elevenlabs_style
        << ",\"use_speaker_boost\":" << (cfg_.elevenlabs_use_speaker_boost ? "true" : "false")
        << "}}";

    return oss.str();
}

size_t ElevenLabsTTS::curl_write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* ctx = static_cast<CurlWriteContext*>(userdata);
    size_t total = size * nmemb;

    if (ctx->abort_flag && ctx->abort_flag->load(std::memory_order_relaxed)) {
        return 0;  
    }

    if (ctx->had_error) {
        return total;  
    }

    ctx->pcm_buffer.insert(ctx->pcm_buffer.end(), ptr, ptr + total);

    const size_t chunk_bytes = (size_t)ctx->sample_rate * 2 * 20 / 1000;
    const size_t min_chunk = std::max(chunk_bytes, (size_t)640);

    while (ctx->pcm_buffer.size() - ctx->pcm_buffer_pos >= min_chunk) {
        size_t avail = ctx->pcm_buffer.size() - ctx->pcm_buffer_pos;
        size_t emit_bytes = (avail / min_chunk) * min_chunk;
        if (emit_bytes > min_chunk * 4) emit_bytes = min_chunk * 4;

        const int16_t* samples = reinterpret_cast<const int16_t*>(
            ctx->pcm_buffer.data() + ctx->pcm_buffer_pos
        );
        size_t num_samples = emit_bytes / sizeof(int16_t);

        if (ctx->audio_cb) {
            ctx->audio_cb(samples, num_samples, false, ctx->sentence);
        }

        ctx->pcm_buffer_pos += emit_bytes;
    }

    if (ctx->pcm_buffer_pos > 8192) {
        ctx->pcm_buffer.erase(
            ctx->pcm_buffer.begin(),
            ctx->pcm_buffer.begin() + ctx->pcm_buffer_pos
        );
        ctx->pcm_buffer_pos = 0;
    }

    return total;
}

size_t ElevenLabsTTS::curl_header_callback(char* buffer, size_t size, size_t nitems, void* userdata) {
    auto* ctx = static_cast<CurlWriteContext*>(userdata);
    size_t total = size * nitems;

    std::string header(buffer, total);
    if (header.find("HTTP/") == 0) {
        size_t space1 = header.find(' ');
        if (space1 != std::string::npos) {
            ctx->http_status = atoi(header.c_str() + space1 + 1);
            if (ctx->http_status >= 400) {
                ctx->had_error = true;
            }
        }
    }

    return total;
}

int ElevenLabsTTS::curl_progress_callback(void* clientp, double, double, double, double) {
    auto* ctx = static_cast<CurlWriteContext*>(clientp);
    if (ctx->abort_flag && ctx->abort_flag->load(std::memory_order_relaxed)) {
        return 1; 
    }
    return 0;
}

bool ElevenLabsTTS::synthesize(
    const std::string& text,
    TTSAudioCallback audio_cb,
    TTSErrorCallback error_cb,
    std::atomic<bool>& abort_flag
) {
    if (text.empty()) return true;
    if (abort_flag.load(std::memory_order_relaxed)) return false;

    CURL* curl = curl_easy_init();
    if (!curl) {
        if (error_cb) error_cb("Failed to initialize libcurl", -1);
        return false;
    }

    std::string body = build_request_body(text);

    CurlWriteContext ctx;
    ctx.audio_cb       = audio_cb;
    ctx.error_cb       = error_cb;
    ctx.abort_flag     = &abort_flag;
    ctx.sentence       = text;
    ctx.sample_rate    = output_sr_;
    ctx.had_error      = false;
    ctx.http_status    = 0;
    ctx.pcm_buffer_pos = 0;
    ctx.header_parsed  = false;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: audio/pcm");

    std::string auth_header = "xi-api-key: " + cfg_.elevenlabs_api_key;
    headers = curl_slist_append(headers, auth_header.c_str());

    curl_easy_setopt(curl, CURLOPT_URL, api_url_.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, curl_header_callback);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, curl_progress_callback);
    curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, (long)cfg_.connect_timeout_ms);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, (long)cfg_.request_timeout_ms);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
    CURLcode res = curl_easy_perform(curl);

    bool success = false;

    if (res == CURLE_OK && !ctx.had_error) {
        size_t remaining = ctx.pcm_buffer.size() - ctx.pcm_buffer_pos;
        if (remaining >= 2 && audio_cb) {
            remaining = (remaining / 2) * 2;
            if (remaining > 0) {
                const int16_t* samples = reinterpret_cast<const int16_t*>(
                    ctx.pcm_buffer.data() + ctx.pcm_buffer_pos
                );
                audio_cb(samples, remaining / 2, true, text);
            }
        } else if (audio_cb) {
            audio_cb(nullptr, 0, true, text);
        }
        success = true;
    } else if (res == CURLE_ABORTED_BY_CALLBACK) {
        success = false;
    } else {
        std::string err_msg;
        if (ctx.had_error) {
            char buf[256];
            snprintf(buf, sizeof(buf), "ElevenLabs API error: HTTP %d", ctx.http_status);
            err_msg = buf;
        } else {
            err_msg = "ElevenLabs curl error: ";
            err_msg += curl_easy_strerror(res);
        }
        if (error_cb) error_cb(err_msg, (int)res);
        success = false;
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return success;
}

OpenAITTS::OpenAITTS() {}
OpenAITTS::~OpenAITTS() {}

void OpenAITTS::configure(const TTSConfig& cfg) {
    cfg_ = cfg;
    api_url_ = "https://api.openai.com/v1/audio/speech";
}

std::string OpenAITTS::build_request_body(const std::string& text) const {
    std::ostringstream oss;
    oss << "{\"model\":\"" << cfg_.openai_model << "\""
        << ",\"input\":\"";

    for (char c : text) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            case '\t': oss << "\\t";  break;
            default:   oss << c; break;
        }
    }

    oss << "\",\"voice\":\"" << cfg_.openai_voice << "\""
        << ",\"response_format\":\"pcm\""
        << ",\"speed\":" << cfg_.openai_speed
        << "}";

    return oss.str();
}

size_t OpenAITTS::curl_write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* ctx = static_cast<CurlWriteContext*>(userdata);
    size_t total = size * nmemb;

    if (ctx->abort_flag && ctx->abort_flag->load(std::memory_order_relaxed)) {
        return 0;
    }

    ctx->pcm_buffer.insert(ctx->pcm_buffer.end(), ptr, ptr + total);

    const size_t chunk_bytes = 960;

    size_t pos = 0;
    while (ctx->pcm_buffer.size() - pos >= chunk_bytes) {
        const int16_t* samples = reinterpret_cast<const int16_t*>(
            ctx->pcm_buffer.data() + pos
        );
        if (ctx->audio_cb) {
            ctx->audio_cb(samples, chunk_bytes / 2, false, ctx->sentence);
        }
        pos += chunk_bytes;
    }

    if (pos > 0) {
        ctx->pcm_buffer.erase(ctx->pcm_buffer.begin(), ctx->pcm_buffer.begin() + pos);
    }

    return total;
}

int OpenAITTS::curl_progress_callback(void* clientp, double, double, double, double) {
    auto* ctx = static_cast<CurlWriteContext*>(clientp);
    if (ctx->abort_flag && ctx->abort_flag->load(std::memory_order_relaxed)) {
        return 1;
    }
    return 0;
}

bool OpenAITTS::synthesize(
    const std::string& text,
    TTSAudioCallback audio_cb,
    TTSErrorCallback error_cb,
    std::atomic<bool>& abort_flag
) {
    if (text.empty()) return true;
    if (abort_flag.load(std::memory_order_relaxed)) return false;

    CURL* curl = curl_easy_init();
    if (!curl) {
        if (error_cb) error_cb("Failed to initialize libcurl", -1);
        return false;
    }

    std::string body = build_request_body(text);

    CurlWriteContext ctx;
    ctx.audio_cb   = audio_cb;
    ctx.error_cb   = error_cb;
    ctx.abort_flag  = &abort_flag;
    ctx.sentence   = text;
    ctx.sample_rate = 24000;
    ctx.had_error  = false;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth = "Authorization: Bearer " + cfg_.openai_api_key;
    headers = curl_slist_append(headers, auth.c_str());

    curl_easy_setopt(curl, CURLOPT_URL, api_url_.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, curl_progress_callback);
    curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, &ctx);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, (long)cfg_.connect_timeout_ms);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, (long)cfg_.request_timeout_ms);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);

    CURLcode res = curl_easy_perform(curl);

    bool success = false;
    if (res == CURLE_OK) {
        size_t rem = ctx.pcm_buffer.size();
        if (rem >= 2 && audio_cb) {
            rem = (rem / 2) * 2;
            audio_cb(reinterpret_cast<const int16_t*>(ctx.pcm_buffer.data()),
                     rem / 2, true, text);
        } else if (audio_cb) {
            audio_cb(nullptr, 0, true, text);
        }
        success = true;
    } else if (res != CURLE_ABORTED_BY_CALLBACK) {
        if (error_cb) {
            error_cb(std::string("OpenAI TTS error: ") + curl_easy_strerror(res), (int)res);
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return success;
}

FailoverTTS::FailoverTTS(
    std::unique_ptr<ITTSEngine> primary,
    std::unique_ptr<ITTSEngine> fallback
) : primary_(std::move(primary))
  , fallback_(std::move(fallback))
{}

FailoverTTS::~FailoverTTS() = default;

void FailoverTTS::configure(const TTSConfig& cfg) {
    if (primary_) primary_->configure(cfg);
    if (fallback_) fallback_->configure(cfg);
}

int FailoverTTS::output_sample_rate() const {
    int active_sr = last_active_sr_.load(std::memory_order_relaxed);
    if (active_sr > 0) return active_sr;
    if (consecutive_failures_.load(std::memory_order_relaxed) >= kFailoverThreshold && fallback_) {
        return fallback_->output_sample_rate();
    }
    return primary_ ? primary_->output_sample_rate() : 16000;
}

bool FailoverTTS::synthesize(
    const std::string& text,
    TTSAudioCallback audio_cb,
    TTSErrorCallback error_cb,
    std::atomic<bool>& abort_flag
) {
    int failures = consecutive_failures_.load(std::memory_order_relaxed);

    if (failures < kFailoverThreshold && primary_) {
        last_active_sr_.store(primary_->output_sample_rate(), std::memory_order_relaxed);
        bool ok = primary_->synthesize(text, audio_cb, error_cb, abort_flag);
        if (ok) {
            consecutive_failures_.store(0, std::memory_order_relaxed);
            return true;
        }
        consecutive_failures_.fetch_add(1, std::memory_order_relaxed);
    }

    if (fallback_) {
        last_active_sr_.store(fallback_->output_sample_rate(), std::memory_order_relaxed);
        return fallback_->synthesize(text, audio_cb, error_cb, abort_flag);
    }

    return false;
}

std::unique_ptr<ITTSEngine> create_tts_engine(const TTSConfig& cfg) {
    std::unique_ptr<ITTSEngine> primary;
    std::unique_ptr<ITTSEngine> fallback;

    if (cfg.provider == "elevenlabs") {
        primary = std::make_unique<ElevenLabsTTS>();
        primary->configure(cfg);

        if (!cfg.openai_api_key.empty()) {
            fallback = std::make_unique<OpenAITTS>();
            fallback->configure(cfg);
        }
    } else if (cfg.provider == "openai") {
        primary = std::make_unique<OpenAITTS>();
        primary->configure(cfg);
    } else {
        primary = std::make_unique<ElevenLabsTTS>();
        primary->configure(cfg);
    }

    if (fallback) {
        return std::make_unique<FailoverTTS>(std::move(primary), std::move(fallback));
    }

    return primary;
}

}
