#ifndef AI_ENGINE_DSP_PIPELINE_H
#define AI_ENGINE_DSP_PIPELINE_H

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ai_engine {

struct DSPConfig {
    int    sample_rate         = 8000;
    bool   dc_blocker_enabled  = true;
    float  dc_blocker_r        = 0.995f;
    bool   noise_gate_enabled  = true;
    float  noise_gate_threshold_db  = -45.0f;
    float  noise_gate_attack_ms     = 1.0f;
    float  noise_gate_release_ms    = 50.0f;
    bool   compressor_enabled       = true;
    float  compressor_threshold_db  = -18.0f;
    float  compressor_ratio         = 3.0f;
    float  compressor_attack_ms     = 5.0f;
    float  compressor_release_ms    = 50.0f;
    float  compressor_makeup_db     = 6.0f;
    float  compressor_knee_db       = 6.0f;
    bool   pre_emphasis_enabled     = false;
    float  pre_emphasis_coeff       = 0.97f;
    bool   high_shelf_enabled       = true;
    float  high_shelf_freq_hz       = 3000.0f;
    float  high_shelf_gain_db       = 3.0f;
    float  high_shelf_q             = 0.707f;
    bool   lpf_enabled              = true;
    float  lpf_cutoff_hz            = 3800.0f;
    float  lpf_q                    = 0.707f;
    bool   soft_clipper_enabled     = true;
    float  soft_clipper_threshold   = 0.85f;
    bool   de_esser_enabled         = false;
    float  de_esser_freq_hz         = 5500.0f;
    float  de_esser_threshold_db    = -20.0f;
    float  de_esser_ratio           = 4.0f;
};

class DCBlocker {
public:
    void init(float r = 0.995f) { r_ = r; x1_ = y1_ = 0.0f; }
    void process(int16_t* samples, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(samples[i]);
            float y = x - x1_ + r_ * y1_;
            x1_ = x;
            y1_ = y;
            samples[i] = clamp16(y);
        }
    }
private:
    float r_  = 0.995f;
    float x1_ = 0.0f;
    float y1_ = 0.0f;
    static inline int16_t clamp16(float v) {
        if (v > 32767.0f) return 32767;
        if (v < -32768.0f) return -32768;
        return static_cast<int16_t>(v);
    }
};

class NoiseGate {
public:
    void init(int sample_rate, float threshold_db,
              float attack_ms, float release_ms) {
        threshold_ = db_to_linear(threshold_db);
        attack_coeff_  = 1.0f - expf(-1.0f / (attack_ms * sample_rate / 1000.0f));
        release_coeff_ = 1.0f - expf(-1.0f / (release_ms * sample_rate / 1000.0f));
        envelope_ = 0.0f;
        gain_ = 1.0f;
    }
    void process(int16_t* samples, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            float x = std::fabs(static_cast<float>(samples[i]) / 32768.0f);
            if (x > envelope_)
                envelope_ += attack_coeff_ * (x - envelope_);
            else
                envelope_ += release_coeff_ * (x - envelope_);
            float target = (envelope_ > threshold_) ? 1.0f : 0.0f;
            gain_ += release_coeff_ * (target - gain_);
            float out = static_cast<float>(samples[i]) * gain_;
            samples[i] = clamp16(out);
        }
    }
private:
    float threshold_    = 0.0f;
    float attack_coeff_ = 0.0f;
    float release_coeff_= 0.0f;
    float envelope_     = 0.0f;
    float gain_         = 1.0f;
    static inline float db_to_linear(float db) {
        return powf(10.0f, db / 20.0f);
    }
    static inline int16_t clamp16(float v) {
        if (v > 32767.0f) return 32767;
        if (v < -32768.0f) return -32768;
        return static_cast<int16_t>(v);
    }
};

class DynamicCompressor {
public:
    void init(int sample_rate, float threshold_db, float ratio,
              float attack_ms, float release_ms,
              float makeup_db, float knee_db) {
        threshold_     = threshold_db;
        ratio_         = ratio;
        makeup_linear_ = powf(10.0f, makeup_db / 20.0f);
        knee_          = knee_db;
        half_knee_     = knee_db / 2.0f;
        attack_coeff_  = 1.0f - expf(-1.0f / (attack_ms * sample_rate / 1000.0f));
        release_coeff_ = 1.0f - expf(-1.0f / (release_ms * sample_rate / 1000.0f));
        env_db_ = -96.0f;
    }
    void process(int16_t* samples, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(samples[i]) / 32768.0f;
            float x_db = linear_to_db(std::fabs(x));
            if (x_db > env_db_)
                env_db_ += attack_coeff_ * (x_db - env_db_);
            else
                env_db_ += release_coeff_ * (x_db - env_db_);
            float gain_db = compute_gain(env_db_);
            float gain = db_to_linear(gain_db) * makeup_linear_;
            float out = x * gain;
            samples[i] = clamp16(out * 32768.0f);
        }
    }
private:
    float threshold_     = -18.0f;
    float ratio_         = 3.0f;
    float makeup_linear_ = 1.0f;
    float knee_          = 6.0f;
    float half_knee_     = 3.0f;
    float attack_coeff_  = 0.0f;
    float release_coeff_ = 0.0f;
    float env_db_        = -96.0f;
    inline float compute_gain(float input_db) const {
        float over = input_db - threshold_;
        if (over <= -half_knee_) {
            return 0.0f;
        } else if (over >= half_knee_) {
            return -(over * (1.0f - 1.0f / ratio_));
        } else {
            float x = over + half_knee_;
            return -(x * x / (2.0f * knee_)) * (1.0f - 1.0f / ratio_);
        }
    }
    static inline float linear_to_db(float v) {
        if (v < 1e-10f) return -96.0f;
        return 20.0f * log10f(v);
    }
    static inline float db_to_linear(float db) {
        return powf(10.0f, db / 20.0f);
    }
    static inline int16_t clamp16(float v) {
        if (v > 32767.0f) return 32767;
        if (v < -32768.0f) return -32768;
        return static_cast<int16_t>(v);
    }
};

class PreEmphasis {
public:
    void init(float coeff = 0.97f) { coeff_ = coeff; prev_ = 0.0f; }
    void process(int16_t* samples, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(samples[i]);
            float y = x - coeff_ * prev_;
            prev_ = x;
            samples[i] = clamp16(y);
        }
    }
private:
    float coeff_ = 0.97f;
    float prev_  = 0.0f;
    static inline int16_t clamp16(float v) {
        if (v > 32767.0f) return 32767;
        if (v < -32768.0f) return -32768;
        return static_cast<int16_t>(v);
    }
};

class BiquadFilter {
public:
    struct Coefficients {
        float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
        float a1 = 0.0f, a2 = 0.0f;
    };
    void set_coefficients(const Coefficients& c) { c_ = c; }
    void reset() { x1_ = x2_ = y1_ = y2_ = 0.0f; }
    void process(int16_t* samples, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(samples[i]) / 32768.0f;
            float y = c_.b0 * x + c_.b1 * x1_ + c_.b2 * x2_
                     - c_.a1 * y1_ - c_.a2 * y2_;
            x2_ = x1_; x1_ = x;
            y2_ = y1_; y1_ = y;
            samples[i] = clamp16(y * 32768.0f);
        }
    }
    static Coefficients make_high_shelf(int sample_rate, float freq_hz,
                                         float gain_db, float Q) {
        float A  = powf(10.0f, gain_db / 40.0f);
        float w0 = 2.0f * M_PI * freq_hz / sample_rate;
        float alpha = sinf(w0) / (2.0f * Q);
        float cos_w0 = cosf(w0);
        float sqA = sqrtf(A);
        float a0 =        (A + 1) - (A - 1) * cos_w0 + 2.0f * sqA * alpha;
        Coefficients c;
        c.b0 = (A * ((A + 1) + (A - 1) * cos_w0 + 2.0f * sqA * alpha)) / a0;
        c.b1 = (-2.0f * A * ((A - 1) + (A + 1) * cos_w0))              / a0;
        c.b2 = (A * ((A + 1) + (A - 1) * cos_w0 - 2.0f * sqA * alpha)) / a0;
        c.a1 = (2.0f * ((A - 1) - (A + 1) * cos_w0))                   / a0;
        c.a2 = ((A + 1) - (A - 1) * cos_w0 - 2.0f * sqA * alpha)       / a0;
        return c;
    }
    static Coefficients make_lowpass(int sample_rate, float freq_hz, float Q) {
        float w0 = 2.0f * M_PI * freq_hz / sample_rate;
        float alpha = sinf(w0) / (2.0f * Q);
        float cos_w0 = cosf(w0);
        float a0 = 1.0f + alpha;
        Coefficients c;
        c.b0 = ((1.0f - cos_w0) / 2.0f)  / a0;
        c.b1 = (1.0f - cos_w0)            / a0;
        c.b2 = ((1.0f - cos_w0) / 2.0f)  / a0;
        c.a1 = (-2.0f * cos_w0)           / a0;
        c.a2 = (1.0f - alpha)             / a0;
        return c;
    }
    static Coefficients make_highpass(int sample_rate, float freq_hz, float Q) {
        float w0 = 2.0f * M_PI * freq_hz / sample_rate;
        float alpha = sinf(w0) / (2.0f * Q);
        float cos_w0 = cosf(w0);
        float a0 = 1.0f + alpha;
        Coefficients c;
        c.b0 = ((1.0f + cos_w0) / 2.0f)  / a0;
        c.b1 = (-(1.0f + cos_w0))         / a0;
        c.b2 = ((1.0f + cos_w0) / 2.0f)  / a0;
        c.a1 = (-2.0f * cos_w0)           / a0;
        c.a2 = (1.0f - alpha)             / a0;
        return c;
    }
    static Coefficients make_bandpass(int sample_rate, float freq_hz, float Q) {
        float w0 = 2.0f * M_PI * freq_hz / sample_rate;
        float alpha = sinf(w0) / (2.0f * Q);
        float cos_w0 = cosf(w0);
        float a0 = 1.0f + alpha;
        Coefficients c;
        c.b0 = alpha               / a0;
        c.b1 = 0.0f;
        c.b2 = -alpha              / a0;
        c.a1 = (-2.0f * cos_w0)    / a0;
        c.a2 = (1.0f - alpha)      / a0;
        return c;
    }
private:
    Coefficients c_;
    float x1_ = 0.0f, x2_ = 0.0f;
    float y1_ = 0.0f, y2_ = 0.0f;
    static inline int16_t clamp16(float v) {
        if (v > 32767.0f) return 32767;
        if (v < -32768.0f) return -32768;
        return static_cast<int16_t>(v);
    }
};

class SoftClipper {
public:
    void init(float threshold = 0.85f) {
        threshold_ = threshold;
        if (threshold_ <= 0.0f) threshold_ = 0.01f;
        if (threshold_ > 1.0f) threshold_ = 1.0f;
    }
    void process(int16_t* samples, size_t n) {
        const float inv_thresh = 1.0f / threshold_;
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(samples[i]) / 32768.0f;
            float ax = std::fabs(x);
            if (ax <= threshold_) {
            } else if (ax < 1.0f) {
                float over = (ax - threshold_) * inv_thresh;
                float clipped = threshold_ + (1.0f - threshold_) *
                    (1.0f - (1.0f - over) * (1.0f - over) * (1.0f - over));
                if (clipped > 0.999f) clipped = 0.999f;
                x = (x > 0.0f) ? clipped : -clipped;
            } else {
                x = (x > 0.0f) ? 0.999f : -0.999f;
            }
            samples[i] = static_cast<int16_t>(x * 32768.0f);
        }
    }
private:
    float threshold_ = 0.85f;
};

class DeEsser {
public:
    void init(int sample_rate, float freq_hz, float threshold_db, float ratio) {
        detector_.set_coefficients(
            BiquadFilter::make_bandpass(sample_rate, freq_hz, 2.0f)
        );
        detector_.reset();
        threshold_ = powf(10.0f, threshold_db / 20.0f);
        ratio_ = ratio;
        envelope_ = 0.0f;
        release_coeff_ = 1.0f - expf(-1.0f / (10.0f * sample_rate / 1000.0f));
    }
    void process(int16_t* samples, size_t n) {
        std::vector<int16_t> det_copy(samples, samples + n);
        detector_.process(det_copy.data(), n);
        for (size_t i = 0; i < n; ++i) {
            float det_level = std::fabs(static_cast<float>(det_copy[i]) / 32768.0f);
            if (det_level > envelope_)
                envelope_ = det_level;
            else
                envelope_ += release_coeff_ * (det_level - envelope_);
            if (envelope_ > threshold_) {
                float over = envelope_ / threshold_;
                float gain = 1.0f / powf(over, 1.0f - 1.0f / ratio_);
                float x = static_cast<float>(samples[i]) * gain;
                samples[i] = clamp16(x);
            }
        }
    }
private:
    BiquadFilter detector_;
    float threshold_     = 0.0f;
    float ratio_         = 4.0f;
    float envelope_      = 0.0f;
    float release_coeff_ = 0.0f;
    static inline int16_t clamp16(float v) {
        if (v > 32767.0f) return 32767;
        if (v < -32768.0f) return -32768;
        return static_cast<int16_t>(v);
    }
};

class DSPPipeline {
public:
    DSPPipeline() = default;
    void init(const DSPConfig& cfg);
    void process(int16_t* samples, size_t num_samples);
    void reconfigure(const DSPConfig& cfg);
    const DSPConfig& config() const { return cfg_; }
private:
    DSPConfig        cfg_;
    DCBlocker        dc_blocker_;
    NoiseGate        noise_gate_;
    DynamicCompressor compressor_;
    PreEmphasis      pre_emphasis_;
    BiquadFilter     high_shelf_;
    BiquadFilter     lpf_;
    SoftClipper      soft_clipper_;
    DeEsser          de_esser_;
    bool             initialized_ = false;
};

}

#endif
