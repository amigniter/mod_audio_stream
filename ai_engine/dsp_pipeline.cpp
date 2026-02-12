#include "dsp_pipeline.h"

namespace ai_engine {

void DSPPipeline::init(const DSPConfig& cfg) {
    cfg_ = cfg;

    dc_blocker_.init(cfg_.dc_blocker_r);

    if (cfg_.noise_gate_enabled) {
        noise_gate_.init(cfg_.sample_rate,
                         cfg_.noise_gate_threshold_db,
                         cfg_.noise_gate_attack_ms,
                         cfg_.noise_gate_release_ms);
    }

    if (cfg_.compressor_enabled) {
        compressor_.init(cfg_.sample_rate,
                         cfg_.compressor_threshold_db,
                         cfg_.compressor_ratio,
                         cfg_.compressor_attack_ms,
                         cfg_.compressor_release_ms,
                         cfg_.compressor_makeup_db,
                         cfg_.compressor_knee_db);
    }

    if (cfg_.pre_emphasis_enabled) {
        pre_emphasis_.init(cfg_.pre_emphasis_coeff);
    }

    if (cfg_.high_shelf_enabled) {
        auto coeff = BiquadFilter::make_high_shelf(
            cfg_.sample_rate,
            cfg_.high_shelf_freq_hz,
            cfg_.high_shelf_gain_db,
            cfg_.high_shelf_q
        );
        high_shelf_.set_coefficients(coeff);
        high_shelf_.reset();
    }

    if (cfg_.lpf_enabled) {
        auto coeff = BiquadFilter::make_lowpass(
            cfg_.sample_rate,
            cfg_.lpf_cutoff_hz,
            cfg_.lpf_q
        );
        lpf_.set_coefficients(coeff);
        lpf_.reset();
    }

    if (cfg_.soft_clipper_enabled) {
        soft_clipper_.init(cfg_.soft_clipper_threshold);
    }

    if (cfg_.de_esser_enabled) {
        de_esser_.init(cfg_.sample_rate,
                       cfg_.de_esser_freq_hz,
                       cfg_.de_esser_threshold_db,
                       cfg_.de_esser_ratio);
    }

    initialized_ = true;
}

void DSPPipeline::process(int16_t* samples, size_t num_samples) {
    if (!initialized_ || !samples || num_samples == 0) return;

    {
        bool all_silent = true;
        for (size_t i = 0; i < num_samples; ++i) {
            if (samples[i] != 0) { all_silent = false; break; }
        }
        if (all_silent) return;
    }

    if (cfg_.dc_blocker_enabled)     dc_blocker_.process(samples, num_samples);
    if (cfg_.noise_gate_enabled)     noise_gate_.process(samples, num_samples);
    if (cfg_.compressor_enabled)     compressor_.process(samples, num_samples);
    if (cfg_.pre_emphasis_enabled)   pre_emphasis_.process(samples, num_samples);
    if (cfg_.high_shelf_enabled)     high_shelf_.process(samples, num_samples);
    if (cfg_.lpf_enabled)            lpf_.process(samples, num_samples);
    if (cfg_.de_esser_enabled)       de_esser_.process(samples, num_samples);
    if (cfg_.soft_clipper_enabled)   soft_clipper_.process(samples, num_samples);
}

void DSPPipeline::reconfigure(const DSPConfig& cfg) {
    init(cfg);
}

} 
