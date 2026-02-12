#include "tts_cache.h"
#include <functional>
#include <algorithm>

namespace ai_engine {

TTSCache::TTSCache(const TTSCacheConfig& cfg)
    : cfg_(cfg)
{}


std::string TTSCache::make_key(const std::string& text) {
    
    std::string normalized;
    normalized.reserve(text.size());

    bool in_space = true;  
    for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!in_space && !normalized.empty()) {
                normalized += ' ';
                in_space = true;
            }
            continue;
        }
        in_space = false;
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
        normalized += c;
    }

    while (!normalized.empty() && normalized.back() == ' ') {
        normalized.pop_back();
    }

    return normalized;
}

TTSCache::CachedAudio TTSCache::get(const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key = make_key(text);
    auto it = map_.find(key);

    if (it == map_.end()) {
        misses_++;
        return CachedAudio{};
    }

    auto& entry = *(it->second);

    if (is_expired(entry)) {
        lru_list_.erase(it->second);
        map_.erase(it);
        misses_++;
        return CachedAudio{};
    }

    entry.last_accessed = Clock::now();
    lru_list_.splice(lru_list_.begin(), lru_list_, it->second);

    hits_++;

    CachedAudio result;
    result.samples = entry.samples;  
    result.sample_rate = entry.sample_rate;
    result.valid = true;
    return result;
}

void TTSCache::put(const std::string& text,
                   const std::vector<int16_t>& samples,
                   int sample_rate) {
    if (samples.empty() || cfg_.max_entries == 0) return;

    size_t audio_bytes = samples.size() * sizeof(int16_t);
    if (audio_bytes > cfg_.max_audio_bytes) return;

    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_key(text);

    auto it = map_.find(key);
    if (it != map_.end()) {
        auto& entry = *(it->second);
        entry.samples = samples;
        entry.sample_rate = sample_rate;
        entry.created_at = Clock::now();
        entry.last_accessed = Clock::now();
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        return;
    }

    evict_if_needed();

    CacheEntry entry;
    entry.key = key;
    entry.samples = samples;
    entry.sample_rate = sample_rate;
    entry.created_at = Clock::now();
    entry.last_accessed = Clock::now();

    lru_list_.push_front(std::move(entry));
    map_[key] = lru_list_.begin();
}

void TTSCache::put(const std::string& text,
                   const int16_t* samples, size_t count,
                   int sample_rate) {
    if (!samples || count == 0) return;
    std::vector<int16_t> vec(samples, samples + count);
    put(text, vec, sample_rate);
}

void TTSCache::invalidate(const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key = make_key(text);
    auto it = map_.find(key);
    if (it != map_.end()) {
        lru_list_.erase(it->second);
        map_.erase(it);
    }
}

void TTSCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    lru_list_.clear();
    map_.clear();
}

size_t TTSCache::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return map_.size();
}

double TTSCache::hit_rate() const {
    size_t total = hits_ + misses_;
    if (total == 0) return 0.0;
    return (double)hits_ / (double)total;
}

void TTSCache::evict_if_needed() {

    for (auto it = lru_list_.begin(); it != lru_list_.end(); ) {
        if (is_expired(*it)) {
            map_.erase(it->key);
            it = lru_list_.erase(it); 
        } else {
            ++it;
        }
    }

    while (lru_list_.size() >= cfg_.max_entries && !lru_list_.empty()) {
        auto& back = lru_list_.back();
        map_.erase(back.key);
        lru_list_.pop_back();
    }
}

bool TTSCache::is_expired(const CacheEntry& entry) const {
    if (cfg_.ttl_seconds <= 0) return false;

    auto now = Clock::now();
    auto age = std::chrono::duration_cast<std::chrono::seconds>(
        now - entry.created_at
    ).count();

    return age > cfg_.ttl_seconds;
}

}
