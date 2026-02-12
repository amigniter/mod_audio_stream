#ifndef AI_ENGINE_TTS_CACHE_H
#define AI_ENGINE_TTS_CACHE_H

#include <string>
#include <vector>
#include <cstdint>
#include <mutex>
#include <list>
#include <unordered_map>
#include <chrono>
#include <functional>

namespace ai_engine {

struct TTSCacheConfig {
    size_t max_entries   = 200;
    int    ttl_seconds   = 3600;     
    size_t max_audio_bytes = 5 * 1024 * 1024;  
};

class TTSCache {
public:
    explicit TTSCache(const TTSCacheConfig& cfg = {});

    struct CachedAudio {
        std::vector<int16_t> samples;
        int                  sample_rate = 0;
        bool                 valid       = false;
    };

    CachedAudio get(const std::string& text);

    void put(const std::string& text,
             const std::vector<int16_t>& samples,
             int sample_rate);
    void put(const std::string& text,
             const int16_t* samples, size_t count,
             int sample_rate);
    void invalidate(const std::string& text);
    void clear();

    size_t size() const;
    size_t hits() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return hits_;
    }
    size_t misses() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return misses_;
    }
    double hit_rate() const;

private:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    struct CacheEntry {
        std::string          key;
        std::vector<int16_t> samples;
        int                  sample_rate;
        TimePoint            created_at;
        TimePoint            last_accessed;
    };

    using ListIterator = std::list<CacheEntry>::iterator;

    TTSCacheConfig                                   cfg_;
    mutable std::mutex                               mutex_;
    std::list<CacheEntry>                            lru_list_;   
    std::unordered_map<std::string, ListIterator>    map_;

    size_t  hits_   = 0;
    size_t  misses_ = 0;
    static std::string make_key(const std::string& text);
    void evict_if_needed();
    bool is_expired(const CacheEntry& entry) const;
};

} 

#endif 