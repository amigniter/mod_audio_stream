

#ifndef AI_ENGINE_RING_BUFFER_H
#define AI_ENGINE_RING_BUFFER_H

#include <atomic>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <new>         
#include <cassert>
#include <cstdlib>

#if defined(__APPLE__) || defined(__unix__)
#include <cstdlib> 
#endif

namespace ai_engine {


#if defined(__cpp_lib_hardware_interference_size) && __cpp_lib_hardware_interference_size >= 201703L
static constexpr size_t kCacheLineSize = std::hardware_destructive_interference_size;
#else
static constexpr size_t kCacheLineSize = 64;
#endif



static inline constexpr size_t next_power_of_2(size_t v) {
    if (v == 0) return 1;
    v--;
    v |= v >> 1;  v |= v >> 2;  v |= v >> 4;
    v |= v >> 8;  v |= v >> 16;
    if (sizeof(size_t) > 4) v |= v >> 32;
    return v + 1;
}

static inline constexpr bool is_power_of_2(size_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}



class SPSCRingBuffer {
public:


    explicit SPSCRingBuffer(size_t min_capacity_bytes)
        : capacity_(next_power_of_2(std::max(min_capacity_bytes, kCacheLineSize)))
        , mask_(capacity_ - 1)
    {
#if defined(_MSC_VER)
#if defined(_MSC_VER)
        buffer_ = static_cast<uint8_t*>(_aligned_malloc(capacity_, kCacheLineSize));
#else
        {
            void* p = nullptr;
            if (posix_memalign(&p, kCacheLineSize, capacity_) != 0) p = nullptr;
            buffer_ = static_cast<uint8_t*>(p);
        }
#endif
        if (!buffer_) {
            capacity_ = 0;
            mask_ = 0;
        } else {
            std::memset(buffer_, 0, capacity_);
        }
    }

    ~SPSCRingBuffer() {
        if (buffer_) {
#if defined(_MSC_VER)
            _aligned_free(buffer_);
#else
            std::free(buffer_);
#endif
            buffer_ = nullptr;
        }
    }


    SPSCRingBuffer(const SPSCRingBuffer&) = delete;
    SPSCRingBuffer& operator=(const SPSCRingBuffer&) = delete;

    SPSCRingBuffer(SPSCRingBuffer&& o) noexcept
        : buffer_(o.buffer_), capacity_(o.capacity_), mask_(o.mask_)
        , head_(o.head_.load(std::memory_order_relaxed))
        , tail_(o.tail_.load(std::memory_order_relaxed))
    {
        o.buffer_ = nullptr;
        o.capacity_ = 0;
        o.mask_ = 0;
    }



    size_t capacity()  const { return capacity_; }

    size_t size_approx() const {
        const size_t h = head_.load(std::memory_order_acquire);
        const size_t t = tail_.load(std::memory_order_acquire);
    return h - t;
    }

    size_t free_approx() const {
        return capacity_ - size_approx();
    }

    bool empty_approx() const {
        return head_.load(std::memory_order_acquire)
            == tail_.load(std::memory_order_acquire);
    }




    size_t try_write(const uint8_t* data, size_t len, size_t frame_align = 0) {
        if (!buffer_ || !data || len == 0) return 0;

        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t t = tail_.load(std::memory_order_acquire);

        const size_t avail = capacity_ - (h - t);
        size_t to_write = std::min(len, avail);

        if (frame_align > 0 && to_write < len) {
            to_write = (to_write / frame_align) * frame_align;
        }
        if (to_write == 0) return 0;


        const size_t pos = h & mask_;
        const size_t first = std::min(to_write, capacity_ - pos);
        std::memcpy(buffer_ + pos, data, first);
        if (first < to_write) {
            std::memcpy(buffer_, data + first, to_write - first);
        }


        head_.store(h + to_write, std::memory_order_release);
        return to_write;
    }


    size_t write_all(const uint8_t* data, size_t len) {
        if (!buffer_ || !data || len == 0) return 0;


        if (len > capacity_) {
            data += (len - capacity_);
            len = capacity_;
        }

        const size_t h = head_.load(std::memory_order_relaxed);
        const size_t t = tail_.load(std::memory_order_acquire);
        const size_t avail = capacity_ - (h - t);


        size_t to_write = std::min(len, avail);
        if (to_write == 0) return 0;

        const size_t pos = h & mask_;
        const size_t first = std::min(to_write, capacity_ - pos);
        std::memcpy(buffer_ + pos, data, first);
        if (first < to_write) {
            std::memcpy(buffer_, data + first, to_write - first);
        }

        head_.store(h + to_write, std::memory_order_release);
        return to_write;
    }




    size_t try_read(uint8_t* dest, size_t len) {
        if (!buffer_ || !dest || len == 0) return 0;

        const size_t t = tail_.load(std::memory_order_relaxed);
        const size_t h = head_.load(std::memory_order_acquire);

        const size_t avail = h - t;
        const size_t to_read = std::min(len, avail);


        const size_t pos = t & mask_;
        const size_t first = std::min(to_read, capacity_ - pos);
        std::memcpy(dest, buffer_ + pos, first);
        if (first < to_read) {
            std::memcpy(dest + first, buffer_, to_read - first);
        }

        tail_.store(t + to_read, std::memory_order_release);
        return to_read;
    }


    bool try_read_exact(uint8_t* dest, size_t len) {
        if (!buffer_ || !dest || len == 0) return false;

        const size_t t = tail_.load(std::memory_order_relaxed);
        const size_t h = head_.load(std::memory_order_acquire);

        if ((h - t) < len) return false;

        const size_t pos = t & mask_;
        const size_t first = std::min(len, capacity_ - pos);
        std::memcpy(dest, buffer_ + pos, first);
        if (first < len) {
            std::memcpy(dest + first, buffer_, len - first);
        }

        tail_.store(t + len, std::memory_order_release);
        return true;
    }


    size_t peek(uint8_t* dest, size_t len) const {
        if (!buffer_ || !dest || len == 0) return 0;

        const size_t t = tail_.load(std::memory_order_relaxed);
        const size_t h = head_.load(std::memory_order_acquire);

        const size_t avail = h - t;
        const size_t to_peek = std::min(len, avail);
        if (to_peek == 0) return 0;

        const size_t pos = t & mask_;
        const size_t first = std::min(to_peek, capacity_ - pos);
        std::memcpy(dest, buffer_ + pos, first);
        if (first < to_peek) {
            std::memcpy(dest + first, buffer_, to_peek - first);
        }

        return to_peek;
    }


    size_t skip(size_t len) {
        if (!buffer_ || len == 0) return 0;

        const size_t t = tail_.load(std::memory_order_relaxed);
        const size_t h = head_.load(std::memory_order_acquire);

        const size_t avail = h - t;
        const size_t to_skip = std::min(len, avail);
        if (to_skip == 0) return 0;

        tail_.store(t + to_skip, std::memory_order_release);
        return to_skip;
    }




    void flush() {
        const size_t h = head_.load(std::memory_order_acquire);
        tail_.store(h, std::memory_order_release);
    }


    void reset() {
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        if (buffer_) {
            std::memset(buffer_, 0, capacity_);
        }
    }




    size_t write_pcm16(const int16_t* samples, size_t num_samples) {
        return write_all(
            reinterpret_cast<const uint8_t*>(samples),
            num_samples * sizeof(int16_t)
        );
    }


    bool read_pcm16(int16_t* samples, size_t num_samples) {
        return try_read_exact(
            reinterpret_cast<uint8_t*>(samples),
            num_samples * sizeof(int16_t)
        );
    }


    size_t available_samples() const {
        return size_approx() / sizeof(int16_t);
    }


    double available_ms(int sample_rate) const {
        if (sample_rate <= 0) return 0.0;
        return (double)available_samples() * 1000.0 / (double)sample_rate;
    }

private:
    uint8_t*   buffer_   = nullptr;
    size_t     capacity_ = 0;
    size_t     mask_     = 0;


    alignas(kCacheLineSize) std::atomic<size_t> head_{0};
    alignas(kCacheLineSize) std::atomic<size_t> tail_{0};


    char pad_[kCacheLineSize - sizeof(std::atomic<size_t>)]{};
};

}

#endif
