#ifndef AI_ENGINE_SENTENCE_BUFFER_H
#define AI_ENGINE_SENTENCE_BUFFER_H

#include <string>
#include <vector>
#include <functional>
#include <cstddef>

namespace ai_engine {

struct SentenceBufferConfig {
    size_t min_sentence_chars  = 10;    
    size_t max_sentence_chars  = 250;
    size_t flush_after_ms      = 500;
    bool   split_on_comma      = false;
    bool   split_on_semicolon  = true;
    bool   split_on_colon      = true;
    bool   split_on_newline    = true;
};

using SentenceCallback = std::function<void(const std::string& sentence, bool is_final)>;

class SentenceBuffer {
public:
    explicit SentenceBuffer(const SentenceBufferConfig& cfg = {});


        void add_token(const std::string& token, SentenceCallback cb);


        void flush(SentenceCallback cb);


        void reset();

    
    const std::string& current_buffer() const { return buffer_; }
    size_t total_sentences_emitted() const { return sentences_emitted_; }
    bool   is_empty() const { return buffer_.empty(); }

private:
    SentenceBufferConfig cfg_;
    std::string          buffer_;
    size_t               sentences_emitted_ = 0;

    
    bool is_sentence_boundary(size_t pos) const;

    
    bool is_abbreviation(size_t pos) const;

    
    void emit(size_t end_pos, SentenceCallback& cb, bool is_final);

    
    static bool is_known_abbreviation(const std::string& word);
};

}

#endif
