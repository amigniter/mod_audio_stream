#include "sentence_buffer.h"
#include <algorithm>
#include <cctype>

namespace ai_engine {

static const char* const kAbbreviations[] = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "st", "ave", "blvd", "rd", "apt", "dept",
    "vs", "etc", "inc", "ltd", "co", "corp",
    "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "oct", "nov", "dec",
    "no", "nos", "vol", "vols",
    "approx", "est", "govt", "assn",
    "a.m", "p.m", "e.g", "i.e",
    nullptr
};

bool SentenceBuffer::is_known_abbreviation(const std::string& word) {
    if (word.empty()) return false;
    std::string lower = word;
    for (auto& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (const char* const* p = kAbbreviations; *p; ++p) {
        if (lower == *p) return true;
    }
    if (lower.size() == 1 && std::isalpha(static_cast<unsigned char>(lower[0]))) {
        return true;
    }
    return false;
}

SentenceBuffer::SentenceBuffer(const SentenceBufferConfig& cfg)
    : cfg_(cfg)
{
    buffer_.reserve(512);
}

void SentenceBuffer::add_token(const std::string& token, SentenceCallback cb) {
    if (token.empty()) return;
    buffer_ += token;
    while (true) {
        if (buffer_.size() >= cfg_.max_sentence_chars) {
            size_t split = std::string::npos;
            for (size_t i = buffer_.size(); i > cfg_.min_sentence_chars; --i) {
                if (is_sentence_boundary(i - 1)) {
                    split = i;
                    break;
                }
            }
            if (split == std::string::npos) {
                split = buffer_.rfind(' ', cfg_.max_sentence_chars);
                if (split == std::string::npos || split < cfg_.min_sentence_chars) {
                    split = cfg_.max_sentence_chars;
                }
            }
            emit(split, cb, false);
            continue;
        }
        bool found = false;
        for (size_t i = cfg_.min_sentence_chars; i < buffer_.size(); ++i) {
            if (is_sentence_boundary(i)) {
                emit(i + 1, cb, false);
                found = true;
                break;
            }
        }
        if (!found) break;
    }
}

void SentenceBuffer::flush(SentenceCallback cb) {
    if (buffer_.empty()) return;
    size_t start = 0;
    while (start < buffer_.size() &&
           std::isspace(static_cast<unsigned char>(buffer_[start]))) {
        ++start;
    }
    if (start < buffer_.size()) {
        std::string sentence = buffer_.substr(start);
        while (!sentence.empty() &&
               std::isspace(static_cast<unsigned char>(sentence.back()))) {
            sentence.pop_back();
        }
        if (!sentence.empty()) {
            sentences_emitted_++;
            cb(sentence, true);
        }
    }
    buffer_.clear();
}

void SentenceBuffer::reset() {
    buffer_.clear();
    sentences_emitted_ = 0;
}

bool SentenceBuffer::is_sentence_boundary(size_t pos) const {
    if (pos >= buffer_.size()) return false;
    char c = buffer_[pos];
    if (cfg_.split_on_newline && c == '\n') {
        return pos >= cfg_.min_sentence_chars;
    }
    if (c == '.' || c == '?' || c == '!') {
        bool followed_by_boundary = false;
        if (pos + 1 >= buffer_.size()) {
            return false;
        }
        char next = buffer_[pos + 1];
        followed_by_boundary = std::isspace(static_cast<unsigned char>(next))
                            || next == '"'
                            || next == '\''
                            || next == ')';
        if (!followed_by_boundary) return false;
        if (c == '.' && is_abbreviation(pos)) {
            return false;
        }
        if (c == '.' && pos >= 2 && buffer_[pos-1] == '.' && buffer_[pos-2] == '.') {
            if (pos + 2 < buffer_.size() &&
                std::isupper(static_cast<unsigned char>(buffer_[pos + 2]))) {
                return true;
            }
            return false;
        }
        for (size_t j = pos + 1; j < buffer_.size() && j < pos + 4; ++j) {
            if (std::isspace(static_cast<unsigned char>(buffer_[j]))) continue;
            if (std::isupper(static_cast<unsigned char>(buffer_[j]))) return true;
            if (c == '?' || c == '!') return true;
            return false;
        }
        return (c == '?' || c == '!');
    }
    if (cfg_.split_on_semicolon && c == ';') {
        return pos >= cfg_.min_sentence_chars &&
               pos + 1 < buffer_.size() &&
               buffer_[pos + 1] == ' ';
    }
    if (cfg_.split_on_colon && c == ':') {
        return pos >= cfg_.min_sentence_chars &&
               pos + 1 < buffer_.size() &&
               buffer_[pos + 1] == ' ';
    }
    if (cfg_.split_on_comma && c == ',') {
        return pos >= std::max(cfg_.min_sentence_chars, (size_t)40) &&
               pos + 1 < buffer_.size() &&
               buffer_[pos + 1] == ' ';
    }
    return false;
}

bool SentenceBuffer::is_abbreviation(size_t pos) const {
    if (pos == 0) return false;
    if (buffer_[pos] != '.') return false;
    size_t end = pos;
    size_t start = pos;
    while (start > 0 && std::isalpha(static_cast<unsigned char>(buffer_[start - 1]))) {
        --start;
    }
    if (start > 0 && start == end) {
        if (start >= 2 && buffer_[start - 1] == '.') {
            return true;
        }
    }
    if (start == end) return false;
    std::string word = buffer_.substr(start, end - start);
    return is_known_abbreviation(word);
}

void SentenceBuffer::emit(size_t end_pos, SentenceCallback& cb, bool is_final) {
    if (end_pos == 0 || end_pos > buffer_.size()) return;
    std::string sentence = buffer_.substr(0, end_pos);
    size_t trim_start = 0;
    while (trim_start < sentence.size() &&
           std::isspace(static_cast<unsigned char>(sentence[trim_start]))) {
        ++trim_start;
    }
    if (trim_start > 0) {
        sentence = sentence.substr(trim_start);
    }
    while (!sentence.empty() &&
           std::isspace(static_cast<unsigned char>(sentence.back()))) {
        sentence.pop_back();
    }
    buffer_.erase(0, end_pos);
    if (!sentence.empty()) {
        sentences_emitted_++;
        cb(sentence, is_final);
    }
}

}
