/*
 * mod_audio_stream
 * Copyright (C) 2026 AMSOFTSWITCH LTD
 *
 * Licensed under the GNU Affero General Public License v3.0 only.
 * See LICENSE and LICENSE_EXCEPTION for licensing terms.
 */

#include "base64.h"

#include <cstdint>
#include <stdexcept>

namespace {

int decode_value(unsigned char c)
{
    if (c >= 'A' && c <= 'Z') {
        return static_cast<int>(c - 'A');
    }

    if (c >= 'a' && c <= 'z') {
        return static_cast<int>(c - 'a') + 26;
    }

    if (c >= '0' && c <= '9') {
        return static_cast<int>(c - '0') + 52;
    }

    // Standard and URL-safe variants.
    if (c == '+' || c == '-') {
        return 62;
    }

    if (c == '/' || c == '_') {
        return 63;
    }

    return -1;
}

bool is_padding(unsigned char c)
{
    return c == '=' || c == '.';
}

} // namespace

std::string base64_decode(const std::string& input)
{
    if (input.empty()) {
        return {};
    }

    // A Base64 stream can never have a final group containing
    // only one encoded character.
    if (input.size() % 4 == 1) {
        throw std::runtime_error("Invalid Base64 length");
    }

    std::string output;
    output.reserve((input.size() * 3) / 4 + 3);

    std::size_t pos = 0;

    while (pos < input.size()) {
        const std::size_t remaining = input.size() - pos;

        const unsigned char c0 =
            static_cast<unsigned char>(input[pos]);

        const unsigned char c1 =
            static_cast<unsigned char>(input[pos + 1]);

        if (is_padding(c0) || is_padding(c1)) {
            throw std::runtime_error("Invalid Base64 padding");
        }

        const int v0 = decode_value(c0);
        const int v1 = decode_value(c1);

        if (v0 < 0 || v1 < 0) {
            throw std::runtime_error("Invalid Base64 character");
        }

        output.push_back(static_cast<char>(
            (v0 << 2) | (v1 >> 4)));

        /*
         * Two-character final group:
         *
         *   xx
         *
         * Equivalent to xx==.
         */
        if (remaining == 2) {
            return output;
        }

        const unsigned char c2 =
            static_cast<unsigned char>(input[pos + 2]);

        /*
         * Padded one-byte final group:
         *
         *   xx==
         *   xx..
         */
        if (is_padding(c2)) {
            if (remaining != 4 ||
                !is_padding(
                    static_cast<unsigned char>(input[pos + 3]))) {
                throw std::runtime_error("Invalid Base64 padding");
            }

            return output;
        }

        const int v2 = decode_value(c2);

        if (v2 < 0) {
            throw std::runtime_error("Invalid Base64 character");
        }

        output.push_back(static_cast<char>(
            ((v1 & 0x0f) << 4) | (v2 >> 2)));

        /*
         * Three-character final group:
         *
         *   xxx
         *
         * Equivalent to xxx=.
         */
        if (remaining == 3) {
            return output;
        }

        const unsigned char c3 =
            static_cast<unsigned char>(input[pos + 3]);

        /*
         * Padded two-byte final group:
         *
         *   xxx=
         *   xxx.
         */
        if (is_padding(c3)) {
            if (remaining != 4) {
                throw std::runtime_error("Invalid Base64 padding");
            }

            return output;
        }

        const int v3 = decode_value(c3);

        if (v3 < 0) {
            throw std::runtime_error("Invalid Base64 character");
        }

        output.push_back(static_cast<char>(
            ((v2 & 0x03) << 6) | v3));

        pos += 4;
    }

    return output;
}
