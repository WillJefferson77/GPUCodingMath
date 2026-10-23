#pragma once
#include "fixed320.hpp"
#include "fp_constants.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace zeta256 {

inline Fx fxParseDecimal(const std::string& input) {
    if (input.empty()) throw std::runtime_error("empty numeric argument");
    std::size_t pos = 0;
    int sign = 1;
    if (input[pos] == '+' || input[pos] == '-') {
        if (input[pos] == '-') sign = -1;
        ++pos;
    }
    if (pos == input.size()) throw std::runtime_error("invalid numeric argument: " + input);

    uint64_t whole = 0;
    bool sawDigit = false;
    while (pos < input.size() && input[pos] != '.') {
        const unsigned char ch = static_cast<unsigned char>(input[pos]);
        if (!std::isdigit(ch)) throw std::runtime_error("invalid decimal number: " + input);
        sawDigit = true;
        const uint32_t digit = input[pos] - '0';
        if (whole > (std::numeric_limits<uint64_t>::max() - digit) / 10)
            throw std::runtime_error("integer part is too large: " + input);
        whole = whole * 10 + digit;
        ++pos;
    }
    Fx result = fxFromUInt(whole);

    if (pos < input.size() && input[pos] == '.') {
        ++pos;
        const std::size_t fractionStart = pos;
        while (pos < input.size()) {
            const unsigned char ch = static_cast<unsigned char>(input[pos]);
            if (!std::isdigit(ch)) throw std::runtime_error("invalid decimal number: " + input);
            sawDigit = true;
            ++pos;
        }
        Fx frac = fxZero();
        for (std::size_t i = pos; i > fractionStart; --i) {
            const uint32_t digit = input[i - 1] - '0';
            frac = fxDivU32(fxAdd(fxFromUInt(digit), frac), 10);
        }
        result = fxAdd(result, frac);
    }
    if (!sawDigit) throw std::runtime_error("invalid decimal number: " + input);
    if (fxIsZero(result)) return fxZero();
    result.sign = sign;
    return result;
}

inline std::string fxToString(const Fx& value, unsigned digits = 24) {
    if (value.sign == 0 || fxIsZero(value)) return "0";
    uint32_t integer[4] = { value.limb[10], value.limb[11], value.limb[12], value.limb[13] };
    bool integerZero = true;
    for (uint32_t word : integer) integerZero = integerZero && (word == 0);
    std::string integerText;
    if (integerZero) integerText = "0";
    else {
        while (!(integer[0] == 0 && integer[1] == 0 && integer[2] == 0 && integer[3] == 0)) {
            uint64_t rem = 0;
            for (int i = 3; i >= 0; --i) {
                const uint64_t cur = (rem << 32) | integer[i];
                integer[i] = static_cast<uint32_t>(cur / 10);
                rem = cur % 10;
            }
            integerText.push_back(static_cast<char>('0' + rem));
        }
        std::reverse(integerText.begin(), integerText.end());
    }

    uint32_t fraction[10];
    for (int i = 0; i < 10; ++i) fraction[i] = value.limb[i];
    std::string fractionalText;
    for (unsigned d = 0; d < digits; ++d) {
        uint64_t carry = 0;
        for (int i = 0; i < 10; ++i) {
            const uint64_t cur = static_cast<uint64_t>(fraction[i]) * 10 + carry;
            fraction[i] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
        }
        fractionalText.push_back(static_cast<char>('0' + carry));
        bool allZero = true;
        for (uint32_t word : fraction) allZero = allZero && (word == 0);
        if (allZero) break;
    }
    while (!fractionalText.empty() && fractionalText.back() == '0') fractionalText.pop_back();
    std::string out = (value.sign < 0 ? "-" : "") + integerText;
    if (!fractionalText.empty()) out += "." + fractionalText;
    return out;
}

inline int64_t fxToInt64Trunc(const Fx& value) {
    if (value.limb[12] != 0 || value.limb[13] != 0) {
        return value.sign < 0 ? INT64_MIN : INT64_MAX;
    }
    const uint64_t mag = static_cast<uint64_t>(value.limb[10])
                       | (static_cast<uint64_t>(value.limb[11]) << 32);
    if (value.sign < 0) {
        if (mag >= (uint64_t(1) << 63)) return INT64_MIN;
        return -static_cast<int64_t>(mag);
    }
    if (mag > static_cast<uint64_t>(INT64_MAX)) return INT64_MAX;
    return static_cast<int64_t>(mag);
}

inline Fx fxPiHost() { return fxLoadRaw(kPi_words, kPi_sign); }
inline Fx fxTwoPiHost() { return fxLoadRaw(kTwoPi_words, kTwoPi_sign); }
inline Fx fxInvTwoPiHost() { return fxLoadRaw(kInvTwoPi_words, kInvTwoPi_sign); }
inline Fx fxLn2Host() { return fxLoadRaw(kLn2_words, kLn2_sign); }

inline std::vector<Fx> fxBernoulliHost() {
    std::vector<Fx> values(EM_TERMS);
    for (int k = 0; k < EM_TERMS; ++k)
        values[k] = fxLoadRaw(kBernoulliWords[k], kBernoulliSigns[k]);
    return values;
}

} // namespace zeta256
