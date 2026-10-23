#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define ZETA_HD __host__ __device__ __forceinline__
#else
#define ZETA_HD inline
#endif

namespace zeta256 {

// Signed-magnitude fixed point: 14 little-endian 32-bit limbs, with 320
// fractional bits and 128 integer bits (Q128.320). No float/double arithmetic.
struct Fx {
    int32_t sign; // -1, 0, +1
    uint32_t limb[14];
};

ZETA_HD Fx fxZero() {
    Fx r{};
    r.sign = 0;
    return r;
}

ZETA_HD bool fxIsZero(const Fx& a) {
    for (int i = 0; i < 14; ++i) if (a.limb[i] != 0) return false;
    return true;
}

ZETA_HD Fx fxNormalize(Fx a) {
    if (fxIsZero(a)) a.sign = 0;
    else if (a.sign == 0) a.sign = 1;
    return a;
}

ZETA_HD Fx fxFromUInt(uint64_t v) {
    Fx r{};
    if (v == 0) return r;
    r.sign = 1;
    r.limb[10] = static_cast<uint32_t>(v);
    r.limb[11] = static_cast<uint32_t>(v >> 32);
    return r;
}

ZETA_HD Fx fxFromSmallRatio(uint32_t numerator, uint32_t denominator) {
    // numerator/denominator in [0,1); binary long division into 320 fractional bits.
    Fx r{};
    if (denominator == 0 || numerator == 0) return r;
    uint64_t rem = numerator;
    for (int i = 0; i < 320; ++i) {
        rem <<= 1;
        if (rem >= denominator) {
            rem -= denominator;
            const int bit = 319 - i;
            r.limb[bit / 32] |= (uint32_t(1) << (bit % 32));
        }
    }
    r.sign = 1;
    return r;
}

ZETA_HD Fx fxFromPow2(int exponent) {
    // Exact 2^exponent, for exponents in [-320,127].
    Fx r{};
    const int bit = exponent + 320;
    if (bit < 0 || bit >= 448) return r;
    r.sign = 1;
    r.limb[bit / 32] = (uint32_t(1) << (bit % 32));
    return r;
}

ZETA_HD int fxCompareAbs(const Fx& a, const Fx& b) {
    for (int i = 13; i >= 0; --i) {
        if (a.limb[i] > b.limb[i]) return 1;
        if (a.limb[i] < b.limb[i]) return -1;
    }
    return 0;
}

ZETA_HD Fx fxNeg(Fx a) {
    if (!fxIsZero(a)) a.sign = -a.sign;
    else a.sign = 0;
    return a;
}

ZETA_HD Fx fxAbs(Fx a) {
    if (!fxIsZero(a)) a.sign = 1;
    else a.sign = 0;
    return a;
}

ZETA_HD Fx fxAddAbs(const Fx& a, const Fx& b, int sign) {
    Fx r{};
    uint64_t carry = 0;
    for (int i = 0; i < 14; ++i) {
        const uint64_t sum = static_cast<uint64_t>(a.limb[i]) + b.limb[i] + carry;
        r.limb[i] = static_cast<uint32_t>(sum);
        carry = sum >> 32;
    }
    r.sign = sign;
    return fxNormalize(r);
}

ZETA_HD Fx fxSubAbs(const Fx& a, const Fx& b, int sign) {
    // Requires |a| >= |b|.
    Fx r{};
    uint64_t borrow = 0;
    for (int i = 0; i < 14; ++i) {
        const uint64_t av = a.limb[i];
        const uint64_t bv = static_cast<uint64_t>(b.limb[i]) + borrow;
        r.limb[i] = static_cast<uint32_t>(av - bv);
        borrow = (av < bv) ? 1 : 0;
    }
    r.sign = sign;
    return fxNormalize(r);
}

ZETA_HD Fx fxAdd(const Fx& a, const Fx& b) {
    if (a.sign == 0 || fxIsZero(a)) return b;
    if (b.sign == 0 || fxIsZero(b)) return a;
    if (a.sign == b.sign) return fxAddAbs(a, b, a.sign);
    const int cmp = fxCompareAbs(a, b);
    if (cmp == 0) return fxZero();
    if (cmp > 0) return fxSubAbs(a, b, a.sign);
    return fxSubAbs(b, a, b.sign);
}

ZETA_HD Fx fxSub(const Fx& a, const Fx& b) {
    return fxAdd(a, fxNeg(b));
}

ZETA_HD Fx fxMulU32(const Fx& a, uint32_t m) {
    if (m == 0 || a.sign == 0) return fxZero();
    Fx r{};
    uint64_t carry = 0;
    for (int i = 0; i < 14; ++i) {
        const uint64_t cur = static_cast<uint64_t>(a.limb[i]) * m + carry;
        r.limb[i] = static_cast<uint32_t>(cur);
        carry = cur >> 32;
    }
    r.sign = a.sign;
    return fxNormalize(r);
}

ZETA_HD Fx fxMulU64(const Fx& a, uint64_t m) {
    const uint32_t lo = static_cast<uint32_t>(m);
    const uint32_t hi = static_cast<uint32_t>(m >> 32);
    Fx r = fxMulU32(a, lo);
    if (hi != 0) {
        Fx h = fxMulU32(a, hi);
        for (int i = 13; i >= 1; --i) h.limb[i] = h.limb[i - 1];
        h.limb[0] = 0;
        r = fxAdd(r, h);
    }
    return r;
}

ZETA_HD Fx fxDivU32(const Fx& a, uint32_t d) {
    if (d == 0 || a.sign == 0) return fxZero();
    Fx r{};
    uint64_t rem = 0;
    for (int i = 13; i >= 0; --i) {
        const uint64_t cur = (rem << 32) | a.limb[i];
        r.limb[i] = static_cast<uint32_t>(cur / d);
        rem = cur % d;
    }
    // Round to nearest, ties to even.
    const uint64_t twice = rem * 2;
    if (twice > d || (twice == d && (r.limb[0] & 1u))) {
        uint64_t carry = 1;
        for (int i = 0; i < 14 && carry; ++i) {
            const uint64_t cur = static_cast<uint64_t>(r.limb[i]) + carry;
            r.limb[i] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
        }
    }
    r.sign = a.sign;
    return fxNormalize(r);
}

ZETA_HD Fx fxMul(const Fx& a, const Fx& b) {
    if (a.sign == 0 || b.sign == 0) return fxZero();
    uint32_t product[28]{};
    for (int i = 0; i < 14; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 14; ++j) {
            const uint64_t cur = static_cast<uint64_t>(a.limb[i]) * b.limb[j]
                               + product[i + j] + carry;
            product[i + j] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
        }
        int k = i + 14;
        while (carry != 0 && k < 28) {
            const uint64_t cur = static_cast<uint64_t>(product[k]) + carry;
            product[k] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
            ++k;
        }
    }

    Fx r{};
    for (int i = 0; i < 14; ++i) r.limb[i] = product[i + 10];

    // Product was shifted right by 320 bits. Round the discarded bits to nearest-even.
    const bool guard = (product[9] & 0x80000000u) != 0;
    bool sticky = (product[9] & 0x7fffffffu) != 0;
    for (int i = 0; i < 9; ++i) sticky = sticky || (product[i] != 0);
    if (guard && (sticky || (r.limb[0] & 1u))) {
        uint64_t carry = 1;
        for (int i = 0; i < 14 && carry; ++i) {
            const uint64_t cur = static_cast<uint64_t>(r.limb[i]) + carry;
            r.limb[i] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
        }
    }
    r.sign = a.sign * b.sign;
    return fxNormalize(r);
}

ZETA_HD int fxFloorLog2(const Fx& a) {
    for (int i = 13; i >= 0; --i) {
        uint32_t v = a.limb[i];
        if (v != 0) {
            int bit = 31;
            while (((v >> bit) & 1u) == 0u) --bit;
            return i * 32 + bit - 320;
        }
    }
    return -10000;
}

ZETA_HD Fx fxInv(const Fx& x) {
    if (x.sign <= 0 || fxIsZero(x)) return fxZero();
    const int e = fxFloorLog2(x);
    Fx r = fxFromPow2(-e - 1); // x*r is initially in [1/2,1).
    const Fx two = fxFromUInt(2);
    for (int i = 0; i < 10; ++i) {
        const Fx xr = fxMul(x, r);
        r = fxMul(r, fxSub(two, xr));
    }
    return r;
}

ZETA_HD Fx fxInvSqrtUInt(uint32_t n) {
    if (n == 0) return fxZero();
    const Fx x = fxFromUInt(n);
    int e = 0;
    uint32_t v = n;
    while (v >>= 1) ++e;
    Fx y = fxFromPow2(-((e + 1) / 2));
    const Fx three = fxFromUInt(3);
    for (int i = 0; i < 9; ++i) {
        const Fx yy = fxMul(y, y);
        const Fx xyy = fxMul(x, yy);
        const Fx correction = fxDivU32(fxSub(three, xyy), 2);
        y = fxMul(y, correction);
    }
    return y;
}

ZETA_HD Fx fxLogUInt(uint32_t n, const Fx& ln2) {
    if (n == 0) return fxZero();
    int e = 0;
    uint32_t power = 1;
    while (power <= n / 2) { power <<= 1; ++e; }
    const uint32_t numerator = n - power;
    const uint32_t denominator = n + power;
    const Fx z = fxFromSmallRatio(numerator, denominator);
    Fx sum = fxZero();
    if (z.sign != 0) {
        const Fx z2 = fxMul(z, z);
        Fx term = z;
        for (uint32_t k = 0; k < 112; ++k) {
            const Fx piece = fxDivU32(term, 2 * k + 1);
            sum = fxAdd(sum, piece);
            term = fxMul(term, z2);
        }
        sum = fxMulU32(sum, 2);
    }
    return fxAdd(sum, fxMulU32(ln2, static_cast<uint32_t>(e)));
}

ZETA_HD uint64_t fxFloorU64(const Fx& a) {
    if (a.limb[12] != 0 || a.limb[13] != 0) return UINT64_MAX;
    return static_cast<uint64_t>(a.limb[10]) | (static_cast<uint64_t>(a.limb[11]) << 32);
}

ZETA_HD uint64_t fxCeilU64(const Fx& a) {
    uint64_t whole = fxFloorU64(a);
    if (whole == UINT64_MAX) return whole;
    bool fractional = false;
    for (int i = 0; i < 10; ++i) fractional = fractional || (a.limb[i] != 0);
    if (fractional && whole != UINT64_MAX) ++whole;
    return whole;
}

ZETA_HD bool fxLess(const Fx& a, const Fx& b) {
    return fxCompareAbs(a, b) < 0;
}

ZETA_HD Fx fxLoadRaw(const uint32_t words[14], int sign) {
    Fx r{};
    for (int i = 0; i < 14; ++i) r.limb[i] = words[i];
    r.sign = sign;
    return fxNormalize(r);
}

} // namespace zeta256

