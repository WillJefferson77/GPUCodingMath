#pragma once

#include "fixed320.hpp"
#include "fp_constants.hpp"

namespace zeta256 {

struct Cx {
    Fx re;
    Fx im;
};

ZETA_HD Fx fxConst(int sign, const uint32_t words[FX_LIMBS]) {
    return fxLoadRaw(words, sign);
}


ZETA_HD Cx cxZero() { Cx z{}; z.re = fxZero(); z.im = fxZero(); return z; }
ZETA_HD Cx cxAdd(const Cx& a, const Cx& b) {
    Cx r; r.re = fxAdd(a.re, b.re); r.im = fxAdd(a.im, b.im); return r;
}
ZETA_HD Cx cxSub(const Cx& a, const Cx& b) {
    Cx r; r.re = fxSub(a.re, b.re); r.im = fxSub(a.im, b.im); return r;
}
ZETA_HD Cx cxMul(const Cx& a, const Cx& b) {
    Cx r;
    r.re = fxSub(fxMul(a.re, b.re), fxMul(a.im, b.im));
    r.im = fxAdd(fxMul(a.re, b.im), fxMul(a.im, b.re));
    return r;
}
ZETA_HD Cx cxScale(const Cx& a, const Fx& b) {
    Cx r; r.re = fxMul(a.re, b); r.im = fxMul(a.im, b); return r;
}

ZETA_HD Fx fxReduceAngle(Fx angle, const Fx& pi, const Fx& twoPi, const Fx& invTwoPi) {
    // Reduces a nonnegative angle into [-pi, pi], using a fixed-point reciprocal
    // for the quotient estimate and exact fixed-point comparisons for correction.
    if (angle.sign <= 0 || fxIsZero(angle)) return fxZero();
    const Fx estimate = fxMul(angle, invTwoPi);
    uint64_t q = fxFloorU64(estimate);
    Fx remainder = fxSub(angle, fxMulU64(twoPi, q));
    // Rounding of the reciprocal can move the estimate by at most a few units;
    // these loops restore the correct interval for practical plotted heights.
    int adjust = 0;
    while (remainder.sign < 0 && adjust++ < 4) {
        remainder = fxAdd(remainder, twoPi);
        if (q > 0) --q;
    }
    adjust = 0;
    while (fxCompareAbs(remainder, twoPi) >= 0 && adjust++ < 4) {
        remainder = fxSub(remainder, twoPi);
        ++q;
    }
    if (fxCompareAbs(remainder, pi) > 0) remainder = fxSub(remainder, twoPi);
    return remainder;
}

ZETA_HD void fxSinCos(Fx angle, const Fx& pi, const Fx& twoPi, const Fx& invTwoPi, Fx& cosine, Fx& sine) {
    // Taylor series in Q128.320 after reduction to [-pi,pi].
    if (angle.sign <= 0 || fxIsZero(angle)) {
        cosine = fxFromUInt(1);
        sine = fxZero();
        return;
    }
    Fx x = fxReduceAngle(angle, pi, twoPi, invTwoPi);
    const bool negative = x.sign < 0;
    x = fxAbs(x);
    if (fxIsZero(x)) {
        cosine = fxFromUInt(1);
        sine = fxZero();
        return;
    }
    const Fx x2 = fxMul(x, x);
    Fx sinTerm = x;
    Fx sinSum = x;
    Fx cosTerm = fxFromUInt(1);
    Fx cosSum = cosTerm;
    for (uint32_t k = 1; k <= 64; ++k) {
        sinTerm = fxNeg(fxDivU32(fxMul(sinTerm, x2), (2 * k) * (2 * k + 1)));
        sinSum = fxAdd(sinSum, sinTerm);
        cosTerm = fxNeg(fxDivU32(fxMul(cosTerm, x2), (2 * k - 1) * (2 * k)));
        cosSum = fxAdd(cosSum, cosTerm);
    }
    cosine = cosSum;
    sine = negative ? fxNeg(sinSum) : sinSum;
}

ZETA_HD Cx fxUnitPhaseNegative(Fx angle, const Fx& pi, const Fx& twoPi, const Fx& invTwoPi) {
    Fx c, s;
    fxSinCos(angle, pi, twoPi, invTwoPi, c, s);
    Cx phase; phase.re = c; phase.im = fxNeg(s); return phase;
}

ZETA_HD Fx fxLogUIntWithConst(uint32_t n, const Fx& ln2) {
    return fxLogUInt(n, ln2);
}

ZETA_HD Fx fxInvSqrtWithScale(uint32_t n) {
    return fxInvSqrtUInt(n);
}

ZETA_HD Cx fxEulerMaclaurinTail(Cx finiteSum, Fx b, uint32_t N,
                                const Fx& logN, const Fx& invSqrtN,
                                const Fx& pi, const Fx& twoPi, const Fx& invTwoPi,
                                const Fx* bernoulli) {
    const Fx half = fxFromPow2(-1);
    const Fx quarter = fxFromPow2(-2);

    // N^(-s) = N^(-1/2) exp(-i b log N).
    const Fx angle = fxMul(b, logN);
    Cx phaseN = fxUnitPhaseNegative(angle, pi, twoPi, invTwoPi);
    Cx nNegS = cxScale(phaseN, invSqrtN);

    // N^(1-s)/(s-1), with s-1=-1/2+ib.
    const Fx sqrtN = fxMulU32(invSqrtN, N);
    const Cx nOneMinusS = cxScale(phaseN, sqrtN);
    const Fx denominator = fxAdd(fxMul(b, b), quarter);
    const Fx invDenominator = fxInv(denominator);
    Fx boundaryRe = fxSub(fxNeg(fxMul(half, nOneMinusS.re)), fxZero());
    boundaryRe = fxAdd(boundaryRe, fxMul(b, nOneMinusS.im));
    Fx boundaryIm = fxSub(fxNeg(fxMul(b, nOneMinusS.re)), fxMul(half, nOneMinusS.im));
    boundaryRe = fxMul(boundaryRe, invDenominator);
    boundaryIm = fxMul(boundaryIm, invDenominator);
    finiteSum.re = fxAdd(finiteSum.re, boundaryRe);
    finiteSum.im = fxAdd(finiteSum.im, boundaryIm);

    // Add (1/2)N^(-s).
    finiteSum.re = fxAdd(finiteSum.re, fxMul(half, nNegS.re));
    finiteSum.im = fxAdd(finiteSum.im, fxMul(half, nNegS.im));

    // Bernoulli corrections: c_k (s)_(2k-1) N^(-s-2k+1).
    // Keep the rising factorial divided by N at each step to avoid large intermediates.
    Cx rising;
    rising.re = fxDivU32(half, N);
    rising.im = fxDivU32(b, N);
    const Fx bOverN = fxDivU32(b, N);
    for (uint32_t k = 1; k <= EM_TERMS; ++k) {
        Cx correction = cxMul(nNegS, rising);
        correction = cxScale(correction, bernoulli[k - 1]);
        finiteSum = cxAdd(finiteSum, correction);
        if (k < EM_TERMS) {
            Fx firstReal = fxDivU32(fxAdd(fxFromUInt(2 * k - 1), half), N);
            Fx secondReal = fxDivU32(fxAdd(fxFromUInt(2 * k), half), N);
            Cx first; first.re = firstReal; first.im = bOverN;
            Cx second; second.re = secondReal; second.im = bOverN;
            rising = cxMul(cxMul(rising, first), second);
        }
    }
    return finiteSum;
}

} // namespace zeta256
