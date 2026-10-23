#include "fixed_format.hpp"
#include "zeta_math.hpp"
#include <iostream>
#include <vector>

using namespace zeta256;

static Cx zetaDirect(Fx b, const std::vector<Fx>& coefficients, const Fx& ln2,
                     const Fx& pi, const Fx& twoPi, const Fx& invTwoPi) {
    const uint64_t ceil2b = fxCeilU64(fxMulU32(b, 2));
    const uint32_t N = static_cast<uint32_t>(64 + ceil2b);
    Cx sum = cxZero();
    for (uint32_t n = 1; n < N; ++n) {
        const Fx ln = fxLogUInt(n, ln2);
        const Fx invSqrt = fxInvSqrtUInt(n);
        const Fx angle = fxMul(b, ln);
        sum = cxAdd(sum, cxScale(fxUnitPhaseNegative(angle, pi, twoPi, invTwoPi), invSqrt));
    }
    return fxEulerMaclaurinTail(sum, b, N, fxLogUInt(N, ln2), fxInvSqrtUInt(N),
                                pi, twoPi, invTwoPi, coefficients.data());
}

static bool sameRounded(const Fx& a, const Fx& b) {
    // Check the high 256 fractional bits plus integer part; allow the final 64
    // guard bits to differ when comparing recurrence against direct trig.
    if (a.sign != b.sign) return fxIsZero(a) && fxIsZero(b);
    for (int i = 2; i < 14; ++i) if (a.limb[i] != b.limb[i]) return false;
    return true;
}

int main() {
    const Fx ln2 = fxLn2Host();
    const Fx pi = fxPiHost();
    const Fx twoPi = fxTwoPiHost();
    const Fx invTwoPi = fxInvTwoPiHost();
    const std::vector<Fx> coeff = fxBernoulliHost();
    const Fx step = fxParseDecimal("0.25");
    constexpr uint32_t cap = 256;
    std::vector<Fx> logs(cap + 1), invs(cap + 1), rotRe(cap + 1), rotIm(cap + 1), phaseRe(cap + 1), phaseIm(cap + 1);
    for (uint32_t n = 1; n <= cap; ++n) {
        logs[n] = fxLogUInt(n, ln2);
        invs[n] = fxInvSqrtUInt(n);
        const Cx rot = fxUnitPhaseNegative(fxMul(step, logs[n]), pi, twoPi, invTwoPi);
        rotRe[n] = rot.re; rotIm[n] = rot.im;
    }
    Fx b = fxZero();
    uint32_t previousN = 1;
    for (int sampleIndex = 0; sampleIndex <= 5; ++sampleIndex) {
        const uint32_t N = static_cast<uint32_t>(64 + fxCeilU64(fxMulU32(b, 2)));
        for (uint32_t n = previousN; n < N; ++n) {
            const Cx phase = fxUnitPhaseNegative(fxMul(b, logs[n]), pi, twoPi, invTwoPi);
            phaseRe[n] = phase.re; phaseIm[n] = phase.im;
        }
        Cx sum = cxZero();
        for (uint32_t n = 1; n < N; ++n) {
            sum.re = fxAdd(sum.re, fxMul(phaseRe[n], invs[n]));
            sum.im = fxAdd(sum.im, fxMul(phaseIm[n], invs[n]));
            const Fx nextRe = fxSub(fxMul(phaseRe[n], rotRe[n]), fxMul(phaseIm[n], rotIm[n]));
            const Fx nextIm = fxAdd(fxMul(phaseRe[n], rotIm[n]), fxMul(phaseIm[n], rotRe[n]));
            phaseRe[n] = nextRe; phaseIm[n] = nextIm;
        }
        const Cx recurrence = fxEulerMaclaurinTail(sum, b, N, logs[N], invs[N], pi, twoPi, invTwoPi, coeff.data());
        const Cx direct = zetaDirect(b, coeff, ln2, pi, twoPi, invTwoPi);
        std::cout << "b=" << fxToString(b, 4)
                  << " recurrence Re=" << fxToString(recurrence.re, 36)
                  << " Im=" << fxToString(recurrence.im, 36)
                  << "\n  direct agrees (256-bit high portion): "
                  << (sameRounded(recurrence.re, direct.re) && sameRounded(recurrence.im, direct.im) ? "yes" : "no") << "\n";
        previousN = N;
        b = fxAdd(b, step);
    }
}
