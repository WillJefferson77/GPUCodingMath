#include "fixed_format.hpp"
#include "zeta_math.hpp"
#include <iostream>
#include <vector>

using namespace zeta256;

static Cx zetaDirect(Fx b) {
    const Fx ln2 = fxLn2Host();
    const Fx pi = fxPiHost();
    const Fx twoPi = fxTwoPiHost();
    const Fx invTwoPi = fxInvTwoPiHost();
    const std::vector<Fx> coefficients = fxBernoulliHost();
    const uint64_t ceil2b = fxCeilU64(fxMulU32(b, 2));
    const uint32_t N = static_cast<uint32_t>(64 + ceil2b);
    Cx sum = cxZero();
    for (uint32_t n = 1; n < N; ++n) {
        const Fx ln = fxLogUInt(n, ln2);
        const Fx invSqrt = fxInvSqrtUInt(n);
        const Fx angle = fxMul(b, ln);
        const Cx phase = fxUnitPhaseNegative(angle, pi, twoPi, invTwoPi);
        sum = cxAdd(sum, cxScale(phase, invSqrt));
    }
    const Fx logN = fxLogUInt(N, ln2);
    const Fx invSqrtN = fxInvSqrtUInt(N);
    return fxEulerMaclaurinTail(sum, b, N, logN, invSqrtN, pi, twoPi, invTwoPi, coefficients.data());
}

int main(int argc, char** argv) {
    const std::string bText = argc > 1 ? argv[1] : "0";
    Fx b = fxParseDecimal(bText);
    const Cx out = zetaDirect(b);
    std::cout << "b=" << fxToString(b, 16) << "\n";
    std::cout << "re=" << fxToString(out.re, 90) << "\n";
    std::cout << "im=" << fxToString(out.im, 90) << "\n";
}
