#pragma once

#include "fixed_format.hpp"
#include "zeta_cuda.hpp"

#include <cstdint>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>

namespace zeta256 {

// The threshold is applied to |zeta|, using |zeta|^2 = re^2 + im^2 so the
// decision stays entirely in the project's custom fixed-point arithmetic.
struct ZeroSample {
    Fx b{};
    Fx re{};
    Fx im{};
    Fx normSquared{};
};

inline Fx zetaNormSquared(const Fx& re, const Fx& im) {
    return fxAdd(fxMul(re, re), fxMul(im, im));
}

inline bool fxCrossesOrTouchesZero(const Fx& left, const Fx& right) {
    if (fxIsZero(left) || fxIsZero(right)) return true;
    return left.sign != right.sign;
}

inline uint32_t zetaEndpointN(const Fx& b) {
    const Fx twoB = fxMulU32(b, 2);
    const uint64_t ceilTwoB = fxCeilU64(twoB);
    if (ceilTwoB == UINT64_MAX ||
        ceilTwoB > static_cast<uint64_t>(UINT32_MAX) - 65ull) {
        throw std::runtime_error(
            "b exceeded the 32-bit Euler-Maclaurin index range during zero search");
    }
    return static_cast<uint32_t>(64ull + ceilTwoB);
}

class ZeroCandidateLogger {
public:
    ZeroCandidateLogger(const std::string& path, const Fx& threshold, const Fx& sampleStep)
        : path_(path), threshold_(threshold), thresholdSquared_(fxMul(threshold, threshold)),
          duplicateSpan_(fxMulU32(sampleStep, 2)), out_(path.c_str(), std::ios::out | std::ios::app) {
        if (!out_) throw std::runtime_error("Cannot open zero-candidate log: " + path_);
        out_.seekp(0, std::ios::end);
        if (out_.tellp() == std::streampos(0)) {
            out_ << "b,real,imag,norm_squared,magnitude_threshold,method\n";
            out_.flush();
        }
    }

    bool isBelowThreshold(const Fx& normSquared) const {
        return fxCompareAbs(normSquared, thresholdSquared_) < 0;
    }

    bool logIfBelow(const ZeroSample& sample, const char* method) {
        if (!isBelowThreshold(sample.normSquared)) return false;

        if (hasLastLogged_) {
            const Fx separation = fxAbs(fxSub(sample.b, lastLoggedB_));
            if (fxCompareAbs(separation, duplicateSpan_) <= 0) return false;
        }

        // The exact test is norm_squared < threshold^2, equivalent to
        // |zeta| < threshold. Log norm_squared directly so no low-precision
        // floating-point square root is introduced just for display.
        out_ << fxToString(sample.b, 36) << ','
             << fxToString(sample.re, 36) << ','
             << fxToString(sample.im, 36) << ','
             << fxToString(sample.normSquared, 42) << ','
             << fxToString(threshold_, 18) << ',' << method << '\n';
        out_.flush();
        if (!out_) throw std::runtime_error("Writing zero-candidate log failed: " + path_);

        lastLoggedB_ = sample.b;
        hasLastLogged_ = true;
        return true;
    }

    const std::string& path() const { return path_; }

private:
    std::string path_;
    Fx threshold_{};
    Fx thresholdSquared_{};
    Fx duplicateSpan_{};
    std::ofstream out_;
    Fx lastLoggedB_{};
    bool hasLastLogged_ = false;
};

inline ZeroSample evaluateZetaFresh(CudaZetaEngine& engine, const Fx& b,
                                    const std::function<void()>& checkpoint) {
    const uint32_t N = zetaEndpointN(b);
    Fx re, im;
    // Force a complete phase initialization because this may be an arbitrary
    // point inside a bracket rather than the next sequential sample.
    engine.evaluate(b, N, 1, true, re, im, checkpoint);
    return ZeroSample{b, re, im, zetaNormSquared(re, im)};
}

// When both components change sign over the same sample interval, minimize
// |zeta|^2 inside that interval. Only a result below the requested threshold
// is written as a zero candidate. The narrow-bracket search is not a proof of
// a mathematically exact zero; it is a practical detector for the viewer.
inline ZeroSample refineZeroCandidate(CudaZetaEngine& engine,
                                      const ZeroSample& leftSample,
                                      const ZeroSample& rightSample,
                                      const std::function<void()>& checkpoint) {
    Fx best = (fxCompareAbs(leftSample.normSquared, rightSample.normSquared) <= 0)
        ? leftSample.normSquared : rightSample.normSquared;
    ZeroSample bestSample = (fxCompareAbs(leftSample.normSquared, rightSample.normSquared) <= 0)
        ? leftSample : rightSample;

    Fx lo = leftSample.b;
    Fx hi = rightSample.b;
    const Fx one = fxFromUInt(1);
    const Fx phi = fxParseDecimal("0.61803398874989484820458683436563811772");
    const Fx oneMinusPhi = fxSub(one, phi);

    Fx x1 = fxAdd(lo, fxMul(fxSub(hi, lo), oneMinusPhi));
    Fx x2 = fxAdd(lo, fxMul(fxSub(hi, lo), phi));
    ZeroSample s1 = evaluateZetaFresh(engine, x1, checkpoint);
    ZeroSample s2 = evaluateZetaFresh(engine, x2, checkpoint);

    for (int iteration = 0; iteration < 48; ++iteration) {
        if (fxCompareAbs(s1.normSquared, best) < 0) {
            best = s1.normSquared;
            bestSample = s1;
        }
        if (fxCompareAbs(s2.normSquared, best) < 0) {
            best = s2.normSquared;
            bestSample = s2;
        }

        if (fxCompareAbs(s1.normSquared, s2.normSquared) <= 0) {
            hi = x2;
            x2 = x1;
            s2 = s1;
            x1 = fxAdd(lo, fxMul(fxSub(hi, lo), oneMinusPhi));
            s1 = evaluateZetaFresh(engine, x1, checkpoint);
        } else {
            lo = x1;
            x1 = x2;
            s1 = s2;
            x2 = fxAdd(lo, fxMul(fxSub(hi, lo), phi));
            s2 = evaluateZetaFresh(engine, x2, checkpoint);
        }
    }

    const Fx midpoint = fxDivU32(fxAdd(lo, hi), 2);
    const ZeroSample middleSample = evaluateZetaFresh(engine, midpoint, checkpoint);
    if (fxCompareAbs(middleSample.normSquared, best) < 0)
        bestSample = middleSample;

    return bestSample;
}

} // namespace zeta256
