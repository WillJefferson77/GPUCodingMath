#pragma once

#include "fixed320.hpp"
#include <cstdint>
#include <cuda_runtime_api.h>
#include <functional>

namespace zeta256 {

class CudaZetaEngine {
public:
    explicit CudaZetaEngine(const Fx& step);
    ~CudaZetaEngine();
    CudaZetaEngine(const CudaZetaEngine&) = delete;
    CudaZetaEngine& operator=(const CudaZetaEngine&) = delete;

    void evaluate(const Fx& b, uint32_t N, uint32_t previousN, bool fullRephase,
                  Fx& outRe, Fx& outIm,
                  const std::function<void()>& checkpoint);

private:
    static constexpr uint32_t kBlockSize = 128;
    Fx step_{};
    Fx ln2_{};
    Fx pi_{};
    Fx twoPi_{};
    Fx invTwoPi_{};
    uint32_t capacity_ = 0;
    uint32_t partialCapacity_ = 0;

    Fx* d_log_ = nullptr;
    Fx* d_invSqrt_ = nullptr;
    Fx* d_rotRe_ = nullptr;
    Fx* d_rotIm_ = nullptr;
    Fx* d_phaseRe_ = nullptr;
    Fx* d_phaseIm_ = nullptr;
    Fx* d_partialRe_ = nullptr;
    Fx* d_partialIm_ = nullptr;
    Fx* d_sumRe_ = nullptr;
    Fx* d_sumIm_ = nullptr;
    Fx* d_outRe_ = nullptr;
    Fx* d_outIm_ = nullptr;
    Fx* d_bernoulli_ = nullptr;

    void allocateInitial(uint32_t capacity);
    void growCapacity(uint32_t required, const std::function<void()>& checkpoint);
    void check(cudaError_t result, const char* operation) const;
    void synchronize(const char* operation) const;
};

} // namespace zeta256
