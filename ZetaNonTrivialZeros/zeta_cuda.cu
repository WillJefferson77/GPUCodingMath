#include "zeta_cuda.hpp"
#include "fixed_format.hpp"
#include "zeta_math.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace zeta256 {
namespace {

constexpr uint32_t BLOCK_SIZE = 128;

void throwCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

__global__ void precomputeRangeKernel(Fx* logTable, Fx* invSqrtTable,
                                      Fx* rotationRe, Fx* rotationIm,
                                      uint32_t begin, uint32_t endExclusive,
                                      Fx step, Fx ln2, Fx pi, Fx twoPi, Fx invTwoPi) {
    const uint32_t n = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= endExclusive) return;
    const Fx logn = fxLogUInt(n, ln2);
    const Fx invsqrt = fxInvSqrtUInt(n);
    const Fx angle = fxMul(step, logn);
    const Cx rotation = fxUnitPhaseNegative(angle, pi, twoPi, invTwoPi);
    logTable[n] = logn;
    invSqrtTable[n] = invsqrt;
    rotationRe[n] = rotation.re;
    rotationIm[n] = rotation.im;
}

__global__ void initializePhaseRangeKernel(const Fx* logTable,
                                           Fx* phaseRe, Fx* phaseIm,
                                           uint32_t begin, uint32_t endExclusive,
                                           Fx b, Fx pi, Fx twoPi, Fx invTwoPi) {
    const uint32_t n = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= endExclusive) return;
    if (b.sign == 0 || fxIsZero(b)) {
        phaseRe[n] = fxFromUInt(1);
        phaseIm[n] = fxZero();
    } else {
        const Fx angle = fxMul(b, logTable[n]);
        const Cx phase = fxUnitPhaseNegative(angle, pi, twoPi, invTwoPi);
        phaseRe[n] = phase.re;
        phaseIm[n] = phase.im;
    }
}

__global__ void termAndBlockReduceKernel(const Fx* invSqrtTable,
                                        Fx* phaseRe, Fx* phaseIm,
                                        const Fx* rotationRe, const Fx* rotationIm,
                                        uint32_t N,
                                        Fx* partialRe, Fx* partialIm) {
    __shared__ Fx sharedRe[BLOCK_SIZE];
    __shared__ Fx sharedIm[BLOCK_SIZE];

    const uint32_t lane = threadIdx.x;
    const uint32_t n = blockIdx.x * blockDim.x + lane + 1;
    Fx termRe = fxZero();
    Fx termIm = fxZero();

    if (n < N) {
        const Fx pRe = phaseRe[n];
        const Fx pIm = phaseIm[n];
        termRe = fxMul(pRe, invSqrtTable[n]);
        termIm = fxMul(pIm, invSqrtTable[n]);

        const Fx nextRe = fxSub(fxMul(pRe, rotationRe[n]), fxMul(pIm, rotationIm[n]));
        const Fx nextIm = fxAdd(fxMul(pRe, rotationIm[n]), fxMul(pIm, rotationRe[n]));
        phaseRe[n] = nextRe;
        phaseIm[n] = nextIm;
    }
    sharedRe[lane] = termRe;
    sharedIm[lane] = termIm;
    __syncthreads();

    for (uint32_t stride = BLOCK_SIZE / 2; stride != 0; stride >>= 1) {
        if (lane < stride) {
            sharedRe[lane] = fxAdd(sharedRe[lane], sharedRe[lane + stride]);
            sharedIm[lane] = fxAdd(sharedIm[lane], sharedIm[lane + stride]);
        }
        __syncthreads();
    }
    if (lane == 0) {
        partialRe[blockIdx.x] = sharedRe[0];
        partialIm[blockIdx.x] = sharedIm[0];
    }
}

__global__ void reducePartialsKernel(const Fx* partialRe, const Fx* partialIm,
                                     uint32_t partialCount,
                                     Fx* sumRe, Fx* sumIm) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    Fx re = fxZero();
    Fx im = fxZero();
    for (uint32_t i = 0; i < partialCount; ++i) {
        re = fxAdd(re, partialRe[i]);
        im = fxAdd(im, partialIm[i]);
    }
    sumRe[0] = re;
    sumIm[0] = im;
}

__global__ void applyTailKernel(const Fx* logTable, const Fx* invSqrtTable,
                                const Fx* sumRe, const Fx* sumIm,
                                const Fx* bernoulli, Fx b, uint32_t N,
                                Fx pi, Fx twoPi, Fx invTwoPi,
                                Fx* outRe, Fx* outIm) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    Cx finiteSum;
    finiteSum.re = sumRe[0];
    finiteSum.im = sumIm[0];
    const Cx result = fxEulerMaclaurinTail(finiteSum, b, N, logTable[N], invSqrtTable[N],
                                           pi, twoPi, invTwoPi, bernoulli);
    outRe[0] = result.re;
    outIm[0] = result.im;
}

} // namespace

void CudaZetaEngine::check(cudaError_t result, const char* operation) const {
    throwCuda(result, operation);
}

void CudaZetaEngine::synchronize(const char* operation) const {
    check(cudaDeviceSynchronize(), operation);
}

CudaZetaEngine::CudaZetaEngine(const Fx& step)
    : step_(step),
      ln2_(fxLoadRaw(kLn2_words, kLn2_sign)),
      pi_(fxLoadRaw(kPi_words, kPi_sign)),
      twoPi_(fxLoadRaw(kTwoPi_words, kTwoPi_sign)),
      invTwoPi_(fxLoadRaw(kInvTwoPi_words, kInvTwoPi_sign)) {
    try {
        const std::vector<Fx> coefficients = fxBernoulliHost();
        check(cudaMalloc(reinterpret_cast<void**>(&d_bernoulli_), coefficients.size() * sizeof(Fx)),
              "cudaMalloc(Bernoulli coefficients)");
        check(cudaMemcpy(d_bernoulli_, coefficients.data(), coefficients.size() * sizeof(Fx), cudaMemcpyHostToDevice),
              "cudaMemcpy(Bernoulli coefficients)");
        check(cudaMalloc(reinterpret_cast<void**>(&d_sumRe_), sizeof(Fx)), "cudaMalloc(sumRe)");
        check(cudaMalloc(reinterpret_cast<void**>(&d_sumIm_), sizeof(Fx)), "cudaMalloc(sumIm)");
        check(cudaMalloc(reinterpret_cast<void**>(&d_outRe_), sizeof(Fx)), "cudaMalloc(outRe)");
        check(cudaMalloc(reinterpret_cast<void**>(&d_outIm_), sizeof(Fx)), "cudaMalloc(outIm)");
        allocateInitial(256);
    } catch (...) {
        if (d_log_) cudaFree(d_log_);
        if (d_invSqrt_) cudaFree(d_invSqrt_);
        if (d_rotRe_) cudaFree(d_rotRe_);
        if (d_rotIm_) cudaFree(d_rotIm_);
        if (d_phaseRe_) cudaFree(d_phaseRe_);
        if (d_phaseIm_) cudaFree(d_phaseIm_);
        if (d_partialRe_) cudaFree(d_partialRe_);
        if (d_partialIm_) cudaFree(d_partialIm_);
        if (d_sumRe_) cudaFree(d_sumRe_);
        if (d_sumIm_) cudaFree(d_sumIm_);
        if (d_outRe_) cudaFree(d_outRe_);
        if (d_outIm_) cudaFree(d_outIm_);
        if (d_bernoulli_) cudaFree(d_bernoulli_);
        d_log_ = d_invSqrt_ = d_rotRe_ = d_rotIm_ = d_phaseRe_ = d_phaseIm_ = nullptr;
        d_partialRe_ = d_partialIm_ = d_sumRe_ = d_sumIm_ = d_outRe_ = d_outIm_ = d_bernoulli_ = nullptr;
        throw;
    }
}

void CudaZetaEngine::allocateInitial(uint32_t capacity) {
    capacity_ = capacity;
    partialCapacity_ = (capacity_ - 1 + kBlockSize - 1) / kBlockSize;
    check(cudaMalloc(reinterpret_cast<void**>(&d_log_), (capacity_ + 1ull) * sizeof(Fx)), "cudaMalloc(log table)");
    check(cudaMalloc(reinterpret_cast<void**>(&d_invSqrt_), (capacity_ + 1ull) * sizeof(Fx)), "cudaMalloc(inverse-sqrt table)");
    check(cudaMalloc(reinterpret_cast<void**>(&d_rotRe_), (capacity_ + 1ull) * sizeof(Fx)), "cudaMalloc(rotation real)");
    check(cudaMalloc(reinterpret_cast<void**>(&d_rotIm_), (capacity_ + 1ull) * sizeof(Fx)), "cudaMalloc(rotation imag)");
    check(cudaMalloc(reinterpret_cast<void**>(&d_phaseRe_), (capacity_ + 1ull) * sizeof(Fx)), "cudaMalloc(phase real)");
    check(cudaMalloc(reinterpret_cast<void**>(&d_phaseIm_), (capacity_ + 1ull) * sizeof(Fx)), "cudaMalloc(phase imag)");
    check(cudaMalloc(reinterpret_cast<void**>(&d_partialRe_), partialCapacity_ * sizeof(Fx)), "cudaMalloc(partial real)");
    check(cudaMalloc(reinterpret_cast<void**>(&d_partialIm_), partialCapacity_ * sizeof(Fx)), "cudaMalloc(partial imag)");

    const uint32_t count = capacity_;
    precomputeRangeKernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_log_, d_invSqrt_, d_rotRe_, d_rotIm_, 1, capacity_ + 1,
        step_, ln2_, pi_, twoPi_, invTwoPi_);
    check(cudaGetLastError(), "precomputeRangeKernel launch");
    synchronize("precomputeRangeKernel synchronize");
}

void CudaZetaEngine::growCapacity(uint32_t required, const std::function<void()>& checkpoint) {
    if (required <= capacity_) return;
    uint32_t next = capacity_;
    while (next < required) {
        if (next > std::numeric_limits<uint32_t>::max() / 2) {
            next = required;
            break;
        }
        next *= 2;
    }
    Fx *newLog = nullptr, *newInv = nullptr, *newRotRe = nullptr, *newRotIm = nullptr;
    Fx *newPhaseRe = nullptr, *newPhaseIm = nullptr, *newPartialRe = nullptr, *newPartialIm = nullptr;
    const uint32_t newPartialCapacity = (next - 1 + kBlockSize - 1) / kBlockSize;
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&newLog), (next + 1ull) * sizeof(Fx)), "grow log table");
        check(cudaMalloc(reinterpret_cast<void**>(&newInv), (next + 1ull) * sizeof(Fx)), "grow inverse-sqrt table");
        check(cudaMalloc(reinterpret_cast<void**>(&newRotRe), (next + 1ull) * sizeof(Fx)), "grow rotation real");
        check(cudaMalloc(reinterpret_cast<void**>(&newRotIm), (next + 1ull) * sizeof(Fx)), "grow rotation imag");
        check(cudaMalloc(reinterpret_cast<void**>(&newPhaseRe), (next + 1ull) * sizeof(Fx)), "grow phase real");
        check(cudaMalloc(reinterpret_cast<void**>(&newPhaseIm), (next + 1ull) * sizeof(Fx)), "grow phase imag");
        check(cudaMalloc(reinterpret_cast<void**>(&newPartialRe), newPartialCapacity * sizeof(Fx)), "grow partial real");
        check(cudaMalloc(reinterpret_cast<void**>(&newPartialIm), newPartialCapacity * sizeof(Fx)), "grow partial imag");

        const std::size_t oldBytes = (capacity_ + 1ull) * sizeof(Fx);
        check(cudaMemcpy(newLog, d_log_, oldBytes, cudaMemcpyDeviceToDevice), "copy old log table");
        check(cudaMemcpy(newInv, d_invSqrt_, oldBytes, cudaMemcpyDeviceToDevice), "copy old inverse-sqrt table");
        check(cudaMemcpy(newRotRe, d_rotRe_, oldBytes, cudaMemcpyDeviceToDevice), "copy old rotation real");
        check(cudaMemcpy(newRotIm, d_rotIm_, oldBytes, cudaMemcpyDeviceToDevice), "copy old rotation imag");
        check(cudaMemcpy(newPhaseRe, d_phaseRe_, oldBytes, cudaMemcpyDeviceToDevice), "copy old phase real");
        check(cudaMemcpy(newPhaseIm, d_phaseIm_, oldBytes, cudaMemcpyDeviceToDevice), "copy old phase imag");

        const uint32_t begin = capacity_ + 1;
        const uint32_t end = next + 1;
        const uint32_t count = end - begin;
        precomputeRangeKernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            newLog, newInv, newRotRe, newRotIm, begin, end,
            step_, ln2_, pi_, twoPi_, invTwoPi_);
        check(cudaGetLastError(), "grow precomputeRangeKernel launch");
        synchronize("grow precomputeRangeKernel synchronize");
        checkpoint();
    } catch (...) {
        if (newLog) cudaFree(newLog);
        if (newInv) cudaFree(newInv);
        if (newRotRe) cudaFree(newRotRe);
        if (newRotIm) cudaFree(newRotIm);
        if (newPhaseRe) cudaFree(newPhaseRe);
        if (newPhaseIm) cudaFree(newPhaseIm);
        if (newPartialRe) cudaFree(newPartialRe);
        if (newPartialIm) cudaFree(newPartialIm);
        throw;
    }

    if (d_log_) cudaFree(d_log_);
    if (d_invSqrt_) cudaFree(d_invSqrt_);
    if (d_rotRe_) cudaFree(d_rotRe_);
    if (d_rotIm_) cudaFree(d_rotIm_);
    if (d_phaseRe_) cudaFree(d_phaseRe_);
    if (d_phaseIm_) cudaFree(d_phaseIm_);
    if (d_partialRe_) cudaFree(d_partialRe_);
    if (d_partialIm_) cudaFree(d_partialIm_);
    d_log_ = newLog; d_invSqrt_ = newInv; d_rotRe_ = newRotRe; d_rotIm_ = newRotIm;
    d_phaseRe_ = newPhaseRe; d_phaseIm_ = newPhaseIm;
    d_partialRe_ = newPartialRe; d_partialIm_ = newPartialIm;
    capacity_ = next;
    partialCapacity_ = newPartialCapacity;
}

void CudaZetaEngine::evaluate(const Fx& b, uint32_t N, uint32_t previousN, bool fullRephase,
                              Fx& outRe, Fx& outIm,
                              const std::function<void()>& checkpoint) {
    if (N < 64) N = 64;
    checkpoint();
    growCapacity(N, checkpoint);

    const uint32_t phaseBegin = fullRephase ? 1u : previousN;
    if (phaseBegin < N) {
        const uint32_t count = N - phaseBegin;
        initializePhaseRangeKernel<<<(count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_log_, d_phaseRe_, d_phaseIm_, phaseBegin, N, b, pi_, twoPi_, invTwoPi_);
        check(cudaGetLastError(), "initializePhaseRangeKernel launch");
        synchronize("initializePhaseRangeKernel synchronize");
    }
    checkpoint();

    const uint32_t terms = N - 1;
    const uint32_t blocks = (terms + BLOCK_SIZE - 1) / BLOCK_SIZE;
    termAndBlockReduceKernel<<<blocks, BLOCK_SIZE>>>(
        d_invSqrt_, d_phaseRe_, d_phaseIm_, d_rotRe_, d_rotIm_, N,
        d_partialRe_, d_partialIm_);
    check(cudaGetLastError(), "termAndBlockReduceKernel launch");
    synchronize("termAndBlockReduceKernel synchronize");
    checkpoint();

    reducePartialsKernel<<<1, 1>>>(d_partialRe_, d_partialIm_, blocks, d_sumRe_, d_sumIm_);
    check(cudaGetLastError(), "reducePartialsKernel launch");
    synchronize("reducePartialsKernel synchronize");
    checkpoint();

    applyTailKernel<<<1, 1>>>(d_log_, d_invSqrt_, d_sumRe_, d_sumIm_, d_bernoulli_,
                              b, N, pi_, twoPi_, invTwoPi_, d_outRe_, d_outIm_);
    check(cudaGetLastError(), "applyTailKernel launch");
    synchronize("applyTailKernel synchronize");
    checkpoint();

    check(cudaMemcpy(&outRe, d_outRe_, sizeof(Fx), cudaMemcpyDeviceToHost), "copy zeta real part");
    check(cudaMemcpy(&outIm, d_outIm_, sizeof(Fx), cudaMemcpyDeviceToHost), "copy zeta imaginary part");
}

CudaZetaEngine::~CudaZetaEngine() {
    if (d_log_) cudaFree(d_log_);
    if (d_invSqrt_) cudaFree(d_invSqrt_);
    if (d_rotRe_) cudaFree(d_rotRe_);
    if (d_rotIm_) cudaFree(d_rotIm_);
    if (d_phaseRe_) cudaFree(d_phaseRe_);
    if (d_phaseIm_) cudaFree(d_phaseIm_);
    if (d_partialRe_) cudaFree(d_partialRe_);
    if (d_partialIm_) cudaFree(d_partialIm_);
    if (d_sumRe_) cudaFree(d_sumRe_);
    if (d_sumIm_) cudaFree(d_sumIm_);
    if (d_outRe_) cudaFree(d_outRe_);
    if (d_outIm_) cudaFree(d_outIm_);
    if (d_bernoulli_) cudaFree(d_bernoulli_);
}

} // namespace zeta256
