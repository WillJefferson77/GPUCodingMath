# Zeta FP256-target CUDA viewers (CUDA Toolkit 13 + SDL2)

This project computes and plots

\[
\zeta\left(\frac12+ib\right),\qquad b=0,\Delta b,2\Delta b,\ldots
\]

with the zeta computation performed in CUDA kernels. The lower half of the critical line follows from conjugate symmetry:

\[
\zeta\left(\frac12-ib\right)=\overline{\zeta\left(\frac12+ib\right)}.
\]

## Numerical path

- The zeta sum, complex arithmetic, logarithms, inverse square roots, angle reduction, sine/cosine series, and Euler–Maclaurin corrections use custom signed-magnitude multiword fixed-point arithmetic. The GPU math path does **not** use `float` or `double`.
- The internal number format is **Q128.320**: 128 integer bits and 320 fractional bits, represented by fourteen 32-bit limbs. The extra fractional guard bits target 256-bit accuracy for practical plotted heights.
- The algorithm uses the Euler–Maclaurin expansion with 48 Bernoulli correction terms. The finite Dirichlet terms are calculated in CUDA. A high-precision phase table and phase recurrence avoid recalculating logarithms and trig functions for every term at every sample. The phase table is rebased from the fixed-point definition every 256 samples to limit recurrence drift.
- The standalone host tests exercise the same integer arithmetic and compare the direct phase calculation with the phase-recurrence path. During preparation, sample outputs were also checked independently against `mpmath`; the bundled tests do not require or invoke Python/`mpmath`. The viewer's zeta computation runs on the GPU.

**Accuracy caveat:** Q128.320 provides 320 fractional bits; it is not an IEEE-754 floating-point format with a 256-bit significand at every exponent. For the moderate function values shown by this viewer, the absolute precision is designed to exceed a 256-bit target. The Euler–Maclaurin truncation and accumulated fixed-point errors are not formally interval-certified for every possible `b`, so this should not be treated as a proof-grade special-functions library.

## Two separate executables

CMake builds two independent `.exe` files. The original wave display remains available; the second executable plots the **complex value itself** as a point.

### `zeta_waves.exe` — original wave viewer

- Three stacked panels show `Re ζ(1/2+ib)`, `Im ζ(1/2+ib)`, and `Im ζ(1/2-ib) = -Im ζ(1/2+ib)`.
- **Space** pauses/resumes sampling.
- **Esc** or close the window to quit.

### `zeta_complex_plane.exe` — complex-plane point viewer

At each sample `b`, it evaluates

`ζ(1/2 + b*i) = a + d*i`

and plots the point **`(a,d) = (Re ζ, Im ζ)`** on the complex plane. The horizontal axis is the real component `a`; the vertical axis is the imaginary component `d`. A trail shows the path as `b` increases and a highlighted marker identifies the current point. The view expands when needed to keep the latest point visible; the window title shows the current `b`, `a`, `d`, and scale.

- **Space** pauses/resumes. The worker stops at kernel-stage checkpoints; any already-running kernel must finish first.
- **Esc** or close the window to quit.
- Samples are queued with back-pressure so the complex-plane viewer does not intentionally discard samples merely because rendering is slower than calculation.

## Sampled domain

A renderer cannot enumerate all real numbers `b` or literally reach infinity. This viewer samples the ray in fixed increments and continues as long as system resources permit. The Euler–Maclaurin endpoint is `N = 64 + ceil(2b)`; precomputation buffers grow as `N` grows. Available GPU memory and the 32-bit term index are practical limits, not a mathematical enumeration of every real input.

## Dependencies

- NVIDIA CUDA Toolkit 13.x, compatible driver, and NVIDIA GPU.
- CMake 3.25 or newer.
- SDL2 development package.
- C++17 compiler supported by CUDA.
- No MPFR, MPC, GMP, Boost, or other arbitrary-precision package is required by the GPU executable.

## Build on Windows (Visual Studio + vcpkg example)

Install SDL2 with vcpkg:

```powershell
vcpkg install sdl2:x64-windows
```

Configure with the vcpkg toolchain path adjusted for your system:

```powershell
cmake -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

With the Visual Studio generator, the two executables are normally:

- `build/Release/zeta_waves.exe`
- `build/Release/zeta_complex_plane.exe`

Other generators may place them directly in `build/`.

If CMake cannot find CUDA 13, check `nvcc --version` and that the CUDA Toolkit's compiler and CMake package locations are available. If it cannot find SDL2, make sure the x64 SDL2 package is installed and the vcpkg toolchain file was supplied.

## Run

Launch either viewer independently:

```powershell
.\build\Release\zeta_waves.exe --step 0.05
.\build\Release\zeta_complex_plane.exe --step 0.05
```

Smaller `--step` values create denser sampling but require more calculations. Use a plain positive decimal (for example `0.125`, `0.05`, or `0.01`); scientific notation is not accepted yet.

## CPU-side numeric tests

The environment used to prepare this project did not have `nvcc`, so I could not compile the CUDA/SDL executable here. The fixed-point arithmetic and formula were tested with the local C++ compiler:

```bash
g++ -std=c++17 -O2 host_reference_test.cpp -o host_reference_test
./host_reference_test 0
./host_reference_test 1.25
./host_reference_test 50

g++ -std=c++17 -O2 host_phase_test.cpp -o host_phase_test
./host_phase_test
```

Reference values include (the test executable prints 90 decimal places):

- `b=0`: `ζ(1/2) = -1.460354508809586812889499152515298012...`
- `b=1.25`: `Re = 0.270587836323939215668652965437027594...`, `Im = -0.591901560579242483060141209572119881...`
- `b=50`: `Re = -0.081712108320979975048193146802225471...`, `Im = 0.330792194038661295587815274014496897...`

These CPU-side checks do not replace compiling and running the CUDA executable on the target GPU.
