# GPUCodingMath

Using AI-assisted CUDA computing to explore mathematical patterns and numerical behavior.

Hey, welcome! :) 
well we all have seen a lot of math problems and interesting functions or insights, but to truly see and explore them with the eye for deeper understanding and learning, well I haven't, here with the power of GPU, you can see what each looks like and visualize it for better understanding and for maybe a future with brighter ideas :D

This project is a work in progress, so expect experiments rather than a perfectly polished math library. Some parts may change, some results may need further checking, and setup can be a little picky depending on your GPU and software versions.

## What's in here?

The repository is focused on mathematical experiments and visualizations, including things like:

- CUDA-based calculations that use the GPU for computation.
- Experiments with numerical precision and fixed-point arithmetic.
- Mathematical patterns and behavior that can be explored or visualized.
- Graphics experiments, including fractal-style visuals.

The exact state of each experiment may vary as the project develops. If something looks interesting, feel free to explore the code, test it, or build on top of it.

## A note about AI assistance

AI tools were used to help with parts of development. That doesn't make the numerical results automatically correct, though: bugs, precision limits, and implementation mistakes are still possible. Treat the results as experimental, and verify important mathematical claims independently.

## Requirements

The current build setup is intended for Windows and needs the following:

- **Visual Studio C++ Build Tools**, including the MSVC compiler and a Windows SDK.
- **CMake**.
- **NVIDIA CUDA Toolkit** with support for your GPU.
- **NVIDIA driver** compatible with your installed CUDA Toolkit and GPU. The driver version does **not** generally have to exactly match the Toolkit version; use NVIDIA's [CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/) to check supported combinations.
- **vcpkg** for dependency/toolchain configuration.
- **SDL2**.
- **Boost 1.92**.
- The `mpc_cuda` dependency available at `third_party/mpc_cuda`.

Make sure your CUDA Toolkit supports your actual GPU architecture. Having CUDA installed is not enough if the installed Toolkit or driver cannot work with your GPU.
** ^ THIS IS VERY IMPORTANT, if you download a Toolkit that is not compatible with your nvidia driver, you need to then install the compatible driver, so be careful (happened to me btw)**

## Configure and build
After installing everything, including Boost, SDL2, ... through vcpkg,
Open **PowerShell** in the repository's root directory.
First, replace `C:/path/to/vcpkg` in the command below with the actual path to your vcpkg installation.

Run:

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake "-DMPC_CUDA_ROOT=$PWD/third_party/mpc_cuda"
cmake --build build --config Release
```

The first command configures the project and creates the build directory. The second compiles the project in Release mode. Once the build finishes, look in the generated build output for the executable you want to run.

If CMake cannot find a dependency, check that it is installed and that vcpkg is pointing to the correct location. If CUDA configuration fails, double-check your Toolkit installation, GPU support, and driver compatibility first.

## Contributions and reuse

Feel free to study the code, experiment with it, and suggest improvements. Mathematical projects get better when ideas can be tested, compared, and challenged. If you share or redistribute the project, keep the applicable license and copyright notice with it.

Have fun exploring. :D
