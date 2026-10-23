
#define SDL_MAIN_HANDLED

#include <SDL.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>


namespace {

constexpr int WIDTH = 1200;
constexpr int HEIGHT = 800;

constexpr double INITIAL_VIEW_WIDTH = 3.5;
constexpr double INITIAL_CENTER_RE = -0.75;
constexpr double INITIAL_CENTER_IM = 0.0;

constexpr double ZOOM_FACTOR = 1.2;
constexpr double MIN_ZOOM = 1.0e-6;
constexpr double MAX_ZOOM = 1.0e15;


void checkCuda(cudaError_t error, const char* file, int line)
{
    if (error != cudaSuccess) {
        std::cerr
            << "\n[CUDA ERROR] " << file << ':' << line << '\n'
            << cudaGetErrorString(error) << std::endl;

        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) checkCuda((call), __FILE__, __LINE__)

[[noreturn]] void sdlFatal(const char* operation)
{
    std::cerr << "\n[SDL ERROR] " << operation
              << ": " << SDL_GetError() << std::endl;

    std::exit(EXIT_FAILURE);
}


__device__ unsigned char toByte(float value)
{
    value = fminf(255.0f, fmaxf(0.0f, value));
    return static_cast<unsigned char>(value);
}

__global__ void renderMandelbrot(
    std::uint32_t* pixels,
    int width,
    int height,
    double centerRe,
    double centerIm,
    double pixelScale,
    int maxIterations)
{
    const int x =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int y =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const double cr =
        centerRe + (static_cast<double>(x) - width * 0.5)
        * pixelScale;

    const double ci =
        centerIm - (static_cast<double>(y) - height * 0.5)
        * pixelScale;

    double zr = 0.0;
    double zi = 0.0;

    double zr2 = 0.0;
    double zi2 = 0.0;

    int iteration = 0;

    while (zr2 + zi2 <= 16.0 &&
           iteration < maxIterations)
    {
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;

        zr2 = zr * zr;
        zi2 = zi * zi;

        ++iteration;
    }

    const int index = y * width + x;

    if (zr2 + zi2 <= 16.0) {
        pixels[index] = 0xFF05070Du;
        return;
    }

    const double magnitude = sqrt(zr2 + zi2);

    const double smooth =
        iteration + 1.0 - log2(log2(magnitude));

    const float t = static_cast<float>(smooth);

    const float red =
        127.5f + 127.5f * cosf(0.10f * t);

    const float green =
        127.5f + 127.5f *
        cosf(0.10f * t + 2.0943951f);

    const float blue =
        127.5f + 127.5f *
        cosf(0.10f * t + 4.1887902f);

    const std::uint32_t r = toByte(red);
    const std::uint32_t g = toByte(green);
    const std::uint32_t b = toByte(blue);

    pixels[index] =
        0xFF000000u | (r << 16) | (g << 8) | b;
}


std::string makeTitle(
    double zoom,
    double centerRe,
    double centerIm)
{
    std::ostringstream title;

    title << "CUDA Mandelbrot"
          << " | Zoom: " << std::scientific
          << std::setprecision(3) << zoom
          << " | Center: " << std::fixed
          << std::setprecision(8)
          << centerRe << ", " << centerIm
          << " | Wheel: zoom | Left-drag: pan"
          << " | R: reset | Esc: quit";

    return title.str();
}

// No SDL_Renderer or SDL_Texture is used.

void displayPixels(
    SDL_Window* window,
    SDL_Surface* frameSurface,
    const std::vector<std::uint32_t>& hostPixels)
{
    const bool needsLock = SDL_MUSTLOCK(frameSurface) != 0;

    if (needsLock && SDL_LockSurface(frameSurface) != 0)
        sdlFatal("SDL_LockSurface");

    // The surface pitch may include padding, so copy one row
    // at a time rather than assuming all rows are contiguous.
    auto* destination =
        static_cast<std::uint8_t*>(frameSurface->pixels);

    const std::size_t rowBytes =
        static_cast<std::size_t>(WIDTH) * sizeof(std::uint32_t);

    for (int y = 0; y < HEIGHT; ++y) {
        std::memcpy(
            destination + static_cast<std::size_t>(y)
                              * frameSurface->pitch,
            hostPixels.data() + static_cast<std::size_t>(y)
                                    * WIDTH,
            rowBytes
        );
    }

    if (needsLock)
        SDL_UnlockSurface(frameSurface);

    SDL_Surface* windowSurface = SDL_GetWindowSurface(window);

    if (windowSurface == nullptr)
        sdlFatal("SDL_GetWindowSurface");

    if (SDL_BlitScaled(
            frameSurface, nullptr,
            windowSurface, nullptr) != 0)
    {
        sdlFatal("SDL_BlitScaled");
    }

    if (SDL_UpdateWindowSurface(window) != 0)
        sdlFatal("SDL_UpdateWindowSurface");
}


void renderFrame(
    SDL_Window* window,
    SDL_Surface* frameSurface,
    std::uint32_t* devicePixels,
    std::vector<std::uint32_t>& hostPixels,
    double centerRe,
    double centerIm,
    double zoom,
    double basePixelScale,
    bool diagnostics)
{
    const double depth =
        std::max(0.0, std::log10(zoom));

    const int maxIterations = std::min(
        1800,
        200 + static_cast<int>(60.0 * depth)
    );

    const double pixelScale = basePixelScale / zoom;

    const dim3 block(16, 16);

    const dim3 grid(
        (WIDTH + block.x - 1) / block.x,
        (HEIGHT + block.y - 1) / block.y
    );

    if (diagnostics)
        std::cerr << "[FRAME 1] Launching CUDA kernel\n";

    renderMandelbrot<<<grid, block>>>(
        devicePixels,
        WIDTH,
        HEIGHT,
        centerRe,
        centerIm,
        pixelScale,
        maxIterations
    );

    CUDA_CHECK(cudaGetLastError());

    if (diagnostics)
        std::cerr << "[FRAME 2] Kernel launched; waiting for GPU\n";

    CUDA_CHECK(cudaDeviceSynchronize());

    if (diagnostics)
        std::cerr << "[FRAME 3] GPU complete; copying pixels\n";

    CUDA_CHECK(cudaMemcpy(
        hostPixels.data(),
        devicePixels,
        hostPixels.size() * sizeof(std::uint32_t),
        cudaMemcpyDeviceToHost
    ));

    if (diagnostics)
        std::cerr << "[FRAME 4] Copy complete; displaying surface\n";

    displayPixels(window, frameSurface, hostPixels);

    if (diagnostics)
        std::cerr << "[FRAME 5] FIRST FRAME DISPLAYED SUCCESSFULLY\n";
}

} // namespace


int main()
{
    std::cerr << std::unitbuf;

    SDL_SetMainReady();


    std::cerr << "[START 1] Detecting CUDA\n";

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "[FATAL] No CUDA GPU detected\n";
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    std::cerr << "[INFO] GPU: " << properties.name << '\n';
    std::cerr << "[INFO] Initializing CUDA context\n";

    CUDA_CHECK(cudaFree(nullptr));

    std::cerr << "[START 2] CUDA initialized\n";


    if (SDL_Init(SDL_INIT_VIDEO) != 0)
        sdlFatal("SDL_Init");

    std::cerr << "[START 3] SDL initialized\n";


    SDL_Window* window = SDL_CreateWindow(
        "CUDA Mandelbrot",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WIDTH,
        HEIGHT,
        SDL_WINDOW_SHOWN
    );

    if (window == nullptr)
        sdlFatal("SDL_CreateWindow");

    std::cerr << "[START 4] Window created\n";

    // IMPORTANT:
    // We intentionally do NOT call SDL_CreateRenderer().

    std::cerr << "[START 5] Getting window surface\n";

    SDL_Surface* windowSurface = SDL_GetWindowSurface(window);

    if (windowSurface == nullptr)
        sdlFatal("SDL_GetWindowSurface");

    std::cerr << "[INFO] Window surface: "
              << windowSurface->w << "x"
              << windowSurface->h << '\n';

    SDL_Surface* frameSurface = SDL_CreateRGBSurfaceWithFormat(
        0,
        WIDTH,
        HEIGHT,
        32,
        SDL_PIXELFORMAT_ARGB8888
    );

    if (frameSurface == nullptr)
        sdlFatal("SDL_CreateRGBSurfaceWithFormat");

    std::cerr << "[START 6] Off-screen surface created\n";


    const std::size_t pixelCount =
        static_cast<std::size_t>(WIDTH) * HEIGHT;

    std::vector<std::uint32_t> hostPixels(pixelCount);

    std::uint32_t* devicePixels = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&devicePixels),
        pixelCount * sizeof(std::uint32_t)
    ));

    std::cerr << "[START 7] GPU memory allocated\n";


    double centerRe = INITIAL_CENTER_RE;
    double centerIm = INITIAL_CENTER_IM;
    double zoom = 1.0;

    const double basePixelScale =
        INITIAL_VIEW_WIDTH / WIDTH;

    SDL_SetWindowTitle(
        window,
        makeTitle(zoom, centerRe, centerIm).c_str()
    );

    std::cerr << "[START 8] Rendering first frame\n";

    renderFrame(
        window,
        frameSurface,
        devicePixels,
        hostPixels,
        centerRe,
        centerIm,
        zoom,
        basePixelScale,
        true
    );

    std::cerr << "[SUCCESS] First frame rendered\n";


    bool running = true;
    bool dragging = false;
    bool dirty = false;

    int lastMouseX = 0;
    int lastMouseY = 0;

    std::cerr << "[LOOP] Entering event loop\n";

    while (running) {
        SDL_Event event;

        while (SDL_PollEvent(&event)) {

            if (event.type == SDL_QUIT) {
                running = false;
            }

            else if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) {
                    running = false;
                }

                else if (event.key.keysym.sym == SDLK_r) {
                    centerRe = INITIAL_CENTER_RE;
                    centerIm = INITIAL_CENTER_IM;
                    zoom = 1.0;
                    dirty = true;
                }
            }

            else if (
                event.type == SDL_MOUSEBUTTONDOWN &&
                event.button.button == SDL_BUTTON_LEFT
            ) {
                dragging = true;
                lastMouseX = event.button.x;
                lastMouseY = event.button.y;
            }

            else if (
                event.type == SDL_MOUSEBUTTONUP &&
                event.button.button == SDL_BUTTON_LEFT
            ) {
                dragging = false;
            }

            else if (
                event.type == SDL_MOUSEMOTION &&
                dragging
            ) {
                const double scale = basePixelScale / zoom;

                const int dx = event.motion.x - lastMouseX;
                const int dy = event.motion.y - lastMouseY;

                centerRe -= dx * scale;
                centerIm += dy * scale;

                lastMouseX = event.motion.x;
                lastMouseY = event.motion.y;

                dirty = true;
            }

            else if (event.type == SDL_MOUSEWHEEL) {
                int mouseX = 0;
                int mouseY = 0;

                SDL_GetMouseState(&mouseX, &mouseY);

                int wheelY = event.wheel.y;

                if (event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED)
                    wheelY = -wheelY;

                if (wheelY != 0) {
                    const double oldScale =
                        basePixelScale / zoom;

                    const double anchorRe =
                        centerRe +
                        (mouseX - WIDTH * 0.5) * oldScale;

                    const double anchorIm =
                        centerIm -
                        (mouseY - HEIGHT * 0.5) * oldScale;

                    zoom = std::clamp(
                        zoom * pow(ZOOM_FACTOR, wheelY),
                        MIN_ZOOM,
                        MAX_ZOOM
                    );

                    const double newScale =
                        basePixelScale / zoom;

                    centerRe =
                        anchorRe -
                        (mouseX - WIDTH * 0.5) * newScale;

                    centerIm =
                        anchorIm +
                        (mouseY - HEIGHT * 0.5) * newScale;

                    dirty = true;
                }
            }

            else if (event.type == SDL_WINDOWEVENT) {
                if (event.window.event == SDL_WINDOWEVENT_EXPOSED ||
                    event.window.event == SDL_WINDOWEVENT_RESTORED) {
                    dirty = true;
                }
            }
        }

        if (!running)
            break;

        if (dirty) {
            renderFrame(
                window,
                frameSurface,
                devicePixels,
                hostPixels,
                centerRe,
                centerIm,
                zoom,
                basePixelScale,
                false
            );

            SDL_SetWindowTitle(
                window,
                makeTitle(zoom, centerRe, centerIm).c_str()
            );

            dirty = false;
        }

        SDL_Delay(1);
    }


    std::cerr << "[CLEANUP] Releasing resources\n";

    CUDA_CHECK(cudaFree(devicePixels));

    SDL_FreeSurface(frameSurface);
    SDL_DestroyWindow(window);
    SDL_Quit();

    std::cerr << "[SUCCESS] Application exited normally\n";

    return EXIT_SUCCESS;
}
