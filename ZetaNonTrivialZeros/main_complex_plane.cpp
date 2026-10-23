#define SDL_MAIN_HANDLED
#include <SDL.h>

#include "fixed_format.hpp"
#include "zeta_cuda.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int WINDOW_WIDTH = 1100;
constexpr int WINDOW_HEIGHT = 900;
constexpr std::size_t HISTORY_LIMIT = 5000;
constexpr std::size_t QUEUE_LIMIT = 512;

struct Sample {
    zeta256::Fx b;
    zeta256::Fx re;
    zeta256::Fx im;
};

struct SharedState {
    std::atomic<bool> quit{false};
    std::atomic<bool> paused{false};
    std::atomic<bool> workerDone{false};
    std::mutex controlMutex;
    std::condition_variable controlCv;
    std::mutex queueMutex;
    std::condition_variable queueSpaceCv;
    std::deque<Sample> completed;
    std::mutex statusMutex;
    std::string errorText;
    std::string latestStatus = "Starting CUDA zeta complex-plane viewer...";
};

struct StopRequested {};

void workerCheckpoint(SharedState& state) {
    std::unique_lock<std::mutex> lock(state.controlMutex);
    state.controlCv.wait(lock, [&] {
        return state.quit.load() || !state.paused.load();
    });
    if (state.quit.load()) throw StopRequested{};
}

void pushStatus(SharedState& state, const std::string& value) {
    std::lock_guard<std::mutex> lock(state.statusMutex);
    state.latestStatus = value;
}

void zetaWorker(SharedState& state, const zeta256::Fx& step) {
    try {
        zeta256::CudaZetaEngine engine(step);
        zeta256::Fx b = zeta256::fxZero();
        uint32_t previousN = 1;
        uint64_t sampleIndex = 0;

        while (!state.quit.load()) {
            workerCheckpoint(state);

            const zeta256::Fx twoB = zeta256::fxMulU32(b, 2);
            const uint64_t ceilTwoB = zeta256::fxCeilU64(twoB);
            if (ceilTwoB == UINT64_MAX || ceilTwoB > static_cast<uint64_t>(UINT32_MAX) - 65ull) {
                throw std::runtime_error(
                    "b exceeded the 32-bit Euler-Maclaurin index range; this is a sampled viewer, not literal enumeration of all real b.");
            }
            const uint32_t N = static_cast<uint32_t>(64ull + ceilTwoB);

            zeta256::Fx re, im;
            engine.evaluate(b, N, previousN, (sampleIndex % 256) == 0, re, im,
                            [&state] { workerCheckpoint(state); });
            if (state.quit.load()) break;

            // Back-pressure keeps the complex-plane trajectory from silently
            // dropping samples when the GPU worker outpaces the SDL renderer.
            {
                std::unique_lock<std::mutex> lock(state.queueMutex);
                state.queueSpaceCv.wait(lock, [&] {
                    return state.quit.load() || state.completed.size() < QUEUE_LIMIT;
                });
                if (state.quit.load()) break;
                state.completed.push_back(Sample{b, re, im});
            }

            if ((sampleIndex % 4) == 0) {
                pushStatus(state,
                    "b=" + zeta256::fxToString(b, 14) +
                    " | a=Re(zeta)=" + zeta256::fxToString(re, 14) +
                    " | d=Im(zeta)=" + zeta256::fxToString(im, 14) +
                    " | point=(a,d) | GPU Q128.320 fixed-point");
            }

            previousN = N;
            ++sampleIndex;
            b = zeta256::fxAdd(b, step);
        }
    } catch (const StopRequested&) {
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(state.statusMutex);
        state.errorText = e.what();
        state.latestStatus = std::string("ERROR: ") + e.what();
    }

    state.workerDone.store(true);
    state.controlCv.notify_all();
    state.queueSpaceCv.notify_all();
}

bool nearRangeLimit(const zeta256::Fx& value, const zeta256::Fx& extent) {
    const zeta256::Fx threshold = zeta256::fxDivU32(zeta256::fxMulU32(extent, 4), 5);
    return zeta256::fxCompareAbs(value, threshold) >= 0;
}

void ensureVisible(const Sample& sample, zeta256::Fx& extent) {
    while (nearRangeLimit(sample.re, extent) || nearRangeLimit(sample.im, extent)) {
        const zeta256::Fx expanded = zeta256::fxMulU32(extent, 2);
        if (zeta256::fxIsZero(expanded) || zeta256::fxCompareAbs(expanded, extent) <= 0)
            break; // Defensive guard against fixed-point overflow.
        extent = expanded;
    }
}

int mapCoordinate(const zeta256::Fx& value, const zeta256::Fx& inverseExtent,
                  int halfSide, int center, bool invertScreenAxis) {
    const zeta256::Fx normalized = zeta256::fxMul(value, inverseExtent);
    const zeta256::Fx pixelsFx = zeta256::fxMulU32(normalized, static_cast<uint32_t>(halfSide));
    const int64_t displacement = zeta256::fxToInt64Trunc(pixelsFx);
    const int64_t coordinate = static_cast<int64_t>(center) + (invertScreenAxis ? -displacement : displacement);
    return static_cast<int>(std::max<int64_t>(INT32_MIN, std::min<int64_t>(INT32_MAX, coordinate)));
}

SDL_Point toScreen(const Sample& sample, const zeta256::Fx& inverseExtent,
                   int centerX, int centerY, int halfSide) {
    return SDL_Point{
        mapCoordinate(sample.re, inverseExtent, halfSide, centerX, false),
        mapCoordinate(sample.im, inverseExtent, halfSide, centerY, true)
    };
}

void drawCircle(SDL_Renderer* renderer, int cx, int cy, int radius) {
    for (int dy = -radius; dy <= radius; ++dy) {
        const int remaining = radius * radius - dy * dy;
        int span = 0;
        while ((span + 1) * (span + 1) <= remaining) ++span;
        SDL_RenderDrawLine(renderer, cx - span, cy + dy, cx + span, cy + dy);
    }
}

void render(SDL_Renderer* renderer, const std::deque<Sample>& history,
            const zeta256::Fx& extent) {
    int width = WINDOW_WIDTH;
    int height = WINDOW_HEIGHT;
    SDL_GetRendererOutputSize(renderer, &width, &height);
    if (width < 100 || height < 100) return;

    SDL_SetRenderDrawColor(renderer, 12, 15, 22, 255);
    SDL_RenderClear(renderer);

    // Use a square plotting area so one unit along Re and one unit along Im
    // always have the same pixel scale.
    const int side = std::max(40, std::min(width, height) - 100);
    const int halfSide = side / 2 - 8;
    const int centerX = width / 2;
    const int centerY = height / 2;
    const int left = centerX - halfSide;
    const int right = centerX + halfSide;
    const int top = centerY - halfSide;
    const int bottom = centerY + halfSide;

    SDL_SetRenderDrawColor(renderer, 32, 39, 52, 255);
    SDL_RenderDrawLine(renderer, left, top, right, top);
    SDL_RenderDrawLine(renderer, right, top, right, bottom);
    SDL_RenderDrawLine(renderer, right, bottom, left, bottom);
    SDL_RenderDrawLine(renderer, left, bottom, left, top);

    SDL_SetRenderDrawColor(renderer, 25, 31, 42, 255);
    for (int fraction = 1; fraction <= 3; ++fraction) {
        const int offset = (halfSide * fraction) / 4;
        SDL_RenderDrawLine(renderer, centerX - offset, top, centerX - offset, bottom);
        SDL_RenderDrawLine(renderer, centerX + offset, top, centerX + offset, bottom);
        SDL_RenderDrawLine(renderer, left, centerY - offset, right, centerY - offset);
        SDL_RenderDrawLine(renderer, left, centerY + offset, right, centerY + offset);
    }

    SDL_SetRenderDrawColor(renderer, 83, 96, 119, 255);
    SDL_RenderDrawLine(renderer, left, centerY, right, centerY);
    SDL_RenderDrawLine(renderer, centerX, top, centerX, bottom);

    SDL_Rect plotRect{left, top, right - left + 1, bottom - top + 1};
    SDL_RenderSetClipRect(renderer, &plotRect);

    if (!history.empty()) {
        const zeta256::Fx inverseExtent = zeta256::fxInv(extent);
        std::vector<SDL_Point> points;
        points.reserve(history.size());
        for (const Sample& sample : history)
            points.push_back(toScreen(sample, inverseExtent, centerX, centerY, halfSide));

        // Retain the path as well as the current point: each sample is a
        // point in the complex plane, not a value plotted against b.
        SDL_SetRenderDrawColor(renderer, 63, 164, 220, 255);
        if (points.size() > 1)
            SDL_RenderDrawLines(renderer, points.data(), static_cast<int>(points.size()));
        else
            SDL_RenderDrawPoint(renderer, points[0].x, points[0].y);

        SDL_SetRenderDrawColor(renderer, 190, 225, 255, 255);
        for (std::size_t i = 0; i < points.size(); i += 24)
            SDL_RenderDrawPoint(renderer, points[i].x, points[i].y);

        const SDL_Point current = points.back();
        SDL_SetRenderDrawColor(renderer, 255, 192, 82, 255);
        drawCircle(renderer, current.x, current.y, 5);
        SDL_SetRenderDrawColor(renderer, 255, 245, 220, 255);
        SDL_RenderDrawLine(renderer, current.x - 10, current.y, current.x + 10, current.y);
        SDL_RenderDrawLine(renderer, current.x, current.y - 10, current.x, current.y + 10);
    }

    SDL_RenderSetClipRect(renderer, nullptr);
    SDL_RenderPresent(renderer);
}

void usage(const char* exe) {
    std::cout << "Usage: " << exe << " [--step positive-decimal]\n"
              << "Plots each zeta(1/2 + i*b) = a + d*i as the point (a,d)\n"
              << "Space: pause/resume GPU sampling\n"
              << "Esc: quit\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string stepText = "0.05";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            }
            if (arg == "--step") {
                if (i + 1 >= argc) throw std::runtime_error("--step requires a decimal value");
                stepText = argv[++i];
                continue;
            }
            throw std::runtime_error("unknown argument: " + arg);
        }

        const zeta256::Fx step = zeta256::fxParseDecimal(stepText);
        if (step.sign <= 0 || zeta256::fxIsZero(step))
            throw std::runtime_error("--step must be a positive decimal value");

        SDL_SetMainReady();
        if (SDL_Init(SDL_INIT_VIDEO) != 0)
            throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());

        SDL_Window* window = SDL_CreateWindow(
            "Zeta complex plane: zeta(1/2 + i*b) = a + d*i",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
        if (!window) {
            const std::string error = SDL_GetError();
            SDL_Quit();
            throw std::runtime_error("SDL_CreateWindow failed: " + error);
        }

        SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer) renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
        if (!renderer) {
            const std::string error = SDL_GetError();
            SDL_DestroyWindow(window);
            SDL_Quit();
            throw std::runtime_error("SDL_CreateRenderer failed: " + error);
        }

        SharedState state;
        std::thread worker(zetaWorker, std::ref(state), step);
        std::deque<Sample> history;
        zeta256::Fx extent = zeta256::fxFromUInt(2);
        bool running = true;

        while (running) {
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    running = false;
                    break;
                }
                if (event.type == SDL_KEYDOWN && !event.key.repeat) {
                    if (event.key.keysym.sym == SDLK_ESCAPE) {
                        running = false;
                        break;
                    }
                    if (event.key.keysym.sym == SDLK_SPACE) {
                        state.paused.store(!state.paused.load());
                        state.controlCv.notify_all();
                    }
                }
            }
            if (!running) break;

            if (!state.paused.load()) {
                std::lock_guard<std::mutex> lock(state.queueMutex);
                while (!state.completed.empty()) {
                    Sample sample = state.completed.front();
                    state.completed.pop_front();
                    ensureVisible(sample, extent);
                    history.push_back(sample);
                    while (history.size() > HISTORY_LIMIT) history.pop_front();
                }
                state.queueSpaceCv.notify_all();
            }

            std::string title;
            bool hasError = false;
            {
                std::lock_guard<std::mutex> lock(state.statusMutex);
                title = state.latestStatus;
                hasError = !state.errorText.empty();
            }
            title = std::string(state.paused.load() ? "[PAUSED - GPU idle between kernel stages] " : "[RUNNING] ") + title;
            title += " | complex-plane half-range=+/-" + zeta256::fxToString(extent, 2);
            if (title.size() > 240) title.resize(240);
            SDL_SetWindowTitle(window, title.c_str());

            render(renderer, history, extent);
            SDL_Delay(8);
            if (state.workerDone.load() && hasError) SDL_Delay(16);
        }

        state.quit.store(true);
        state.controlCv.notify_all();
        state.queueSpaceCv.notify_all();
        if (worker.joinable()) worker.join();
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();

        if (!state.errorText.empty()) {
            std::cerr << "GPU worker error: " << state.errorText << '\n';
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
