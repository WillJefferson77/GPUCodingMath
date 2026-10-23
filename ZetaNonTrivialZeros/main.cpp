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

constexpr int WINDOW_WIDTH = 1280;
constexpr int WINDOW_HEIGHT = 900;
constexpr int PANEL_HEIGHT = WINDOW_HEIGHT / 3;
constexpr std::size_t HISTORY_LIMIT = WINDOW_WIDTH;

struct GraphSample {
    zeta256::Fx b;
    zeta256::Fx re;
    zeta256::Fx im;
    int yRe = 0;
    int yIm = 0;
    int yMirrorIm = 0;
};

struct SharedState {
    std::atomic<bool> quit{false};
    std::atomic<bool> paused{false};
    std::atomic<bool> workerDone{false};
    std::mutex controlMutex;
    std::condition_variable controlCv;
    std::mutex queueMutex;
    std::deque<GraphSample> completed;
    std::mutex statusMutex;
    std::string errorText;
    std::string latestStatus = "Starting GPU fixed-point zeta engine...";
};

struct StopRequested {};

void workerCheckpoint(SharedState& state) {
    std::unique_lock<std::mutex> lock(state.controlMutex);
    state.controlCv.wait(lock, [&] {
        return state.quit.load() || !state.paused.load();
    });
    if (state.quit.load()) throw StopRequested{};
}

int mapToY(const zeta256::Fx& value, int panelTop) {
    const zeta256::Fx scaled = zeta256::fxDivU32(
        zeta256::fxMulU32(value, static_cast<uint32_t>(PANEL_HEIGHT - 4)), 10);
    const int64_t displacement = zeta256::fxToInt64Trunc(scaled);
    int64_t y = panelTop + PANEL_HEIGHT / 2 - displacement;
    y = std::max<int64_t>(panelTop + 2, std::min<int64_t>(panelTop + PANEL_HEIGHT - 3, y));
    return static_cast<int>(y);
}

GraphSample makeGraphSample(const zeta256::Fx& b, const zeta256::Fx& re, const zeta256::Fx& im) {
    GraphSample sample;
    sample.b = b;
    sample.re = re;
    sample.im = im;
    sample.yRe = mapToY(re, 0);
    sample.yIm = mapToY(im, PANEL_HEIGHT);
    sample.yMirrorIm = mapToY(zeta256::fxNeg(im), PANEL_HEIGHT * 2);
    return sample;
}

void pushStatus(SharedState& state, const std::string& value) {
    std::lock_guard<std::mutex> lock(state.statusMutex);
    state.latestStatus = value;
}

void zetaWorker(SharedState& state, zeta256::Fx step) {
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
                    "The b value exceeded the 32-bit Euler-Maclaurin index range. "
                    "The viewer is an indefinitely advancing sampler, not a literal enumeration of all real b.");
            }
            const uint32_t N = static_cast<uint32_t>(64ull + ceilTwoB);

            zeta256::Fx re, im;
            engine.evaluate(b, N, previousN, (sampleIndex % 256) == 0, re, im,
                            [&state] { workerCheckpoint(state); });
            if (state.quit.load()) break;

            GraphSample sample = makeGraphSample(b, re, im);
            {
                std::lock_guard<std::mutex> lock(state.queueMutex);
                state.completed.push_back(sample);
                while (state.completed.size() > 8) state.completed.pop_front();
            }

            if ((sampleIndex++ % 4) == 0) {
                const std::string status = "b=" + zeta256::fxToString(b, 18) +
                    " | Re=" + zeta256::fxToString(re, 18) +
                    " | Im(+b)=" + zeta256::fxToString(im, 18) +
                    " | Im(-b)=" + zeta256::fxToString(zeta256::fxNeg(im), 18) +
                    " | Q128.320 GPU fixed-point";
                pushStatus(state, status);
            }
            previousN = N;
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
}

void drawPolyline(SDL_Renderer* renderer, const std::deque<GraphSample>& history, int series) {
    if (history.empty()) return;
    std::vector<SDL_Point> points;
    points.reserve(history.size());
    const int xStart = WINDOW_WIDTH - static_cast<int>(history.size());
    int i = 0;
    for (const GraphSample& sample : history) {
        int y = sample.yRe;
        if (series == 1) y = sample.yIm;
        else if (series == 2) y = sample.yMirrorIm;
        points.push_back(SDL_Point{xStart + i, y});
        ++i;
    }
    if (points.size() == 1) SDL_RenderDrawPoint(renderer, points[0].x, points[0].y);
    else SDL_RenderDrawLines(renderer, points.data(), static_cast<int>(points.size()));
}

void render(SDL_Renderer* renderer, const std::deque<GraphSample>& history) {
    SDL_SetRenderDrawColor(renderer, 12, 15, 22, 255);
    SDL_RenderClear(renderer);

    for (int panel = 0; panel < 3; ++panel) {
        const int top = panel * PANEL_HEIGHT;
        SDL_SetRenderDrawColor(renderer, 42, 48, 60, 255);
        SDL_RenderDrawLine(renderer, 0, top, WINDOW_WIDTH - 1, top);
        SDL_RenderDrawLine(renderer, 0, top + PANEL_HEIGHT / 2, WINDOW_WIDTH - 1, top + PANEL_HEIGHT / 2);
        SDL_RenderDrawLine(renderer, 0, top + PANEL_HEIGHT - 1, WINDOW_WIDTH - 1, top + PANEL_HEIGHT - 1);
        for (int x = 160; x < WINDOW_WIDTH; x += 160)
            SDL_RenderDrawLine(renderer, x, top + 1, x, top + PANEL_HEIGHT - 2);
    }

    SDL_SetRenderDrawColor(renderer, 96, 190, 255, 255);
    drawPolyline(renderer, history, 0);
    SDL_SetRenderDrawColor(renderer, 255, 186, 92, 255);
    drawPolyline(renderer, history, 1);
    SDL_SetRenderDrawColor(renderer, 190, 135, 255, 255);
    drawPolyline(renderer, history, 2);

    SDL_RenderPresent(renderer);
}

void usage(const char* exe) {
    std::cout << "Usage: " << exe << " [--step positive-decimal]\n"
              << "Example: " << exe << " --step 0.05\n"
              << "Space: pause/resume sampling (GPU kernels stop between finite stages)\n"
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
            "Zeta(1/2 + i*b) - GPU Q128.320 fixed-point",
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
        std::deque<GraphSample> history;
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
                        const bool newPaused = !state.paused.load();
                        state.paused.store(newPaused);
                        state.controlCv.notify_all();
                    }
                }
            }

            if (!running) break;

            const bool paused = state.paused.load();
            if (!paused) {
                std::lock_guard<std::mutex> lock(state.queueMutex);
                while (!state.completed.empty()) {
                    history.push_back(state.completed.front());
                    state.completed.pop_front();
                    while (history.size() > HISTORY_LIMIT) history.pop_front();
                }
            }

            std::string title;
            bool hasError = false;
            {
                std::lock_guard<std::mutex> lock(state.statusMutex);
                title = state.latestStatus;
                hasError = !state.errorText.empty();
                if (hasError) title = "GPU error - " + state.errorText;
            }
            if (paused) title = "[PAUSED - GPU idle between kernels] " + title;
            else title = "[RUNNING] " + title;
            if (title.size() > 240) title.resize(240);
            SDL_SetWindowTitle(window, title.c_str());

            render(renderer, history);
            SDL_Delay(8);
            if (state.workerDone.load() && hasError) SDL_Delay(16);
        }

        state.quit.store(true);
        state.controlCv.notify_all();
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
