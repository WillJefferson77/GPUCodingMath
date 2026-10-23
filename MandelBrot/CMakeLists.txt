
cmake_minimum_required(VERSION 3.24)

project(mandelbrot_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(SDL2 REQUIRED)

add_executable(mandelbrot main.cu)

if(TARGET SDL2::SDL2)
    target_link_libraries(mandelbrot PRIVATE SDL2::SDL2)
else()
    target_include_directories(mandelbrot PRIVATE ${SDL2_INCLUDE_DIRS})
    target_link_libraries(mandelbrot PRIVATE ${SDL2_LIBRARIES})
endif()