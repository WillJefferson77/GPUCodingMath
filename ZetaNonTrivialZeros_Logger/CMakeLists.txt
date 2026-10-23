cmake_minimum_required(VERSION 3.25)
project(zeta_fp256_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit 13.0 REQUIRED)
find_package(SDL2 CONFIG QUIET)
if(NOT TARGET SDL2::SDL2)
    find_package(SDL2 REQUIRED)
endif()

# Build the CUDA zeta engine once and share it between two independent viewers.
add_library(zeta_cuda STATIC zeta_cuda.cu)
target_link_libraries(zeta_cuda PUBLIC CUDA::cudart)
target_include_directories(zeta_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(zeta_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Shared
)

function(configure_zeta_viewer target output_name)
    if(TARGET SDL2::SDL2)
        target_link_libraries(${target} PRIVATE SDL2::SDL2)
    else()
        target_include_directories(${target} PRIVATE ${SDL2_INCLUDE_DIRS})
        target_link_libraries(${target} PRIVATE ${SDL2_LIBRARIES})
    endif()
    target_link_libraries(${target} PRIVATE zeta_cuda CUDA::cudart)
    set_target_properties(${target} PROPERTIES OUTPUT_NAME "${output_name}")
    if(MSVC)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
    else()
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
    endif()
endfunction()

# Executable 1: the original three-panel Re/Im waves. Kept intact.
add_executable(zeta_waves main.cpp zero_detection.hpp)
configure_zeta_viewer(zeta_waves "zeta_waves")

# Executable 2: plot zeta(1/2 + i*b) = a + d*i as (a,d) in the complex plane.
add_executable(zeta_complex_plane main_complex_plane.cpp zero_detection.hpp)
configure_zeta_viewer(zeta_complex_plane "zeta_complex_plane")

# No --use_fast_math: the zeta engine uses integer/fixed-point arithmetic.
