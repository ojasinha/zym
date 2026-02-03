cmake_minimum_required(VERSION 3.18)
project(code-search-cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

include_directories(include)

# Shared source files
set(COMMON_SOURCES
    src/file_utils.cpp
    src/exact_cpu.cpp
    src/fuzzy_cpu.cpp
    src/exact_gpu.cu
    src/fuzzy_gpu.cu
)

# Main CLI tool
add_executable(zym
    tools/zym.cpp
    ${COMMON_SOURCES}
)

set_target_properties(zym PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Benchmark tool
add_executable(benchmark
    tools/benchmark.cpp
    ${COMMON_SOURCES}
)

set_target_properties(benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
