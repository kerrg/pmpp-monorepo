cmake_minimum_required(VERSION 3.18)

option(BUILD_CUDA_EXECUTABLES "Build CUDA executables" ON)
option(BUILD_HOST_TESTS_ONLY "Build only host-side tests" OFF)

if(BUILD_HOST_TESTS_ONLY)
    set(BUILD_CUDA_EXECUTABLES OFF)
endif()

if(BUILD_CUDA_EXECUTABLES)
    # 1. Set the architecture BEFORE the project() call
    set(CMAKE_CUDA_ARCHITECTURES 52)

    # 2. Add the CUDA language to the project
    project(pmpp-monorepo LANGUAGES CXX CUDA)
else()
    project(pmpp-monorepo LANGUAGES CXX)
endif()

enable_testing()

set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

# 1. Ask CMake to find the directory containing the header
# vcpkg makes sure its 'installed' directory is in the search path
find_path(SAFEINT_INCLUDE_DIRS "SafeInt.hpp")

find_package(JPEG REQUIRED)

include_directories(common)

if(BUILD_CUDA_EXECUTABLES)
    # Per-target exception settings are now used.
    add_executable(${PROJECT_NAME} main.cu)

    set_target_properties(${PROJECT_NAME} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
    )
endif()

add_subdirectory(Chap03)
add_subdirectory(Chap02)
