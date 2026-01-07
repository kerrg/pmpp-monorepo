cmake_minimum_required(VERSION 3.18)

# 1. Set the architecture BEFORE the project() call
set(CMAKE_CUDA_ARCHITECTURES 52)

# 2. Add the CUDA language to the project
project(pmpp-monorepo LANGUAGES CXX CUDA)

# Per-target exception settings are now used.
# 'StandardCpp' will have exceptions enabled by default.

add_executable(${PROJECT_NAME} main.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)

add_executable(s3_rgb_conversion Chap03/s3_rgb_conversion.cu)

# Copy the sample image to the build directory so the executable can find it
file(COPY Chap03/sample.jpg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(s3_rgb_conversion PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)