cmake_minimum_required(VERSION 3.18)

# 1. Set the architecture BEFORE the project() call
set(CMAKE_CUDA_ARCHITECTURES 52)

# 2. Add the CUDA language to the project
project(Chap02 LANGUAGES CXX CUDA)

# Per-target exception settings are now used.
# 'StandardCpp' will have exceptions enabled by default.

add_executable(Chap02 main.cu)
# Disable exceptions for this target
target_compile_options(Chap02 PRIVATE -fno-exceptions)
set_property(TARGET Chap02 APPEND PROPERTY CUDA_FLAGS "-Xcompiler=-fno-exceptions")

add_executable(Grouped grouped.cu)
# Disable exceptions for this target
target_compile_options(Grouped PRIVATE -fno-exceptions)
set_property(TARGET Grouped APPEND PROPERTY CUDA_FLAGS "-Xcompiler=-fno-exceptions")

# Add the local SafeInt library to the include path for both targets
target_include_directories(Chap02 PRIVATE ${SAFEINT_INCLUDE_DIRS})
target_include_directories(Grouped PRIVATE ${SAFEINT_INCLUDE_DIRS})

set_target_properties(Chap02 PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)

add_executable(StandardCpp main_standard_cpp.cu)
target_include_directories(StandardCpp PRIVATE ${SAFEINT_INCLUDE_DIRS})
set_target_properties(StandardCpp PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)