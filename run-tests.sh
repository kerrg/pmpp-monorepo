#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# 1. Configure CMake in host test mode
echo "Configuring CMake..."
cmake -S . -B build -DBUILD_HOST_TESTS_ONLY=ON -DBUILD_TESTING=ON -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake

# 2. Build the tests
echo "Building tests..."
cmake --build build

# 3. Run the tests
echo "Running tests..."
ctest --test-dir build --output-on-failure

# 4. Clean up the build files
echo "Cleaning up..."
rm -rf build

echo "Done."
