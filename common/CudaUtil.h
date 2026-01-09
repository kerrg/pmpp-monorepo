#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include "SafeInt.hpp"

#define LOGIC_CHECK(cond)                                                      \
  if (!(cond)) {                                                               \
    std::cerr << "Logic check failed at " << __FILE__ << ":" << __LINE__       \
              << std::endl;                                                    \
    exit(EXIT_FAILURE);                                                        \
  }

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__             \
                << " - " << cudaGetErrorString(err) << " (" << #call << ")"    \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// The deleter for std::unique_ptr to call cudaFree
struct CudaMemoryDeleter {
  void operator()(void* ptr) const {
    if (ptr) {
      CUDA_CHECK(cudaFree(ptr));
    }
  }
};

// The factory function to create a std::unique_ptr for CUDA device memory
template <typename T>
auto MakeCudaUnique(size_t count) {
  using CudaUniquePtr = std::unique_ptr<T, CudaMemoryDeleter>;
  T* raw_ptr = nullptr;

  // Use SafeInt for the multiplication. This will terminate on overflow
  // when exceptions are disabled, which aligns with our LOGIC_CHECK behavior.
  SafeInt<size_t> size_in_bytes(count);
  size_in_bytes *= sizeof(T);

  CUDA_CHECK(cudaMalloc(&raw_ptr, size_in_bytes));
  return CudaUniquePtr(raw_ptr);
}

// Used to calculate how many blocks are in the grid, as you have to round up or
// elements will be missed.

template <typename T>
T div_up(T a, T b) {
    return (a + b - 1) / b;
}
