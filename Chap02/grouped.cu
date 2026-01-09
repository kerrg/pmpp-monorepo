#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "cuda_util.h"

namespace {
constexpr size_t kTestDataSize = 32;
constexpr size_t kBlockSize = 256;
}  // namespace

// Computes vector sum C = A + B on the GPU
// Each thread performs one pair-wise addition
__global__ void GroupedVecAddKernel(float *A, float *B, float *C, size_t n) {
  // n is the size of the *input* vectors
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t i = tid * 2; // Each thread handles a group of 2 elements

  // We check against n-1 to ensure we don't read past the end of the arrays
  // on the last element if n is not perfectly even (although we assume it is).
  if (i < n - 1) {
    C[tid] = A[i] + A[i + 1] + B[i] + B[i + 1];
  }
}

void GroupedVecAdd(const std::vector<float> &vec_a,
                     const std::vector<float> &vec_b,
                     std::vector<float> *vec_c) {
  LOGIC_CHECK(vec_a.size() == vec_b.size());
  // Output vector C is half the size of the input vectors
  LOGIC_CHECK(vec_a.size() / 2 == vec_c->size());

  const size_t input_element_count = vec_a.size();
  const size_t output_element_count = vec_c->size();

  SafeInt<size_t> input_size_in_bytes(input_element_count);
  input_size_in_bytes *= sizeof(float);

  SafeInt<size_t> output_size_in_bytes(output_element_count);
  output_size_in_bytes *= sizeof(float);

  auto A_d = MakeCudaUnique<float>(input_element_count);
  auto B_d = MakeCudaUnique<float>(input_element_count);
  auto C_d = MakeCudaUnique<float>(output_element_count);

  CUDA_CHECK(cudaMemcpy(A_d.get(), vec_a.data(), input_size_in_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d.get(), vec_b.data(), input_size_in_bytes,
                        cudaMemcpyHostToDevice));

  const size_t grid_size = (output_element_count + kBlockSize - 1) / kBlockSize;
  GroupedVecAddKernel<<<grid_size, kBlockSize>>>(
      A_d.get(), B_d.get(), C_d.get(), input_element_count);

  CUDA_CHECK(cudaMemcpy(vec_c->data(), C_d.get(), output_size_in_bytes,
                        cudaMemcpyDeviceToHost));
}

void PrettyPrintResult(const std::vector<float>& vec_c) {
  std::cout << "Vector C (Result):\n[ ";
  std::cout << std::fixed << std::setprecision(1);

  std::string separator;
  for (const auto val : vec_c) {
    std::cout << separator << val;
    separator = ", ";
  }
  std::cout << " ]" << std::endl;
}

void InitializeTestVectors(std::vector<float>& vec_a, std::vector<float>& vec_b) {
  // Initialize test vectors
  std::iota(vec_a.begin(), vec_a.end(), 0.0f);

  std::generate(vec_b.begin(), vec_b.end(), [counter = 0.0f]() mutable {
    const float val = counter * 2.0f;
    counter += 1.0f;
    return val;
  });
}

void CalculateExpectedResult(const std::vector<float>& vec_a,
                             const std::vector<float>& vec_b,
                             std::vector<float>* vec_c_cpu) {
  // Calculate expected result on the CPU
  for (size_t i = 0; i < vec_c_cpu->size(); ++i) {
    const size_t input_idx = i * 2;
    (*vec_c_cpu)[i] = vec_a[input_idx] + vec_a[input_idx + 1] +
                      vec_b[input_idx] + vec_b[input_idx + 1];
  }
}

void VerifyResults(const std::vector<float>& vec_c_gpu,
                   const std::vector<float>& vec_c_cpu) {
  std::cout << "\n--- Verification ---" << std::endl;
  bool test_passed = true;
  for (size_t i = 0; i < vec_c_cpu.size(); ++i) {
    if (std::abs(vec_c_cpu[i] - vec_c_gpu[i]) > 1e-5) {
      std::cerr << "Mismatch at index " << i << ": CPU=" << vec_c_cpu[i]
                << ", GPU=" << vec_c_gpu[i] << std::endl;
      test_passed = false;
      break;
    }
  }
  LOGIC_CHECK(test_passed && "Test FAILED: GPU result does not match CPU result.");
  std::cout << "Test PASSED!" << std::endl;
}

int main() {
  // --- Basic test ---
  std::vector<float> vec_a(kTestDataSize);
  std::vector<float> vec_b(kTestDataSize);
  std::vector<float> vec_c_gpu(kTestDataSize / 2); // GPU output is half size
  std::vector<float> vec_c_cpu(kTestDataSize / 2); // CPU output is half size

  InitializeTestVectors(vec_a, vec_b);

  CalculateExpectedResult(vec_a, vec_b, &vec_c_cpu);

  // Perform the same calculation on the GPU
  GroupedVecAdd(vec_a, vec_b, &vec_c_gpu);

  std::cout << "GPU result:" << std::endl;
  PrettyPrintResult(vec_c_gpu);
  std::cout << "\nExpected CPU result:" << std::endl;
  PrettyPrintResult(vec_c_cpu);

  // Compare GPU result with CPU result
  VerifyResults(vec_c_gpu, vec_c_cpu);

  return 0;
}
