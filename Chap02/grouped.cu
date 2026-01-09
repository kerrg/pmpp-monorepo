#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <CudaUtil.h>

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
  const std::vector<float> vec_a = {
      0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
      10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f,
      20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f,
      30.0f, 31.0f};
  const std::vector<float> vec_b = {
      0.0f,  2.0f,  4.0f,  6.0f,  8.0f,  10.0f, 12.0f, 14.0f, 16.0f, 18.0f,
      20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f, 32.0f, 34.0f, 36.0f, 38.0f,
      40.0f, 42.0f, 44.0f, 46.0f, 48.0f, 50.0f, 52.0f, 54.0f, 56.0f, 58.0f,
      60.0f, 62.0f};
  std::vector<float> vec_c_gpu(kTestDataSize / 2); // GPU output is half size
  std::vector<float> vec_c_cpu(kTestDataSize / 2); // CPU output is half size

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
