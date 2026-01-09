#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <CudaUtil.h>

namespace {
constexpr size_t kTestDataSize = 16;
constexpr size_t kBlockSize = 256;
}  // namespace
// Computes vector sum C = A + B on the GPU
// Each thread performs one pair-wise addition
__global__ void VecAddKernel(float *A, float *B, float *C, size_t n) {
  size_t i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];  // Ignoring overflow errors for now to keep CUDA code simple
                         // during this learning phase.
  }
}

void VecAdd(const std::vector<float> &vec_a, const std::vector<float> &vec_b,
            std::vector<float> *vec_c) {
  LOGIC_CHECK(vec_a.size() == vec_b.size());
  LOGIC_CHECK(vec_b.size() == vec_c->size());

  const size_t element_count = vec_a.size();
  SafeInt<size_t> size_in_bytes(element_count);
  size_in_bytes *= sizeof(float);

  auto A_d = MakeCudaUnique<float>(element_count);
  auto B_d = MakeCudaUnique<float>(element_count);
  auto C_d = MakeCudaUnique<float>(element_count);

  CUDA_CHECK(cudaMemcpy(A_d.get(), vec_a.data(), size_in_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B_d.get(), vec_b.data(), size_in_bytes,
                        cudaMemcpyHostToDevice));

  VecAddKernel<<<(element_count + kBlockSize - 1) / kBlockSize, kBlockSize>>>(
      A_d.get(), B_d.get(), C_d.get(), element_count);

  CUDA_CHECK(cudaMemcpy(vec_c->data(), C_d.get(), size_in_bytes,
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

int main() {
  // --- Basic 16-element test ---
  const std::vector<float> vec_a = {0.0f, 1.0f, 2.0f,  3.0f,  4.0f, 5.0f, 6.0f, 7.0f,
                                  8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
  const std::vector<float> vec_b = {0.0f, 2.0f, 4.0f,  6.0f,  8.0f, 10.0f, 12.0f, 14.0f,
                                  16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f};
  std::vector<float> vec_c(kTestDataSize);

  VecAdd(vec_a, vec_b, &vec_c);

  PrettyPrintResult(vec_c);

  return 0;
}
