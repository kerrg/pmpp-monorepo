#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <CudaUtil.h>

namespace {
constexpr size_t kTestDataSize = 16;
}  // namespace

void VecAdd(const std::vector<float> &vec_a, const std::vector<float> &vec_b,
            std::vector<float> *vec_c) {
  LOGIC_CHECK(vec_a.size() == vec_b.size());
  LOGIC_CHECK(vec_b.size() == vec_c->size());

  // Copy host vectors to device vectors
  thrust::device_vector<float> d_a = vec_a;
  thrust::device_vector<float> d_b = vec_b;

  // Allocate device vector for the result
  thrust::device_vector<float> d_c(vec_a.size());

  // Perform element-wise addition on the device
  thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(),
                    thrust::plus<float>());

  // Copy the result back to the host
  thrust::copy(d_c.begin(), d_c.end(), vec_c->begin());
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

  try {
    VecAdd(vec_a, vec_b, &vec_c);
  } catch (const std::exception& e) {
    std::cerr << "A Thrust error occurred: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  PrettyPrintResult(vec_c);

  return 0;
}
