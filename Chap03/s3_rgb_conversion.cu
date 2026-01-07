#include <iostream>
#include <memory>
#include <vector>

#include "CudaUtil.h"
#include "SafeInt.hpp"
#include "JpegUtil.h"

namespace {
    constexpr int kCudaBlockSize = 16;
}

__global__ void convertRGBPixelToGrayscaleKernel(uint8_t* gray, const uint8_t* rgb, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        int i = row * width + col;
        uint8_t r = rgb[3*i];
        uint8_t g = rgb[3*i+1];
        uint8_t b = rgb[3*i+2];
        gray[i] = r * 0.21f + g * 0.72f + b * 0.07f;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file.jpg> <output_file.jpg>" << std::endl;
        return 1;
    }
    const char *input_filename = argv[1];
    const char *output_filename = argv[2];

    ImageData image_data = JpegUtil::ReadJpeg(input_filename);
    int width = image_data.width;
    int height = image_data.height;

    SafeInt<size_t> rgb_size_in_bytes = SafeInt<size_t>(sizeof(uint8_t)) * width * height * 3;
    SafeInt<size_t> gray_size_in_bytes = SafeInt<size_t>(sizeof(uint8_t)) * width * height;

    auto in_pixels = MakeCudaUnique<uint8_t>(rgb_size_in_bytes);
    auto out_pixels = MakeCudaUnique<uint8_t>(gray_size_in_bytes);

    CUDA_CHECK(cudaMemcpy(in_pixels.get(), image_data.raw_image.data(), rgb_size_in_bytes,
                          cudaMemcpyHostToDevice));

    dim3 dimBlock(kCudaBlockSize, kCudaBlockSize, 1);
    dim3 dimGrid(div_up(width, kCudaBlockSize),
                    div_up(height, kCudaBlockSize), 1);
    convertRGBPixelToGrayscaleKernel<<<dimGrid, dimBlock>>>(out_pixels.get(), in_pixels.get(), width, height);

    std::vector<uint8_t> grayscale_image(gray_size_in_bytes);
    CUDA_CHECK(cudaMemcpy(grayscale_image.data(), out_pixels.get(), gray_size_in_bytes,
                          cudaMemcpyDeviceToHost));

    // Write the grayscale image to a new JPEG file
    JpegUtil::WriteJpeg(output_filename, grayscale_image, width, height, 1);

    return 0;
}
