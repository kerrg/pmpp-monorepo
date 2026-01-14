#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <CudaUtil.h>
#include "SafeInt.hpp"
#include "JpegUtil.h"

namespace {
    constexpr int kCudaBlockSize = 16;
    constexpr char kModeColorToGrayscale[] = "color2grayscale";
    constexpr char kModeBlur[] = "blur";
    constexpr int kDefaultBlurSize = 1;
    constexpr int kDefaultBlurPasses = 3;
}

enum class Mode {
    ColorToGrayscale,
    Blur,
};

__global__ void blurRGBPixels(uint8_t* blurred, uint8_t* rgb, int width, int height, int blur_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        int neighbors_visited = 0;
        int r = 0;
        int g = 0;
        int b = 0;
        for (int cur_row = row - blur_size; cur_row <= row + blur_size; cur_row++) {
            if (cur_row < 0 || cur_row >= height) {
                continue;
            }
            for (int cur_col = col - blur_size; cur_col <= col + blur_size; cur_col++) {
                if (cur_col < 0 || cur_col >= width) {
                    continue;
                }
                neighbors_visited++;

                int pixel_index = cur_row * width + cur_col;
                r += rgb[3 * pixel_index];
                g += rgb[3 * pixel_index + 1];
                b += rgb[3 * pixel_index + 2];
            }
        }
        // neighbors_visited will always be >= 1
        int pixel_index = row * width + col;
        blurred[3*pixel_index] = (r / neighbors_visited);
        blurred[3*pixel_index + 1] = (g / neighbors_visited);
        blurred[3*pixel_index + 2] = (b / neighbors_visited);
    }
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

Mode ParseMode(const std::string& mode_string) {
    if (mode_string == kModeBlur) {
        return Mode::Blur;
    }
    return Mode::ColorToGrayscale;
}

void RunColorToGrayscale(const ImageData& image_data, const std::string& output_filename) {
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
    CUDA_CHECK(cudaGetLastError());

    std::vector<uint8_t> grayscale_image(gray_size_in_bytes);
    CUDA_CHECK(cudaMemcpy(grayscale_image.data(), out_pixels.get(), gray_size_in_bytes,
                          cudaMemcpyDeviceToHost));

    JpegUtil::WriteJpeg(output_filename.c_str(), grayscale_image, width, height, 1);
}

void RunBlur(const ImageData& image_data, const std::string& output_filename, int blur_size, int blur_passes) {
    int width = image_data.width;
    int height = image_data.height;

    SafeInt<size_t> blurred_size_in_bytes = SafeInt<size_t>(sizeof(uint8_t)) * width * height * 3;
    auto pass_in_pixels = MakeCudaUnique<uint8_t>(blurred_size_in_bytes);
    auto pass_out_pixels = MakeCudaUnique<uint8_t>(blurred_size_in_bytes);
    CUDA_CHECK(cudaMemcpy(pass_in_pixels.get(), image_data.raw_image.data(), blurred_size_in_bytes,
                          cudaMemcpyHostToDevice));

    dim3 dimBlock(kCudaBlockSize, kCudaBlockSize, 1);
    dim3 dimGrid(div_up(width, kCudaBlockSize),
                    div_up(height, kCudaBlockSize), 1);

    for (int pass = 0; pass < blur_passes; ++pass) {
        blurRGBPixels<<<dimGrid, dimBlock>>>(pass_out_pixels.get(), pass_in_pixels.get(),
                                             width, height, blur_size);
        CUDA_CHECK(cudaGetLastError());
        std::swap(pass_in_pixels, pass_out_pixels);
    }

    std::vector<uint8_t> blurred_out_image(blurred_size_in_bytes);
    CUDA_CHECK(cudaMemcpy(blurred_out_image.data(), pass_in_pixels.get(),
                          blurred_size_in_bytes,cudaMemcpyDeviceToHost));
    JpegUtil::WriteJpeg(output_filename.c_str(), blurred_out_image, width, height, 3);
}

int main(int argc, char *argv[]) {
    CLI::App app{"RGB conversion utility"};
    std::string input_filename;
    std::string output_filename;
    std::string mode = kModeColorToGrayscale;
    int blur_size = kDefaultBlurSize;
    int blur_passes = kDefaultBlurPasses;

    app.add_option("-i,--input", input_filename, "Input JPEG file")->required();
    app.add_option("-o,--output", output_filename, "Output JPEG file")->required();
    app.add_option("-m,--mode", mode, "Conversion mode")
        ->check(CLI::IsMember({kModeColorToGrayscale, kModeBlur}));
    auto* blur_size_opt = app.add_option("-b,--blur-size", blur_size, "Blur radius in pixels (blur mode only)")
        ->check(CLI::Range(1, 64));
    auto* blur_passes_opt = app.add_option("-p,--blur-passes", blur_passes, "Number of blur passes (blur mode only)")
        ->check(CLI::Range(1, 32));

    CLI11_PARSE(app, argc, argv);

    Mode mode_kind = ParseMode(mode);
    if (mode_kind != Mode::Blur && (blur_size_opt->count() > 0 || blur_passes_opt->count() > 0)) {
        std::cerr << "Error: --blur-size and --blur-passes require --mode blur." << std::endl;
        return 1;
    }

    ImageData image_data = JpegUtil::ReadJpeg(input_filename.c_str());
    if (mode_kind == Mode::ColorToGrayscale) {
        RunColorToGrayscale(image_data, output_filename);
    } else {
        RunBlur(image_data, output_filename, blur_size, blur_passes);
    }

    return 0;
}
