#include <cctype>
#include <climits>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <CudaUtil.h>
#include "SafeInt.hpp"
#include "JpegUtil.h"

namespace {
    constexpr char kModeBlur[] = "blur";
    constexpr char kModeColorToGrayscale[] = "color2grayscale";
    constexpr int kDefaultBlurSize = 1;
    constexpr int kCudaBlockSize = 16;
    constexpr int kDefaultBlurPasses = 3;

    struct Matrix {
        size_t rows = 0;
        size_t cols = 0;
        std::vector<float> values;
    };
} // namespace

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

                size_t pixel_index = static_cast<size_t>(cur_row) * static_cast<size_t>(width)
                                     + static_cast<size_t>(cur_col);
                size_t base = pixel_index * 3U;
                r += rgb[base];
                g += rgb[base + 1U];
                b += rgb[base + 2U];
            }
        }
        // neighbors_visited will always be >= 1
        size_t pixel_index = static_cast<size_t>(row) * static_cast<size_t>(width)
                             + static_cast<size_t>(col);
        size_t base = pixel_index * 3U;
        blurred[base] = static_cast<uint8_t>(r / neighbors_visited);
        blurred[base + 1U] = static_cast<uint8_t>(g / neighbors_visited);
        blurred[base + 2U] = static_cast<uint8_t>(b / neighbors_visited);
    }
}

__global__ void convertRGBPixelToGrayscaleKernel(uint8_t* gray, const uint8_t* rgb, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < width && row < height) {
        size_t i = static_cast<size_t>(row) * static_cast<size_t>(width)
                   + static_cast<size_t>(col);
        size_t base = i * 3U;
        uint8_t r = rgb[base];
        uint8_t g = rgb[base + 1U];
        uint8_t b = rgb[base + 2U];
        gray[i] = r * 0.21f + g * 0.72f + b * 0.07f;
    }
}

void ValidateImageDimsOrExit(int width, int height, int num_components) {
    if (width <= 0 || height <= 0 || num_components <= 0) {
        std::cerr << "Error: invalid image dimensions." << std::endl;
        exit(EXIT_FAILURE);
    }

    SafeInt<int64_t> pixel_count = SafeInt<int64_t>(width) * height;
    SafeInt<int64_t> total_components = pixel_count * num_components;
    if (total_components > std::numeric_limits<size_t>::max()) {
        std::cerr << "Error: image is too large for addressable memory." << std::endl;
        exit(EXIT_FAILURE);
    }
}

Mode ParseMode(const std::string& mode_string) {
    if (mode_string == kModeBlur) {
        return Mode::Blur;
    }
    return Mode::ColorToGrayscale;
}

std::string Trim(const std::string& input) {
    size_t start = 0;
    while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
        ++start;
    }
    size_t end = input.size();
    while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1U]))) {
        --end;
    }
    return input.substr(start, end - start);
}

std::string RemoveWhitespace(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (char ch : input) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            out.push_back(ch);
        }
    }
    return out;
}

bool ParseDims(const std::string& token, size_t* rows, size_t* cols) {
    const size_t split = token.find_first_of("xX");
    if (split == std::string::npos || split == 0 || split + 1 >= token.size()) {
        return false;
    }
    try {
        *rows = static_cast<size_t>(std::stoul(token.substr(0, split)));
        *cols = static_cast<size_t>(std::stoul(token.substr(split + 1)));
    } catch (...) {
        return false;
    }
    return *rows > 0 && *cols > 0;
}

std::vector<std::string> SplitTopLevelComma(const std::string& input) {
    std::vector<std::string> parts;
    int brace_depth = 0;
    int bracket_depth = 0;
    size_t start = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        const char ch = input[i];
        if (ch == '{') {
            ++brace_depth;
        } else if (ch == '}') {
            --brace_depth;
        } else if (ch == '[') {
            ++bracket_depth;
        } else if (ch == ']') {
            --bracket_depth;
        } else if (ch == ',' && brace_depth == 0 && bracket_depth == 0) {
            parts.push_back(Trim(input.substr(start, i - start)));
            start = i + 1;
        }
        if (brace_depth < 0 || bracket_depth < 0) {
            return {};
        }
    }
    if (brace_depth != 0 || bracket_depth != 0) {
        return {};
    }
    parts.push_back(Trim(input.substr(start)));
    return parts;
}

bool ParseRow(const std::string& row_token, std::vector<float>* row_out) {
    if (row_token.empty()) {
        return false;
    }
    size_t start = 0;
    while (start < row_token.size()) {
        const size_t comma = row_token.find(',', start);
        const size_t end = (comma == std::string::npos) ? row_token.size() : comma;
        if (end == start) {
            return false;
        }
        const std::string token = row_token.substr(start, end - start);
        try {
            size_t parsed = 0;
            float value = std::stof(token, &parsed);
            if (parsed != token.size()) {
                return false;
            }
            row_out->push_back(value);
        } catch (...) {
            return false;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return !row_out->empty();
}

bool ParseMatrixLiteral(const std::string& input, Matrix* matrix_out) {
    const std::string compact = RemoveWhitespace(input);
    if (compact.size() < 4 || compact.front() != '{' || compact.back() != '}') {
        return false;
    }
    const std::string inner = compact.substr(1, compact.size() - 2);
    std::vector<std::vector<float>> rows;
    size_t i = 0;
    while (i < inner.size()) {
        if (inner[i] == ',') {
            ++i;
            continue;
        }
        if (inner[i] != '[') {
            return false;
        }
        const size_t close = inner.find(']', i + 1);
        if (close == std::string::npos) {
            return false;
        }
        std::vector<float> row_values;
        if (!ParseRow(inner.substr(i + 1, close - i - 1), &row_values)) {
            return false;
        }
        rows.push_back(std::move(row_values));
        i = close + 1;
    }
    if (rows.empty()) {
        return false;
    }
    const size_t cols = rows.front().size();
    for (const auto& row : rows) {
        if (row.size() != cols) {
            return false;
        }
    }
    matrix_out->rows = rows.size();
    matrix_out->cols = cols;
    matrix_out->values.clear();
    matrix_out->values.reserve(matrix_out->rows * matrix_out->cols);
    for (const auto& row : rows) {
        matrix_out->values.insert(matrix_out->values.end(), row.begin(), row.end());
    }
    return true;
}

Matrix BuildSequentialMatrix(size_t rows, size_t cols, float start_value) {
    Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.values.resize(rows * cols);
    float value = start_value;
    for (float& cell : matrix.values) {
        cell = value;
        value += 1.0f;
    }
    return matrix;
}

__global__ void MatMulKernel(const float* a, const float* b, float* c,
                             int a_rows, int a_cols, int b_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < a_rows && col < b_cols) {
        float sum = 0.0f;
        for (int k = 0; k < a_cols; ++k) {
            sum += a[row * a_cols + k] * b[k * b_cols + col];
        }
        c[row * b_cols + col] = sum;
    }
}

Matrix Multiply(const Matrix& a, const Matrix& b) {
    Matrix result;
    result.rows = a.rows;
    result.cols = b.cols;
    result.values.assign(result.rows * result.cols, 0.0f);

    const size_t a_count = a.values.size();
    const size_t b_count = b.values.size();
    const size_t c_count = result.values.size();

    auto a_dev = MakeCudaUnique<float>(a_count);
    auto b_dev = MakeCudaUnique<float>(b_count);
    auto c_dev = MakeCudaUnique<float>(c_count);

    SafeInt<size_t> a_bytes(a_count);
    a_bytes *= sizeof(float);
    SafeInt<size_t> b_bytes(b_count);
    b_bytes *= sizeof(float);
    SafeInt<size_t> c_bytes(c_count);
    c_bytes *= sizeof(float);

    CUDA_CHECK(cudaMemcpy(a_dev.get(), a.values.data(), a_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_dev.get(), b.values.data(), b_bytes,
                          cudaMemcpyHostToDevice));

    dim3 dimBlock(kCudaBlockSize, kCudaBlockSize, 1);
    dim3 dimGrid(div_up(static_cast<int>(b.cols), kCudaBlockSize),
                 div_up(static_cast<int>(a.rows), kCudaBlockSize), 1);
    MatMulKernel<<<dimGrid, dimBlock>>>(a_dev.get(), b_dev.get(), c_dev.get(),
                                        static_cast<int>(a.rows),
                                        static_cast<int>(a.cols),
                                        static_cast<int>(b.cols));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(result.values.data(), c_dev.get(), c_bytes,
                          cudaMemcpyDeviceToHost));
    return result;
}

void PrintMatrix(const Matrix& matrix, const std::string& label) {
    std::cout << label << " (" << matrix.rows << "x" << matrix.cols << "):\n{\n";
    for (size_t r = 0; r < matrix.rows; ++r) {
        std::cout << "  [";
        for (size_t c = 0; c < matrix.cols; ++c) {
            if (c > 0) {
                std::cout << ", ";
            }
            std::cout << matrix.values[r * matrix.cols + c];
        }
        std::cout << "]";
        if (r + 1 < matrix.rows) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "}" << std::endl;
}

Matrix MultiplyCpu(const Matrix& a, const Matrix& b) {
    Matrix result;
    result.rows = a.rows;
    result.cols = b.cols;
    result.values.assign(result.rows * result.cols, 0.0f);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t k = 0; k < a.cols; ++k) {
            const float aval = a.values[i * a.cols + k];
            for (size_t j = 0; j < b.cols; ++j) {
                result.values[i * result.cols + j] += aval * b.values[k * b.cols + j];
            }
        }
    }
    return result;
}

bool VerifyMatrixResult(const Matrix& gpu, const Matrix& cpu, float tolerance) {
    if (gpu.rows != cpu.rows || gpu.cols != cpu.cols || gpu.values.size() != cpu.values.size()) {
        std::cerr << "Verification failed: mismatched result dimensions." << std::endl;
        return false;
    }
    for (size_t i = 0; i < cpu.values.size(); ++i) {
        const float diff = std::fabs(cpu.values[i] - gpu.values[i]);
        if (diff > tolerance) {
            const size_t row = i / cpu.cols;
            const size_t col = i % cpu.cols;
            std::cerr << "Mismatch at (" << row << "," << col << "): CPU=" << cpu.values[i]
                      << ", GPU=" << gpu.values[i] << std::endl;
            return false;
        }
    }
    return true;
}

int RunMatrixMultiplication(const std::string& matrix_sizes, const std::string& matrix_values,
                            bool verify) {
    if (!matrix_sizes.empty() && !matrix_values.empty()) {
        std::cerr << "Error: use either --matrix-sizes or --matrix-values, not both." << std::endl;
        return 1;
    }

    Matrix a;
    Matrix b;

    if (!matrix_sizes.empty()) {
        auto parts = SplitTopLevelComma(matrix_sizes);
        if (parts.size() != 2) {
            std::cerr << "Error: --matrix-sizes must be two sizes separated by a comma." << std::endl;
            return 1;
        }
        size_t a_rows = 0;
        size_t a_cols = 0;
        size_t b_rows = 0;
        size_t b_cols = 0;
        if (!ParseDims(Trim(parts[0]), &a_rows, &a_cols) ||
            !ParseDims(Trim(parts[1]), &b_rows, &b_cols)) {
            std::cerr << "Error: invalid matrix size format. Use RxC,RxC (e.g., 2x3,3x2)." << std::endl;
            return 1;
        }
        a = BuildSequentialMatrix(a_rows, a_cols, 0.0f);
        b = BuildSequentialMatrix(b_rows, b_cols, 1.0f);
    } else if (!matrix_values.empty()) {
        auto parts = SplitTopLevelComma(matrix_values);
        if (parts.size() != 2) {
            std::cerr << "Error: --matrix-values must contain two matrices separated by a top-level comma." << std::endl;
            return 1;
        }
        if (!ParseMatrixLiteral(parts[0], &a) || !ParseMatrixLiteral(parts[1], &b)) {
            std::cerr << "Error: invalid matrix literal format. Use {[...],[...]},{[...],[...]}." << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Error: --matrix-multiplication requires --matrix-sizes or --matrix-values." << std::endl;
        return 1;
    }

    if (a.cols != b.rows) {
        std::cerr << "Error: incompatible sizes for multiplication: "
                  << a.rows << "x" << a.cols << " and " << b.rows << "x" << b.cols << "."
                  << std::endl;
        return 1;
    }

    PrintMatrix(a, "Matrix A");
    PrintMatrix(b, "Matrix B");
    Matrix result = Multiply(a, b);
    PrintMatrix(result, "Matrix C");

    if (verify) {
        Matrix cpu_result = MultiplyCpu(a, b);
        constexpr float kTolerance = 1e-5f;
        if (!VerifyMatrixResult(result, cpu_result, kTolerance)) {
            std::cerr << "Matrix verification FAILED." << std::endl;
            return 1;
        }
        std::cout << "Matrix verification PASSED." << std::endl;
    }
    return 0;
}

void RunColorToGrayscale(const ImageData& image_data, const std::string& output_filename) {
    int width = image_data.width;
    int height = image_data.height;

    ValidateImageDimsOrExit(width, height, 3);
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

    ValidateImageDimsOrExit(width, height, 3);
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
    bool matrix_multiplication = false;
    bool matrix_verify = false;
    std::string matrix_sizes;
    std::string matrix_values;

    auto* input_opt = app.add_option("-i,--input", input_filename, "Input JPEG file");
    auto* output_opt = app.add_option("-o,--output", output_filename, "Output JPEG file");
    auto* mode_opt = app.add_option("-m,--mode", mode, "Conversion mode")
        ->check(CLI::IsMember({kModeColorToGrayscale, kModeBlur}));
    auto* blur_size_opt = app.add_option("-b,--blur-size", blur_size, "Blur radius in pixels (blur mode only)")
        ->check(CLI::Range(1, 64));
    auto* blur_passes_opt = app.add_option("-p,--blur-passes", blur_passes, "Number of blur passes (blur mode only)")
        ->check(CLI::Range(1, 32));
    app.add_flag("--matrix-multiplication", matrix_multiplication, "Run matrix multiplication");
    app.add_flag("--matrix-verify", matrix_verify, "Verify matrix multiplication against CPU");
    app.add_option("--matrix-sizes", matrix_sizes, "Matrix sizes as RxC,RxC (e.g., 2x3,3x2)");
    app.add_option("--matrix-values", matrix_values, "Matrix values as {[...],[...]},{[...],[...]}");

    CLI11_PARSE(app, argc, argv);

    if (matrix_multiplication) {
        if (input_opt->count() > 0 || output_opt->count() > 0 ||
            mode_opt->count() > 0 || blur_size_opt->count() > 0 || blur_passes_opt->count() > 0) {
            std::cerr << "Error: matrix multiplication cannot be combined with image conversion options." << std::endl;
            return 1;
        }
        return RunMatrixMultiplication(matrix_sizes, matrix_values, matrix_verify);
    }

    if (input_filename.empty() || output_filename.empty()) {
        std::cerr << "Error: --input and --output are required for image conversion." << std::endl;
        return 1;
    }

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
