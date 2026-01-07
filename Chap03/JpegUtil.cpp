#include "JpegUtil.h"
#include <iostream>
#include <cstdint>
#include <SafeInt.hpp>

ImageData JpegUtil::ReadJpeg(const char* filename) {
    unique_file_ptr infile(fopen(filename, "rb"));
    if (!infile) {
        throw std::runtime_error("Error opening " + std::string(filename));
    }

    JpegDecompressor d;
    jpeg_stdio_src(d.get(), infile.get());

    if (jpeg_read_header(d.get(), TRUE) != JPEG_HEADER_OK) {
        throw std::runtime_error("jpeg_read_header failed: Not a valid JPEG file or header.");
    }

    // For grayscale conversion, we want RGB output from the decompressor
    d->out_color_space = JCS_RGB;

    if (!jpeg_start_decompress(d.get())) {
        throw std::runtime_error("jpeg_start_decompress failed: Decompression failed to start.");
    }

    int width = d->output_width;
    int height = d->output_height;
    // Even if the input is grayscale, we are requesting RGB, so components will be 3
    int num_components = d->output_components;

    std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    std::cout << "Number of components (post-decompression): " << num_components << std::endl;

    if (num_components != 3) {
        throw std::runtime_error("This test requires an RGB image (3 components).");
    }

    SafeInt<size_t> image_size = SafeInt<int>(width) * height * num_components;
    std::vector<uint8_t> raw_image(image_size);
    
    while (d->output_scanline < d->output_height) {
        SafeInt<size_t> offset = SafeInt<int>(d->output_scanline) * width * num_components;
        uint8_t* row_data = &raw_image[offset];
        unsigned char* row_pointer = reinterpret_cast<unsigned char*>(row_data);
        if (jpeg_read_scanlines(d.get(), &row_pointer, 1) != 1) {
            throw std::runtime_error("jpeg_read_scanlines failed: Could not read a scanline.");
        }
    }

    if (!jpeg_finish_decompress(d.get())) {
        throw std::runtime_error("jpeg_finish_decompress failed: Decompression failed to finish.");
    }

    std::cout << "Successfully read JPEG into memory buffer." << std::endl;

    return {raw_image, width, height, num_components};
}

void JpegUtil::WriteJpeg(const char* filename, const std::vector<uint8_t>& raw_image, int width, int height, int num_components, int quality) {
    unique_file_ptr outfile(fopen(filename, "wb"));
    if (!outfile) {
        throw std::runtime_error("Error opening output file " + std::string(filename));
    }

    JpegCompressor c;

    jpeg_stdio_dest(c.get(), outfile.get());

    c->image_width = width;
    c->image_height = height;
    c->input_components = num_components;
    c->in_color_space = (num_components == 3) ? JCS_RGB : JCS_GRAYSCALE;

    jpeg_set_defaults(c.get());
    jpeg_set_quality(c.get(), quality, TRUE);
    jpeg_start_compress(c.get(), TRUE);

    while (c->next_scanline < c->image_height) {
        SafeInt<size_t> offset = SafeInt<int>(c->next_scanline) * width * num_components;
        const uint8_t* row_data = &raw_image[offset];
        auto row_pointer = const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(row_data));
        jpeg_write_scanlines(c.get(), &row_pointer, 1);
    }

    jpeg_finish_compress(c.get());
    
    std::cout << "Successfully wrote JPEG to " << filename << std::endl;
}
