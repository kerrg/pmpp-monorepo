#pragma once

#include <vector>
#include <memory>
#include <jpeglib.h>
#include <cstdio>

struct ImageData {
    std::vector<uint8_t> raw_image;
    int width;
    int height;
    int num_components;
};

class JpegUtil {
public:
    static ImageData ReadJpeg(const char* filename);
    static void WriteJpeg(const char *filename, const std::vector<uint8_t> &raw_image, int width,
                          int height, int num_components, int quality = 75);

private:
    // Custom deleter for FILE*
    struct FileCloser {
        void operator()(FILE* fp) const {
            if (fp) fclose(fp);
        }
    };

    using unique_file_ptr = std::unique_ptr<FILE, FileCloser>;

    class JpegDecompressor {
    public:
        JpegDecompressor() {
            c_info_.err = jpeg_std_error(&j_err_);
            jpeg_create_decompress(&c_info_);
        }

        ~JpegDecompressor() {
            jpeg_destroy_decompress(&c_info_);
        }

        // Delete copy constructor and assignment operator
        JpegDecompressor(const JpegDecompressor&) = delete;
        JpegDecompressor& operator=(const JpegDecompressor&) = delete;

        jpeg_decompress_struct* operator->() { return &c_info_; }
        jpeg_decompress_struct* get() { return &c_info_; }

    private:
        jpeg_decompress_struct c_info_{};
        jpeg_error_mgr j_err_{};
    };

    class JpegCompressor {
    public:
        JpegCompressor() {
            c_info_.err = jpeg_std_error(&j_err_);
            jpeg_create_compress(&c_info_);
        }

        ~JpegCompressor() {
            jpeg_destroy_compress(&c_info_);
        }

        // Delete copy constructor and assignment operator
        JpegCompressor(const JpegCompressor&) = delete;
        JpegCompressor& operator=(const JpegCompressor&) = delete;

        jpeg_compress_struct* operator->() { return &c_info_; }
        jpeg_compress_struct* get() { return &c_info_; }

    private:
        jpeg_compress_struct c_info_{};
        jpeg_error_mgr j_err_{};
    };
};
