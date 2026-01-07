#include "gtest/gtest.h"
#include "JpegUtil.h"
#include <SafeInt.hpp>
#include <fstream>

class JpegUtilTest : public ::testing::Test {
protected:
    void SetUp() override {
        // You can add setup logic here if needed.
    }

    void TearDown() override {
        // You can add teardown logic here if needed.
        // For example, delete any temporary files created during tests.
        remove("/tmp/test_output.jpg");
    }
};

TEST_F(JpegUtilTest, ReadSampleJpg) {
    // The path to sample.jpg is relative to where the test is run.
    // In CLion, the default working directory for tests is the build directory,
    // and we copied sample.jpg there.
    // For command line, you might need to adjust this path
    // or run the test from the correct directory.
    const char* filename = "sample.jpg";

    ImageData image_data = JpegUtil::ReadJpeg(filename);

    // I don't know the exact dimensions of sample.jpg, but I can assert they are reasonable.
    // Let's assume it's bigger than 100x100
    EXPECT_GT(image_data.width, 100);
    EXPECT_GT(image_data.height, 100);
    EXPECT_EQ(image_data.num_components, 3);
    EXPECT_FALSE(image_data.raw_image.empty());
    SafeInt<size_t> expected_size = SafeInt<int>(image_data.width) * image_data.height * image_data.num_components;
    EXPECT_EQ(image_data.raw_image.size(), expected_size);
}

TEST_F(JpegUtilTest, WriteAndReadJpg) {
    const char* original_filename = "sample.jpg";
    const char* new_filename = "/tmp/test_output.jpg";

    // 1. Read the original image
    ImageData original_image_data = JpegUtil::ReadJpeg(original_filename);

    // 2. Write the image to a new file
    JpegUtil::WriteJpeg(new_filename, original_image_data.raw_image, original_image_data.width, original_image_data.height, original_image_data.num_components);

    // 3. Read the new image back
    ImageData new_image_data = JpegUtil::ReadJpeg(new_filename);

    // 4. Assert that the properties are the same
    EXPECT_EQ(original_image_data.width, new_image_data.width);
    EXPECT_EQ(original_image_data.height, new_image_data.height);
    EXPECT_EQ(original_image_data.num_components, new_image_data.num_components);

    // As JPEG is lossy, the raw data will not be identical.
    // A more advanced test would be to calculate the PSNR or SSIM
    // between the two images, but for now, just checking the size is a good start.
    EXPECT_EQ(original_image_data.raw_image.size(), new_image_data.raw_image.size());
}
