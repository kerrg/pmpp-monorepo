set(S3_RGB_CONVERSION rgb_converter) # Use a consistent name for the executable

if(BUILD_CUDA_EXECUTABLES)
    # Add the executable
    add_executable(${S3_RGB_CONVERSION} s3_rgb_conversion.cu JpegUtil.cpp)

    # Link JPEGLIB and SafeInt headers
    target_include_directories(${S3_RGB_CONVERSION} PRIVATE ${SAFEINT_INCLUDE_DIRS})
    target_link_libraries(${S3_RGB_CONVERSION} PRIVATE JPEG::JPEG)

    set_target_properties(${S3_RGB_CONVERSION} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
    )

    # Define a custom target to run the executable with default arguments
    add_custom_target(run
        COMMAND ${S3_RGB_CONVERSION} ${CMAKE_CURRENT_BINARY_DIR}/sample.jpg /tmp/srgb_output.jpg
        DEPENDS ${S3_RGB_CONVERSION}
        COMMENT "Running ${S3_RGB_CONVERSION} with default arguments"
    )
endif()

# Copy the sample image to the build directory so the executable can find it
file(COPY sample.jpg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    include(GoogleTest)
    add_executable(jpeg_util_test JpegUtil_test.cpp JpegUtil.cpp)
    target_link_libraries(jpeg_util_test PRIVATE GTest::gtest_main JPEG::JPEG)
    gtest_discover_tests(jpeg_util_test)
endif()
