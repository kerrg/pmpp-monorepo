#include "matrix_util.h"

#include <gtest/gtest.h>

#include <string>

TEST(MatrixUtilTest, ParseFromSizesBuildsSequentialMatrices) {
    MatrixUtil::Matrix a;
    MatrixUtil::Matrix b;
    std::string error;

    ASSERT_TRUE(MatrixUtil::ParseFromSizes("2x3,3x2", &a, &b, &error));
    EXPECT_TRUE(error.empty());

    ASSERT_EQ(a.rows, 2U);
    ASSERT_EQ(a.cols, 3U);
    ASSERT_EQ(a.values.size(), 6U);
    EXPECT_FLOAT_EQ(a.values[0], 0.0f);
    EXPECT_FLOAT_EQ(a.values[5], 5.0f);

    ASSERT_EQ(b.rows, 3U);
    ASSERT_EQ(b.cols, 2U);
    ASSERT_EQ(b.values.size(), 6U);
    EXPECT_FLOAT_EQ(b.values[0], 1.0f);
    EXPECT_FLOAT_EQ(b.values[5], 6.0f);
}

TEST(MatrixUtilTest, ParseFromValuesUsesRowMajorOrder) {
    MatrixUtil::Matrix a;
    MatrixUtil::Matrix b;
    std::string error;

    ASSERT_TRUE(MatrixUtil::ParseFromValues("{[1,2],[3,4]},{[5,6],[7,8]}", &a, &b, &error));
    EXPECT_TRUE(error.empty());

    ASSERT_EQ(a.rows, 2U);
    ASSERT_EQ(a.cols, 2U);
    ASSERT_EQ(a.values.size(), 4U);
    EXPECT_FLOAT_EQ(a.values[0], 1.0f);
    EXPECT_FLOAT_EQ(a.values[1], 2.0f);
    EXPECT_FLOAT_EQ(a.values[2], 3.0f);
    EXPECT_FLOAT_EQ(a.values[3], 4.0f);

    ASSERT_EQ(b.rows, 2U);
    ASSERT_EQ(b.cols, 2U);
    ASSERT_EQ(b.values.size(), 4U);
    EXPECT_FLOAT_EQ(b.values[0], 5.0f);
    EXPECT_FLOAT_EQ(b.values[1], 6.0f);
    EXPECT_FLOAT_EQ(b.values[2], 7.0f);
    EXPECT_FLOAT_EQ(b.values[3], 8.0f);
}

TEST(MatrixUtilTest, ParseFromValuesRejectsRaggedRows) {
    MatrixUtil::Matrix a;
    MatrixUtil::Matrix b;
    std::string error;

    EXPECT_FALSE(MatrixUtil::ParseFromValues("{[1,2],[3]},{[1]}", &a, &b, &error));
    EXPECT_FALSE(error.empty());
}
