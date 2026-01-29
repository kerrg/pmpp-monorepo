#include "matrix_util.h"

#include <gtest/gtest.h>

#include <stdexcept>
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

TEST(MatrixUtilTest, TrimRemovesLeadingAndTrailingWhitespace) {
    EXPECT_EQ(MatrixUtil::Trim("  alpha beta  "), "alpha beta");
    EXPECT_EQ(MatrixUtil::Trim("\t alpha beta\n"), "alpha beta");
    EXPECT_EQ(MatrixUtil::Trim("alpha"), "alpha");
    EXPECT_EQ(MatrixUtil::Trim("   \t\r\n"), "");
    EXPECT_EQ(MatrixUtil::Trim(" alpha  beta "), "alpha  beta");
}

TEST(MatrixUtilTest, RemoveWhitespaceStripsAllWhitespaceCharacters) {
    EXPECT_EQ(MatrixUtil::RemoveWhitespace(" a b c "), "abc");
    EXPECT_EQ(MatrixUtil::RemoveWhitespace("\talpha\nbeta\r"), "alphabeta");
    EXPECT_EQ(MatrixUtil::RemoveWhitespace("alpha"), "alpha");
    EXPECT_EQ(MatrixUtil::RemoveWhitespace("   \t\r\n"), "");
}

TEST(MatrixUtilTest, ParseDimsParsesValidDimensions) {
    size_t rows = 0;
    size_t cols = 0;
    EXPECT_TRUE(MatrixUtil::ParseDims("2x3", &rows, &cols));
    EXPECT_EQ(rows, 2U);
    EXPECT_EQ(cols, 3U);

    rows = 0;
    cols = 0;
    EXPECT_TRUE(MatrixUtil::ParseDims("4X5", &rows, &cols));
    EXPECT_EQ(rows, 4U);
    EXPECT_EQ(cols, 5U);

    rows = 7;
    cols = 9;
    EXPECT_FALSE(MatrixUtil::ParseDims("0x3", &rows, &cols));
    EXPECT_FALSE(MatrixUtil::ParseDims("3x0", &rows, &cols));
    EXPECT_FALSE(MatrixUtil::ParseDims("3x", &rows, &cols));
    EXPECT_FALSE(MatrixUtil::ParseDims("x3", &rows, &cols));
    EXPECT_FALSE(MatrixUtil::ParseDims("3-3", &rows, &cols));
    EXPECT_FALSE(MatrixUtil::ParseDims("3x3x3", &rows, &cols));

    rows = 0;
    cols = 0;
    EXPECT_TRUE(MatrixUtil::ParseDims(" 3x3 ", &rows, &cols));
    EXPECT_EQ(rows, 3U);
    EXPECT_EQ(cols, 3U);
}

TEST(MatrixUtilTest, SplitTopLevelCommaRejectsMismatchedBraces) {
    EXPECT_THROW(MatrixUtil::SplitTopLevelComma("{[}]"), std::invalid_argument);
}

TEST(MatrixUtilTest, SplitTopLevelCommaRejectsQuotedStrings) {
    EXPECT_THROW(MatrixUtil::SplitTopLevelComma("{\"foo, bar\", \"baz\"}"),
                 std::invalid_argument);
}

TEST(MatrixUtilTest, SplitTopLevelCommaHandlesEmptyString) {
    const std::vector<std::string> parts = MatrixUtil::SplitTopLevelComma("");
    ASSERT_EQ(parts.size(), 1U);
    EXPECT_EQ(parts[0], "");
}
