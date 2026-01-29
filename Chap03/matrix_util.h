#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

class MatrixUtil {
public:
    struct Matrix {
        size_t rows = 0;
        size_t cols = 0;
        std::vector<float> values;
    };

    static bool ParseFromSizes(const std::string& matrix_sizes, Matrix* a, Matrix* b,
                               std::string* error);
    static bool ParseFromValues(const std::string& matrix_values, Matrix* a, Matrix* b,
                                std::string* error);
    static void PrintMatrix(const Matrix& matrix, const std::string& label);

private:
    friend class MatrixUtilTest_TrimRemovesLeadingAndTrailingWhitespace_Test;
    friend class MatrixUtilTest_RemoveWhitespaceStripsAllWhitespaceCharacters_Test;
    friend class MatrixUtilTest_ParseDimsParsesValidDimensions_Test;
    friend class MatrixUtilTest_SplitTopLevelCommaRejectsMismatchedBraces_Test;
    friend class MatrixUtilTest_SplitTopLevelCommaRejectsQuotedStrings_Test;
    friend class MatrixUtilTest_SplitTopLevelCommaHandlesEmptyString_Test;

    static std::string Trim(const std::string& input);
    static std::string RemoveWhitespace(const std::string& input);
    static bool ParseDims(const std::string& token, size_t* rows, size_t* cols);
    static std::vector<std::string> SplitTopLevelComma(const std::string& input);
    static bool ParseRow(std::string_view row_token, std::vector<float>* row_out);
    static bool ParseMatrixLiteral(const std::string& input, Matrix* matrix_out);
    static Matrix BuildSequentialMatrix(size_t rows, size_t cols, float start_value);
};
