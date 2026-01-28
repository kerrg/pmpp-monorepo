#include "matrix_util.h"

#include <cctype>
#include <iostream>
#include <string>
#include <vector>

std::string MatrixUtil::Trim(const std::string& input) {
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

std::string MatrixUtil::RemoveWhitespace(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (char ch : input) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            out.push_back(ch);
        }
    }
    return out;
}

bool MatrixUtil::ParseDims(const std::string& token, size_t* rows, size_t* cols) {
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

std::vector<std::string> MatrixUtil::SplitTopLevelComma(const std::string& input) {
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

bool MatrixUtil::ParseRow(const std::string& row_token, std::vector<float>* row_out) {
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

bool MatrixUtil::ParseMatrixLiteral(const std::string& input, Matrix* matrix_out) {
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

MatrixUtil::Matrix MatrixUtil::BuildSequentialMatrix(size_t rows, size_t cols, float start_value) {
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

bool MatrixUtil::ParseFromSizes(const std::string& matrix_sizes, Matrix* a, Matrix* b,
                                std::string* error) {
    auto parts = SplitTopLevelComma(matrix_sizes);
    if (parts.size() != 2) {
        if (error) {
            *error = "Error: --matrix-sizes must be two sizes separated by a comma.";
        }
        return false;
    }
    size_t a_rows = 0;
    size_t a_cols = 0;
    size_t b_rows = 0;
    size_t b_cols = 0;
    if (!ParseDims(Trim(parts[0]), &a_rows, &a_cols) ||
        !ParseDims(Trim(parts[1]), &b_rows, &b_cols)) {
        if (error) {
            *error = "Error: invalid matrix size format. Use RxC,RxC (e.g., 2x3,3x2).";
        }
        return false;
    }
    *a = BuildSequentialMatrix(a_rows, a_cols, 0.0f);
    *b = BuildSequentialMatrix(b_rows, b_cols, 1.0f);
    return true;
}

bool MatrixUtil::ParseFromValues(const std::string& matrix_values, Matrix* a, Matrix* b,
                                 std::string* error) {
    auto parts = SplitTopLevelComma(matrix_values);
    if (parts.size() != 2) {
        if (error) {
            *error = "Error: --matrix-values must contain two matrices separated by a top-level comma.";
        }
        return false;
    }
    if (!ParseMatrixLiteral(parts[0], a) || !ParseMatrixLiteral(parts[1], b)) {
        if (error) {
            *error = "Error: invalid matrix literal format. Use {[...],[...]},{[...],[...]}.";
        }
        return false;
    }
    return true;
}

void MatrixUtil::PrintMatrix(const Matrix& matrix, const std::string& label) {
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
