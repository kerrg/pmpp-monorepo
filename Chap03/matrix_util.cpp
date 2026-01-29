#include "matrix_util.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "LogicCheck.h"
std::string MatrixUtil::Trim(const std::string& input) {
    const char* kWhitespace = " \t\n\r\f\v";
    const size_t start = input.find_first_not_of(kWhitespace);
    if (start == std::string::npos) {
        return "";
    }
    const size_t end = input.find_last_not_of(kWhitespace);
    return input.substr(start, end - start + 1);
}

std::string MatrixUtil::RemoveWhitespace(const std::string& input) {
    std::string out = input;
    out.erase(
        std::remove_if(out.begin(),
                       out.end(),
                       [](unsigned char ch) { return std::isspace(ch) != 0; }),
        out.end());
    return out;
}

bool MatrixUtil::ParseDims(const std::string& token, size_t* rows, size_t* cols) {
    const size_t start = token.find_first_not_of(" \t\n\r\f\v");
    if (start == std::string::npos) {
        return false;
    }
    const size_t end = token.find_last_not_of(" \t\n\r\f\v");
    const std::string trimmed = token.substr(start, end - start + 1);
    const size_t split = trimmed.find_first_of("xX");
    if (split == std::string::npos || split == 0 || split + 1 >= trimmed.size()) {
        return false;
    }
    try {
        size_t parsed = 0;
        *rows = static_cast<size_t>(std::stoul(trimmed.substr(0, split), &parsed));
        if (parsed != split) {
            return false;
        }
        *cols =
            static_cast<size_t>(std::stoul(trimmed.substr(split + 1), &parsed));
        if (parsed != trimmed.size() - split - 1) {
            return false;
        }
    } catch (...) {
        return false;
    }
    return *rows > 0 && *cols > 0;
}

std::vector<std::string> MatrixUtil::SplitTopLevelComma(const std::string& input) {
    // Split on commas that are not nested inside {} or [].
    // Throws std::invalid_argument if braces/brackets are unbalanced or mismatched.
    // Examples:
    //   "a,b" -> ["a", "b"]
    //   "{[1,2],[3,4]}, {[5,6],[7,8]}" -> ["{[1,2],[3,4]}", "{[5,6],[7,8]}"]
    std::vector<std::string> parts;
    std::vector<char> stack;
    size_t start = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        const char ch = input[i];
        if (ch == '"' || ch == '\'') {
            throw std::invalid_argument("Quoted strings are not supported");
        }
        if (ch == '{') {
            stack.push_back('}');
        } else if (ch == '}') {
            if (stack.empty() || stack.back() != '}') {
                throw std::invalid_argument("Mismatched closing brace");
            }
            stack.pop_back();
        } else if (ch == '[') {
            stack.push_back(']');
        } else if (ch == ']') {
            if (stack.empty() || stack.back() != ']') {
                throw std::invalid_argument("Mismatched closing bracket");
            }
            stack.pop_back();
        } else if (ch == ',' && stack.empty()) {
            parts.push_back(Trim(input.substr(start, i - start)));
            start = i + 1;
        }
    }
    if (!stack.empty()) {
        throw std::invalid_argument("Unbalanced braces or brackets");
    }
    parts.push_back(Trim(input.substr(start)));
    return parts;
}

bool MatrixUtil::ParseRow(std::string_view row_token, std::vector<float>* row_out) {
    LOGIC_CHECK(row_out != nullptr);
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
        const std::string token(row_token.substr(start, end - start));
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
    std::string_view inner(compact);
    inner.remove_prefix(1);
    inner.remove_suffix(1);
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
    std::vector<std::string> parts;
    try {
        parts = SplitTopLevelComma(matrix_sizes);
    } catch (const std::exception& ex) {
        if (error) {
            *error = ex.what();
        }
        return false;
    }
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
    std::vector<std::string> parts;
    try {
        parts = SplitTopLevelComma(matrix_values);
    } catch (const std::exception& ex) {
        if (error) {
            *error = ex.what();
        }
        return false;
    }
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
