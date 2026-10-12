#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ml {

class Vector {
public:
    Vector() = default;
    explicit Vector(std::size_t size, double value = 0.0) : data_(size, value) {}
    Vector(std::initializer_list<double> values) : data_(values) {}

    static Vector from(std::vector<double> values) {
        Vector result;
        result.data_ = std::move(values);
        return result;
    }

    double& operator[](std::size_t index) { return data_[index]; }
    double operator[](std::size_t index) const { return data_[index]; }

    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    auto begin() noexcept { return data_.begin(); }
    auto end() noexcept { return data_.end(); }
    auto begin() const noexcept { return data_.begin(); }
    auto end() const noexcept { return data_.end(); }
    auto rbegin() noexcept { return data_.rbegin(); }
    auto rend() noexcept { return data_.rend(); }
    auto rbegin() const noexcept { return data_.rbegin(); }
    auto rend() const noexcept { return data_.rend(); }

    void push_back(double value) { data_.push_back(value); }
    void resize(std::size_t size, double value = 0.0) { data_.resize(size, value); }
    void clear() noexcept { data_.clear(); }
    void assign(std::size_t size, double value) { data_.assign(size, value); }

    bool operator==(const Vector& other) const { return data_ == other.data_; }

    const std::vector<double>& storage() const noexcept { return data_; }

private:
    std::vector<double> data_;
};

inline double dot(const Vector& left, const Vector& right) {
    if (left.size() != right.size()) throw std::invalid_argument("dot: vectors must have the same size");
    double sum = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) sum += left[i] * right[i];
    return sum;
}

inline Vector operator+(const Vector& left, const Vector& right) {
    if (left.size() != right.size()) throw std::invalid_argument("vector +: sizes must match");
    Vector result(left.size());
    for (std::size_t i = 0; i < left.size(); ++i) result[i] = left[i] + right[i];
    return result;
}

inline Vector operator-(const Vector& left, const Vector& right) {
    if (left.size() != right.size()) throw std::invalid_argument("vector -: sizes must match");
    Vector result(left.size());
    for (std::size_t i = 0; i < left.size(); ++i) result[i] = left[i] - right[i];
    return result;
}

inline Vector operator*(double scale, const Vector& vector) {
    Vector result(vector.size());
    for (std::size_t i = 0; i < vector.size(); ++i) result[i] = scale * vector[i];
    return result;
}

struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> data;

    Matrix() = default;
    Matrix(std::size_t rows_, std::size_t cols_, double value = 0.0)
        : rows(rows_), cols(cols_), data(rows_ * cols_, value) {}

    double& operator()(std::size_t row, std::size_t column) { return data[row * cols + column]; }
    double operator()(std::size_t row, std::size_t column) const { return data[row * cols + column]; }

    Vector row(std::size_t index) const {
        return Vector::from(std::vector<double>(data.begin() + static_cast<std::ptrdiff_t>(index * cols),
                                                data.begin() + static_cast<std::ptrdiff_t>((index + 1) * cols)));
    }
    void set_row(std::size_t index, const Vector& values) {
        if (values.size() != cols) throw std::invalid_argument("set_row: size must match columns");
        for (std::size_t j = 0; j < cols; ++j) data[index * cols + j] = values[j];
    }
};

inline Vector operator*(const Matrix& matrix, const Vector& vector) {
    if (matrix.cols != vector.size()) throw std::invalid_argument("matrix *: sizes must match");
    Vector result(matrix.rows, 0.0);
    for (std::size_t i = 0; i < matrix.rows; ++i)
        for (std::size_t j = 0; j < matrix.cols; ++j) result[i] += matrix(i, j) * vector[j];
    return result;
}

inline Matrix operator*(const Matrix& left, const Matrix& right) {
    if (left.cols != right.rows) throw std::invalid_argument("matrix *: inner sizes must match");
    Matrix result(left.rows, right.cols);
    for (std::size_t i = 0; i < left.rows; ++i)
        for (std::size_t k = 0; k < left.cols; ++k) {
            const double value = left(i, k);
            if (value == 0.0) continue;
            for (std::size_t j = 0; j < right.cols; ++j) result(i, j) += value * right(k, j);
        }
    return result;
}

inline double mean_of(const Vector& vector) {
    if (vector.empty()) throw std::invalid_argument("mean_of: vector must not be empty");
    double sum = 0.0;
    for (double value : vector) sum += value;
    return sum / static_cast<double>(vector.size());
}

inline double stddev_of(const Vector& vector) {
    if (vector.size() < 2) throw std::invalid_argument("stddev_of: need at least two values");
    const double mean = mean_of(vector);
    double sum = 0.0;
    for (double value : vector) sum += (value - mean) * (value - mean);
    return std::sqrt(sum / static_cast<double>(vector.size() - 1));
}

}  // namespace ml
