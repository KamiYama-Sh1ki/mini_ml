#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ml/core/linalg.hpp>

int main() {
    using ml::Vector;
    using ml::Matrix;

    {
        const Vector a{1.0, 2.0, 3.0};
        const Vector b{4.0, -5.0, 6.0};
        assert(std::abs(ml::dot(a, b) - 12.0) < 1e-12);
        const Vector sum = a + b;
        const Vector difference = a - b;
        const Vector scaled = 2.5 * a;
        assert(sum == Vector({5.0, -3.0, 9.0}));
        assert(difference == Vector({-3.0, 7.0, -3.0}));
        assert(scaled == Vector({2.5, 5.0, 7.5}));
    }

    {
        Matrix A(2, 3);
        A.set_row(0, {1.0, 2.0, 3.0});
        A.set_row(1, {-1.0, 0.5, 2.0});
        assert(A.row(0) == Vector({1.0, 2.0, 3.0}));
        assert(A(1, 1) == 0.5);
        const Vector x{1.0, -1.0, 2.0};
        assert((A * x) == Vector({5.0, 2.5}));

        Matrix B(3, 2);
        B.set_row(0, {1.0, 0.0});
        B.set_row(1, {2.0, -1.0});
        B.set_row(2, {0.0, 3.0});
        const Matrix C = A * B;
        assert(C.rows == 2 && C.cols == 2);
        assert(std::abs(C(0, 0) - 5.0) < 1e-12);
        assert(std::abs(C(0, 1) - 7.0) < 1e-12);
        assert(std::abs(C(1, 0) - 0.0) < 1e-12);
        assert(std::abs(C(1, 1) - 5.5) < 1e-12);
    }

    {
        const Vector values{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
        assert(std::abs(ml::mean_of(values) - 5.0) < 1e-12);
        assert(std::abs(ml::stddev_of(values) - std::sqrt(32.0 / 7.0)) < 1e-12);
    }

    {
        bool thrown = false;
        try {
            ml::dot(Vector{1.0}, Vector{1.0, 2.0});
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);

        thrown = false;
        try {
            Matrix(2, 3) * Vector{1.0, 2.0};
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);

        thrown = false;
        try {
            Matrix(2, 2) * Matrix(3, 2);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);
    }

    std::cout << "test_linalg passed\n";
}
