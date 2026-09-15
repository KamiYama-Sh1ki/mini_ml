#pragma once

#include <cstddef>
#include <vector>

namespace ml {

class PolynomialRegression {
public:
    explicit PolynomialRegression(std::size_t degree)
        : weights_(degree + 1, 0.0) {}

    double predict(double x) const {
        double result = 0.0;
        for (auto it = weights_.rbegin(); it != weights_.rend(); ++it) {
            result = result * x + *it;
        }
        return result;
    }

    std::size_t degree() const noexcept {
        return weights_.size() - 1;
    }

    std::vector<double>& weights() noexcept {
        return weights_;
    }

    const std::vector<double>& weights() const noexcept {
        return weights_;
    }

private:
    std::vector<double> weights_;
};

}