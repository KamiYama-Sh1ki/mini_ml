#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ml {

class SGD {
public:
    explicit SGD(double learning_rate, double weight_decay = 0.0)
        : learning_rate_(learning_rate), weight_decay_(weight_decay) {
        if (!(learning_rate > 0.0)) throw std::invalid_argument("SGD: learning_rate must be > 0");
        if (!(weight_decay >= 0.0)) throw std::invalid_argument("SGD: weight_decay must be >= 0");
    }

    void step(std::vector<double>& weights, const std::vector<double>& gradient) const {
        if (weights.size() != gradient.size()) throw std::invalid_argument("SGD::step: weights and gradient must have the same size");

        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learning_rate_ * (gradient[i] + weight_decay_ * weights[i]);
        }
    }

private:
    double learning_rate_;
    double weight_decay_;
};

}
