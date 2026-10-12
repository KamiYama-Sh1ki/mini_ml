#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include <ml/core/linalg.hpp>

namespace lab {

struct ConfusionMatrix {
    std::size_t true_negative = 0;
    std::size_t false_positive = 0;
    std::size_t false_negative = 0;
    std::size_t true_positive = 0;

    std::size_t total() const { return true_negative + false_positive + false_negative + true_positive; }
    double accuracy() const { return static_cast<double>(true_negative + true_positive) / static_cast<double>(total()); }
};

inline ConfusionMatrix confusion(const ml::Vector& predicted, const ml::Vector& truth) {
    if (predicted.size() != truth.size()) throw std::invalid_argument("confusion: sizes must match");
    ConfusionMatrix matrix;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        if (truth[i] == 0.0)
            predicted[i] == 0.0 ? ++matrix.true_negative : ++matrix.false_positive;
        else
            predicted[i] == 1.0 ? ++matrix.true_positive : ++matrix.false_negative;
    }
    return matrix;
}

}  // namespace lab
