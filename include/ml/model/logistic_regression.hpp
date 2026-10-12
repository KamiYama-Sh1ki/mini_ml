#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <ml/core/linalg.hpp>
#include <ml/core/result.hpp>
#include <ml/opt/optimizer.hpp>

namespace ml {

inline double sigmoid(double z) {
    if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
    const double e = std::exp(z);
    return e / (1.0 + e);
}

inline double softplus(double z) {
    return z > 30.0 ? z : std::log1p(std::exp(z));
}

inline double l2_penalty(const Vector& weights, double decay) {
    double squared = 0.0;
    for (std::size_t i = 1; i < weights.size(); ++i) squared += weights[i] * weights[i];
    return 0.5 * decay * squared;
}

class LogisticRegression {
public:
    explicit LogisticRegression(std::size_t features) : weights_(features + 1, 0.0) {}

    Vector& weights() noexcept { return weights_; }
    const Vector& weights() const noexcept { return weights_; }

    Vector logits(const Matrix& features) const {
        if (features.cols + 1 != weights_.size())
            throw std::invalid_argument("logistic regression feature count does not match weights");
        Vector result(features.rows, weights_[0]);
        for (std::size_t i = 0; i < features.rows; ++i)
            for (std::size_t j = 0; j < features.cols; ++j)
                result[i] += weights_[j + 1] * features(i, j);
        return result;
    }

    Vector predict_proba(const Matrix& features) const {
        Vector result = logits(features);
        for (double& value : result) value = sigmoid(value);
        return result;
    }

    Vector predict(const Matrix& features) const {
        Vector result = logits(features);
        for (double& value : result) value = value >= 0.0 ? 1.0 : 0.0;
        return result;
    }

private:
    Vector weights_;
};

inline TrainResult evaluate_logistic(const Vector& weights, const Matrix& features,
                                     const Vector& target, double decay) {
    if (features.rows != target.size() || features.cols + 1 != weights.size())
        throw std::invalid_argument("logistic objective dimensions do not match");
    if (features.rows == 0) throw std::invalid_argument("logistic objective requires samples");

    TrainResult result;
    result.gradient.assign(weights.size(), 0.0);
    for (std::size_t i = 0; i < features.rows; ++i) {
        double z = weights[0];
        for (std::size_t j = 0; j < features.cols; ++j) z += weights[j + 1] * features(i, j);
        result.loss += softplus(z) - target[i] * z;
        const double residual = sigmoid(z) - target[i];
        result.gradient[0] += residual;
        for (std::size_t j = 0; j < features.cols; ++j) result.gradient[j + 1] += residual * features(i, j);
    }
    const double inverse = 1.0 / static_cast<double>(features.rows);
    result.loss *= inverse;
    for (double& value : result.gradient) value *= inverse;
    result.loss += l2_penalty(weights, decay);
    for (std::size_t i = 1; i < weights.size(); ++i) result.gradient[i] += decay * weights[i];
    return result;
}

inline Objective make_logistic_objective(Matrix features, Vector target, double decay) {
    return Objective([features = std::move(features), target = std::move(target), decay](const Vector& weights) {
        return evaluate_logistic(weights, features, target, decay);
    });
}

}
