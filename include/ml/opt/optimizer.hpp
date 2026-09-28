#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

#include <ml/core/result.hpp>

namespace ml {

using Objective = std::function<TrainResult(const std::vector<double>& weights)>;

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void step(std::vector<double>& weights, const Objective& objective) = 0;

protected:
    Optimizer() = default;
};

class SGD final : public Optimizer {
public:
    explicit SGD(double learning_rate)
        : learning_rate_(learning_rate) {
        if (!(learning_rate > 0.0)) throw std::invalid_argument("SGD: learning_rate must be > 0");
    }

    void step(std::vector<double>& weights, const Objective& objective) override {
        if (!objective) throw std::invalid_argument("SGD::step: objective must not be empty");

        const TrainResult result = objective(weights);
        if (weights.size() != result.gradient.size()) {
            throw std::invalid_argument("SGD::step: weights and gradient must have the same size");
        }

        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learning_rate_ * result.gradient[i];
        }
    }

private:
    double learning_rate_;
};

class ConjugateGradient final : public Optimizer {
public:
    ConjugateGradient() = default;

    void step(std::vector<double>& weights, const Objective& objective) override {
        if (!objective) throw std::invalid_argument("ConjugateGradient::step: objective must not be empty");

        const TrainResult current = objective(weights);
        if (weights.size() != current.gradient.size()) {
            throw std::invalid_argument("ConjugateGradient::step: weights and gradient must have the same size");
        }

        double squared_norm = 0.0;
        for (double value : current.gradient) squared_norm += value * value;
        if (squared_norm == 0.0) return;

        double beta = 0.0;
        if (has_previous_state_ && previous_squared_norm_ > 0.0) {
            beta = squared_norm / previous_squared_norm_;
            if (!(beta > 0.0) || !std::isfinite(beta)) beta = 0.0;
            if (beta > 1.0) beta = 1.0;
        }

        direction_.resize(weights.size());
        double direction_dot_gradient = 0.0;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            direction_[i] = -current.gradient[i] + beta * direction_[i];
            direction_dot_gradient += direction_[i] * current.gradient[i];
        }

        double probe_scale = 1.0;
        if (has_previous_state_ && last_step_size_ > 0.0 && std::isfinite(last_step_size_)) {
            probe_scale = last_step_size_;
        }

        TrainResult probed;
        for (int attempt = 0;; ++attempt) {
            probe_weights_.resize(weights.size());
            for (std::size_t i = 0; i < weights.size(); ++i) {
                probe_weights_[i] = weights[i] + probe_scale * direction_[i];
            }
            probed = objective(probe_weights_);
            bool finite = std::isfinite(probed.loss);
            for (double value : probed.gradient) finite = finite && std::isfinite(value);
            if (finite || attempt >= 40) break;
            probe_scale *= 1.0e-3;
        }

        double direction_dot_dg = 0.0;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            direction_dot_dg += direction_[i] * (probed.gradient[i] - current.gradient[i]);
        }
        const double curvature = direction_dot_dg / probe_scale;

        const double alpha = -direction_dot_gradient / curvature;
        if (!(curvature > 0.0) || !(alpha > 0.0) || !std::isfinite(alpha)) {
            reset();
            return;
        }

        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] += alpha * direction_[i];
        }

        previous_squared_norm_ = squared_norm;
        last_step_size_ = alpha;
        has_previous_state_ = true;
    }

    void reset() noexcept {
        direction_.clear();
        probe_weights_.clear();
        previous_squared_norm_ = 0.0;
        last_step_size_ = 0.0;
        has_previous_state_ = false;
    }

private:
    std::vector<double> direction_;
    std::vector<double> probe_weights_;
    double previous_squared_norm_ = 0.0;
    double last_step_size_ = 0.0;
    bool has_previous_state_ = false;
};

}
