#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

#include <ml/core/result.hpp>

namespace ml {

// Callable that evaluates the training objective at arbitrary weights and
// returns the loss together with its gradient. Built by the caller (for
// example from a backend plus dataset); optimizers may evaluate it several
// times per step, which is what enables line-search style methods.
using Objective = std::function<TrainResult(const std::vector<double>& weights)>;

class Optimizer {
public:
    virtual ~Optimizer() = default;

    virtual void step(std::vector<double>& weights, const Objective& objective) = 0;

protected:
    Optimizer() = default;
};

// Stochastic gradient step: one evaluation, one update. Intended to be driven
// with a per-sample (or mini-batch) objective; works with the full-batch
// objective as plain gradient descent. Weight decay, if any, must already be
// part of the objective.
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

// Fletcher-Reeves conjugate gradient for objectives that are quadratic in the
// weights. Each step evaluates the objective at the current point and at one
// probe point along the new direction; the curvature recovered from the two
// gradients yields the exact line-search step, so no extra evaluations are
// needed. Guards: beta clamped to [0, 1], step size bounded, probe step size
// reused from the previous exact step.
class ConjugateGradient final : public Optimizer {
public:
    explicit ConjugateGradient(double learning_rate)
        : learning_rate_(learning_rate), max_step_size_(1.0e6 * learning_rate) {
        if (!(learning_rate > 0.0)) throw std::invalid_argument("ConjugateGradient: learning_rate must be > 0");
    }

    void step(std::vector<double>& weights, const Objective& objective) override {
        if (!objective) throw std::invalid_argument("ConjugateGradient::step: objective must not be empty");

        const TrainResult current = objective(weights);
        if (weights.size() != current.gradient.size()) {
            throw std::invalid_argument("ConjugateGradient::step: weights and gradient must have the same size");
        }

        double squared_norm = 0.0;
        for (double value : current.gradient) squared_norm += value * value;

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

        const double probe_alpha =
            (has_previous_state_ && last_step_size_ > 0.0 && std::isfinite(last_step_size_))
                ? last_step_size_
                : learning_rate_;
        probe_weights_.resize(weights.size());
        for (std::size_t i = 0; i < weights.size(); ++i) {
            probe_weights_[i] = weights[i] + probe_alpha * direction_[i];
        }
        const TrainResult probed = objective(probe_weights_);

        double direction_dot_dg = 0.0;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            direction_dot_dg += direction_[i] * (probed.gradient[i] - current.gradient[i]);
        }
        const double direction_dot_hessian_direction = direction_dot_dg / probe_alpha;

        double alpha = probe_alpha;
        if (direction_dot_hessian_direction > 0.0) {
            alpha = -direction_dot_gradient / direction_dot_hessian_direction;
        }
        if (!(alpha > 0.0) || !std::isfinite(alpha) || alpha > max_step_size_) {
            alpha = probe_alpha;
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
    double learning_rate_;
    double max_step_size_;
    std::vector<double> direction_;
    std::vector<double> probe_weights_;
    double previous_squared_norm_ = 0.0;
    double last_step_size_ = 0.0;
    bool has_previous_state_ = false;
};

}
