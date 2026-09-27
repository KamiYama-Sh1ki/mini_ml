#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ml {

class ConjugateGradient {
public:
    explicit ConjugateGradient(double learning_rate, double weight_decay = 0.0)
        : learning_rate_(learning_rate), weight_decay_(weight_decay),
          max_step_size_(1.0e6 * learning_rate) {
        if (!(learning_rate > 0.0)) throw std::invalid_argument("ConjugateGradient: learning_rate must be > 0");
        if (!(weight_decay >= 0.0)) throw std::invalid_argument("ConjugateGradient: weight_decay must be >= 0");
    }

    void step(std::vector<double>& weights, const std::vector<double>& gradient) {
        if (weights.size() != gradient.size()) throw std::invalid_argument("ConjugateGradient::step: weights and gradient must have the same size");

        gradient_.resize(weights.size());
        for (std::size_t i = 0; i < weights.size(); ++i) {
            gradient_[i] = gradient[i] + weight_decay_ * weights[i];
        }

        double previous_squared_norm = 0.0;
        if (has_previous_state_) {
            for (std::size_t i = 0; i < weights.size(); ++i) {
                previous_squared_norm += previous_gradient_[i] * previous_gradient_[i];
            }
        }

        if (has_previous_state_) {
            double direction_dot_gradient = 0.0;
            double direction_dot_dg = 0.0;
            for (std::size_t i = 0; i < weights.size(); ++i) {
                direction_dot_gradient += previous_gradient_[i] * direction_[i];
                direction_dot_dg += direction_[i] * (gradient_[i] - previous_gradient_[i]) / previous_alpha_;
            }

            double alpha = learning_rate_;
            if (direction_dot_dg > 0.0) alpha = -direction_dot_gradient / direction_dot_dg;
            if (!(alpha > 0.0) || !std::isfinite(alpha) || alpha > max_step_size_) {
                alpha = learning_rate_;
            } else {
                last_line_search_alpha_ = alpha;
            }

            const double correction = alpha - previous_alpha_;
            if (correction != 0.0) {
                for (std::size_t i = 0; i < weights.size(); ++i) {
                    weights[i] += correction * direction_[i];
                    gradient_[i] += (correction / previous_alpha_) * (gradient_[i] - previous_gradient_[i]);
                }
            }
        }

        double beta = 0.0;
        double squared_norm = 0.0;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            squared_norm += gradient_[i] * gradient_[i];
        }
        if (has_previous_state_ && previous_squared_norm > 0.0) {
            beta = squared_norm / previous_squared_norm;
            if (!(beta > 0.0) || !std::isfinite(beta)) beta = 0.0;
            if (beta > 1.0) beta = 1.0;
        }

        direction_.resize(weights.size());
        for (std::size_t i = 0; i < weights.size(); ++i) {
            direction_[i] = -gradient_[i] + beta * direction_[i];
        }

        double probe_alpha = learning_rate_;
        if (has_previous_state_ && last_line_search_alpha_ > 0.0 &&
            std::isfinite(last_line_search_alpha_)) {
            probe_alpha = last_line_search_alpha_;
        }
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weights[i] += probe_alpha * direction_[i];
        }

        previous_gradient_ = gradient_;
        previous_squared_norm = squared_norm;
        previous_alpha_ = probe_alpha;
        has_previous_state_ = true;
    }

    void reset() noexcept {
        previous_gradient_.clear();
        previous_squared_norm = 0.0;
        last_line_search_alpha_ = 0.0;
        direction_.clear();
        gradient_.clear();
        previous_alpha_ = 0.0;
        has_previous_state_ = false;
    }

private:
    double learning_rate_;
    double weight_decay_;
    double max_step_size_;
    std::vector<double> previous_gradient_;
    std::vector<double> direction_;
    std::vector<double> gradient_;
    double previous_squared_norm = 0.0;
    double previous_alpha_ = 0.0;
    double last_line_search_alpha_ = 0.0;
    bool has_previous_state_ = false;
};

}

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
