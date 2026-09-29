#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
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

class NewtonCG final : public Optimizer {
public:
    explicit NewtonCG(double tolerance = 0.1, std::size_t max_inner = 20)
        : tolerance_(tolerance), max_inner_(max_inner) {
        if (!(tolerance > 0.0)) throw std::invalid_argument("NewtonCG: tolerance must be > 0");
        if (max_inner_ == 0) throw std::invalid_argument("NewtonCG: max_inner must be > 0");
    }

    void step(std::vector<double>& weights, const Objective& objective) override {
        if (!objective) throw std::invalid_argument("NewtonCG::step: objective must not be empty");

        const TrainResult current = objective(weights);
        if (weights.size() != current.gradient.size()) {
            throw std::invalid_argument("NewtonCG::step: weights and gradient must have the same size");
        }
        const std::size_t n = weights.size();
        gradient_ = current.gradient;

        double gradient_squared_norm = 0.0;
        for (double value : gradient_) gradient_squared_norm += value * value;
        if (gradient_squared_norm == 0.0) return;

        direction_.assign(n, 0.0);
        residual_.assign(n, 0.0);
        conjugate_.assign(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) residual_[i] = -gradient_[i];
        conjugate_ = residual_;

        for (std::size_t iteration = 0; iteration < max_inner_; ++iteration) {
            hessian_vector(weights, conjugate_, objective);
            double p_dot_hp = 0.0;
            for (std::size_t i = 0; i < n; ++i) p_dot_hp += conjugate_[i] * hessian_vector_[i];
            if (!(p_dot_hp > 0.0) || !std::isfinite(p_dot_hp)) break;

            double r_dot_r = 0.0;
            for (double value : residual_) r_dot_r += value * value;
            const double alpha = r_dot_r / p_dot_hp;
            for (std::size_t i = 0; i < n; ++i) {
                direction_[i] += alpha * conjugate_[i];
                residual_[i] -= alpha * hessian_vector_[i];
            }

            double r_new_dot = 0.0;
            for (double value : residual_) r_new_dot += value * value;
            if (r_new_dot <= tolerance_ * tolerance_ * gradient_squared_norm) break;
            for (std::size_t i = 0; i < n; ++i) conjugate_[i] = residual_[i] + (r_new_dot / r_dot_r) * conjugate_[i];
        }

        double direction_dot_gradient = 0.0;
        double direction_squared = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            direction_dot_gradient += direction_[i] * gradient_[i];
            direction_squared += direction_[i] * direction_[i];
        }
        if (direction_squared == 0.0) {
            for (std::size_t i = 0; i < n; ++i) direction_[i] = -gradient_[i];
            direction_dot_gradient = -gradient_squared_norm;
        }

        double step_size = 1.0;
        bool accepted = false;
        for (int backtrack = 0; backtrack < 60; ++backtrack) {
            for (std::size_t i = 0; i < n; ++i) probe_weights_[i] = weights[i] + step_size * direction_[i];
            const TrainResult candidate = objective(probe_weights_);
            if (std::isfinite(candidate.loss) &&
                candidate.loss <= current.loss + 1.0e-4 * step_size * direction_dot_gradient) {
                accepted = true;
                break;
            }
            step_size *= 0.5;
        }
        if (!accepted) return;

        for (std::size_t i = 0; i < n; ++i) weights[i] += step_size * direction_[i];
    }

private:
    void hessian_vector(const std::vector<double>& weights, const std::vector<double>& vector,
                        const Objective& objective) {
        double vector_norm = 0.0;
        for (double value : vector) vector_norm += value * value;
        vector_norm = std::sqrt(vector_norm);
        double weights_norm = 0.0;
        for (double value : weights) weights_norm += value * value;
        weights_norm = std::sqrt(weights_norm);

        double epsilon = std::sqrt(std::numeric_limits<double>::epsilon()) *
                         std::max(1.0, weights_norm) / (vector_norm > 0.0 ? vector_norm : 1.0);
        probe_weights_.resize(weights.size());
        hessian_vector_.assign(weights.size(), 0.0);
        for (int attempt = 0; attempt < 40; ++attempt) {
            for (std::size_t i = 0; i < weights.size(); ++i) {
                probe_weights_[i] = weights[i] + epsilon * vector[i];
            }
            const TrainResult probed = objective(probe_weights_);
            bool finite = std::isfinite(probed.loss);
            for (double value : probed.gradient) finite = finite && std::isfinite(value);
            if (finite) {
                for (std::size_t i = 0; i < weights.size(); ++i) {
                    hessian_vector_[i] = (probed.gradient[i] - gradient_[i]) / epsilon;
                }
                return;
            }
            epsilon *= 1.0e-3;
        }
    }

    double tolerance_;
    std::size_t max_inner_;
    std::vector<double> gradient_;
    std::vector<double> direction_;
    std::vector<double> residual_;
    std::vector<double> conjugate_;
    std::vector<double> probe_weights_;
    std::vector<double> hessian_vector_;
};

}
