#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <ml/opt/optimizer.hpp>

namespace ml {

class NonlinearConjugateGradient final : public Optimizer {
public:
    void step(Vector& weights, const Objective& objective) override {
        if (!objective) throw std::invalid_argument("nonlinear CG requires an objective");
        const auto current = objective(weights);
        if (current.gradient.size() != weights.size() || !finite(current))
            throw std::invalid_argument("nonlinear CG requires finite loss and matching gradient");
        const double squared = dot(current.gradient, current.gradient);
        if (squared < 1e-20) return;

        Vector direction = -1.0 * current.gradient;
        if (!previous_gradient_.empty()) {
            const double denominator = dot(previous_gradient_, previous_gradient_);
            const double beta = std::max(0.0, dot(current.gradient, current.gradient - previous_gradient_) / denominator);
            direction = direction + beta * previous_direction_;
        }
        double slope = dot(current.gradient, direction);
        if (!(slope < -1e-3 * squared)) {
            direction = -1.0 * current.gradient;
            slope = -squared;
        }

        auto candidate = search(weights, direction, current, slope, objective);
        if (!candidate.accepted) {
            direction = -1.0 * current.gradient;
            slope = -squared;
            candidate = search(weights, direction, current, slope, objective);
        }
        if (!candidate.accepted) {
            previous_gradient_.clear();
            previous_direction_.clear();
            ++failed_searches_;
            return;
        }

        weights = weights + candidate.alpha * direction;
        previous_gradient_ = current.gradient;
        previous_direction_ = direction;
        ++accepted_steps_;
    }

    std::size_t failed_searches() const noexcept { return failed_searches_; }
    std::size_t accepted_steps() const noexcept { return accepted_steps_; }

private:
    struct Candidate {
        double alpha = 0.0;
        TrainResult value;
        double slope = 0.0;
        bool accepted = false;
    };

    static bool finite(const TrainResult& result) {
        return std::isfinite(result.loss) &&
            std::all_of(result.gradient.begin(), result.gradient.end(), [](double value) {
                return std::isfinite(value);
            });
    }

    static Candidate evaluate(double alpha, const Vector& weights, const Vector& direction,
                              const Objective& objective) {
        Candidate result;
        result.alpha = alpha;
        result.value = objective(weights + alpha * direction);
        if (result.value.gradient.size() != weights.size())
            throw std::invalid_argument("CG gradient size changed");
        result.slope = finite(result.value) ? dot(result.value.gradient, direction)
                                            : std::numeric_limits<double>::infinity();
        return result;
    }

    static Candidate zoom(Candidate low, Candidate high, const Vector& weights,
                          const Vector& direction, const TrainResult& initial,
                          double slope, const Objective& objective) {
        for (int iteration = 0; iteration < 80; ++iteration) {
            const double alpha = 0.5 * (low.alpha + high.alpha);
            if (alpha == low.alpha || alpha == high.alpha) break;
            auto trial = evaluate(alpha, weights, direction, objective);
            if (!finite(trial.value) || trial.value.loss > initial.loss + 1e-4 * alpha * slope ||
                trial.value.loss >= low.value.loss) {
                high = std::move(trial);
            } else {
                if (std::abs(trial.slope) <= -0.4 * slope) {
                    trial.accepted = true;
                    return trial;
                }
                if (trial.slope * (high.alpha - low.alpha) >= 0.0) high = low;
                low = std::move(trial);
            }
        }
        return {};
    }

    static Candidate search(const Vector& weights, const Vector& direction,
                            const TrainResult& initial, double slope,
                            const Objective& objective) {
        Candidate previous{0.0, initial, slope, false};
        double alpha = 1.0;
        for (int iteration = 0; iteration < 50; ++iteration) {
            auto trial = evaluate(alpha, weights, direction, objective);
            if (!finite(trial.value) || trial.value.loss > initial.loss + 1e-4 * alpha * slope ||
                (iteration > 0 && trial.value.loss >= previous.value.loss))
                return zoom(previous, trial, weights, direction, initial, slope, objective);
            if (std::abs(trial.slope) <= -0.4 * slope) {
                trial.accepted = true;
                return trial;
            }
            if (trial.slope >= 0.0) return zoom(trial, previous, weights, direction, initial, slope, objective);
            previous = std::move(trial);
            alpha *= 2.0;
        }
        return {};
    }

    Vector previous_gradient_;
    Vector previous_direction_;
    std::size_t failed_searches_ = 0;
    std::size_t accepted_steps_ = 0;
};

}
