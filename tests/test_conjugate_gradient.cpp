#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ml/opt/optimizer.hpp>

namespace {

ml::TrainResult quadratic(const ml::Vector& weights) {
    const double dx = weights[0] - 1.0;
    const double dy = weights[1] - 0.2;
    return ml::TrainResult{0.5 * (dx * dx + 100.0 * dy * dy), ml::Vector{dx, 100.0 * dy}};
}

double distance_to_optimum(const ml::Vector& weights) {
    const double dx = weights[0] - 1.0;
    const double dy = weights[1] - 0.2;
    return std::sqrt(dx * dx + dy * dy);
}

}  // namespace

int main() {
    {
        ml::ConjugateGradient optimizer;
        ml::Vector weights{0.0, 0.0};
        for (int step = 0; step < 20; ++step) {
            optimizer.step(weights, quadratic);
        }
        assert(distance_to_optimum(weights) < 1e-8);
    }

    {
        ml::ConjugateGradient optimizer;
        optimizer.reset();
        ml::Vector weights{0.0, 0.0};
        optimizer.step(weights, quadratic);
        assert(distance_to_optimum(weights) < 1.0);

        bool thrown = false;
        try {
            optimizer.step(weights, nullptr);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);

        thrown = false;
        try {
            ml::Objective mismatched = [](const ml::Vector&) {
                return ml::TrainResult{0.0, ml::Vector{1.0}};
            };
            optimizer.step(weights, mismatched);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);
    }

    {
        ml::SGD sgd(0.01);
        ml::Vector sgd_weights{0.0, 0.0};
        for (int step = 0; step < 100; ++step) {
            sgd.step(sgd_weights, quadratic);
        }

        ml::ConjugateGradient optimizer;
        ml::Vector cg_weights{0.0, 0.0};
        for (int step = 0; step < 100; ++step) {
            optimizer.step(cg_weights, quadratic);
        }

        assert(distance_to_optimum(cg_weights) < distance_to_optimum(sgd_weights));
    }

    std::cout << "test_conjugate_gradient passed\n";
}
