#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <ml/optimizer/conjugate_gradient.hpp>
#include <ml/optimizer/sgd.hpp>

namespace {

double quadratic_value(const std::vector<double>& weights) {
    return 0.5 * (weights[0] * weights[0] + 100.0 * weights[1] * weights[1]) -
           weights[0] - 20.0 * weights[1];
}

std::vector<double> quadratic_gradient(const std::vector<double>& weights) {
    return {weights[0] - 1.0, 100.0 * weights[1] - 20.0};
}

double distance_to_optimum(const std::vector<double>& weights) {
    const double dx = weights[0] - 1.0;
    const double dy = weights[1] - 0.2;
    return std::sqrt(dx * dx + dy * dy);
}

}  // namespace

int main() {
    {
        ml::ConjugateGradient optimizer(0.01);
        std::vector<double> weights{0.0, 0.0};
        for (int step = 0; step < 20; ++step) {
            optimizer.step(weights, quadratic_gradient(weights));
        }
        assert(distance_to_optimum(weights) < 1e-8);
    }

    {
        ml::ConjugateGradient optimizer(0.01);
        optimizer.reset();
        std::vector<double> weights{0.0, 0.0};
        optimizer.step(weights, quadratic_gradient(weights));
        assert(distance_to_optimum(weights) < 1.0);
        try {
            std::vector<double> mismatched{0.0, 0.0, 0.0};
            optimizer.step(mismatched, quadratic_gradient(weights));
            assert(false);
        } catch (const std::invalid_argument&) {
        }
    }

    {
        try {
            ml::ConjugateGradient invalid(0.0);
            (void)invalid;
            assert(false);
        } catch (const std::invalid_argument&) {
        }
        try {
            ml::ConjugateGradient invalid(0.01, -1.0);
            (void)invalid;
            assert(false);
        } catch (const std::invalid_argument&) {
        }
    }

    {
        ml::SGD sgd(0.01);
        std::vector<double> sgd_weights{0.0, 0.0};
        for (int step = 0; step < 100; ++step) {
            sgd.step(sgd_weights, quadratic_gradient(sgd_weights));
        }

        ml::ConjugateGradient optimizer(0.01);
        std::vector<double> cg_weights{0.0, 0.0};
        for (int step = 0; step < 100; ++step) {
            optimizer.step(cg_weights, quadratic_gradient(cg_weights));
        }

        assert(distance_to_optimum(cg_weights) < distance_to_optimum(sgd_weights));
    }

    std::cout << "test_conjugate_gradient passed\n";
}
