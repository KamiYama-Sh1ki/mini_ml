#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ml/opt/optimizer.hpp>

namespace {

// J(w) = 0.5 * ((w0 - 1)^2 + 100 * (w1 - 0.2)^2), optimum at (1, 0.2),
// condition number 100. Exact line search should finish it in ~2 steps.
ml::TrainResult quadratic(const std::vector<double>& weights) {
    const double dx = weights[0] - 1.0;
    const double dy = weights[1] - 0.2;
    return ml::TrainResult{0.5 * (dx * dx + 100.0 * dy * dy), {dx, 100.0 * dy}};
}

double distance_to_optimum(const std::vector<double>& weights) {
    const double dx = weights[0] - 1.0;
    const double dy = weights[1] - 0.2;
    return std::sqrt(dx * dx + dy * dy);
}

}  // namespace

int main() {
    {
        ml::ConjugateGradient optimizer;
        std::vector<double> weights{0.0, 0.0};
        for (int step = 0; step < 20; ++step) {
            optimizer.step(weights, quadratic);
        }
        assert(distance_to_optimum(weights) < 1e-8);
    }

    {
        ml::ConjugateGradient optimizer;
        optimizer.reset();
        std::vector<double> weights{0.0, 0.0};
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
            ml::Objective mismatched = [](const std::vector<double>&) {
                return ml::TrainResult{0.0, {1.0}};
            };
            optimizer.step(weights, mismatched);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);
    }

    {
        ml::SGD sgd(0.01);
        std::vector<double> sgd_weights{0.0, 0.0};
        for (int step = 0; step < 100; ++step) {
            sgd.step(sgd_weights, quadratic);
        }

        ml::ConjugateGradient optimizer;
        std::vector<double> cg_weights{0.0, 0.0};
        for (int step = 0; step < 100; ++step) {
            optimizer.step(cg_weights, quadratic);
        }

        assert(distance_to_optimum(cg_weights) < distance_to_optimum(sgd_weights));
    }

    std::cout << "test_conjugate_gradient passed\n";
}
