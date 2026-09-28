#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ml/opt/optimizer.hpp>
#include <ml/runtime/serial.hpp>

int main() {
    ml::Dataset1D data{
        {
            {0.0, 2.0},
            {1.0, 5.0},
            {-1.0, 0.0}
        }
    };

    ml::MSELoss loss;
    assert(std::abs(loss.value(4.0, 1.0) - 4.5) < 1e-12);
    assert(std::abs(loss.gradient(4.0, 1.0) - 3.0) < 1e-12);

    ml::SerialBackend backend;
    std::vector<double> weights{1.0, 2.0, 3.0};
    ml::TrainResult result = backend.compute_loss_and_gradient(weights, data, loss);

    assert(std::abs(result.loss - 1.0) < 1e-12);
    assert(result.gradient.size() == 3);
    assert(std::abs(result.gradient[0] - 2.0 / 3.0) < 1e-12);
    assert(std::abs(result.gradient[1] + 1.0 / 3.0) < 1e-12);
    assert(std::abs(result.gradient[2] - 1.0) < 1e-12);
    assert(weights == std::vector<double>({1.0, 2.0, 3.0}));

    bool thrown = false;
    try {
        backend.compute_loss_and_gradient(weights, ml::Dataset1D{}, loss);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown);

    thrown = false;
    try {
        backend.compute_loss_and_gradient({}, data, loss);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown);

    {
        ml::SGD optimizer(0.1);
        std::vector<double> updated{1.0, -2.0};
        ml::Objective constant_gradient = [](const std::vector<double>&) {
            return ml::TrainResult{0.0, {0.5, -0.25}};
        };
        optimizer.step(updated, constant_gradient);
        assert(std::abs(updated[0] - 0.95) < 1e-12);
        assert(std::abs(updated[1] + 1.975) < 1e-12);

        thrown = false;
        try {
            optimizer.step(updated, nullptr);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);

        thrown = false;
        try {
            ml::Objective mismatched = [](const std::vector<double>&) {
                return ml::TrainResult{0.0, {1.0}};
            };
            optimizer.step(updated, mismatched);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);
    }

    thrown = false;
    try {
        static_cast<void>(ml::SGD(0.0));
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown);

    std::cout << "test_backend passed\n";
}
