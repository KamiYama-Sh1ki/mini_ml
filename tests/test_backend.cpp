#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ml/opt/optimizer.hpp>
#include <ml/runtime/serial.hpp>

int main() {
    ml::PolynomialRegression model(2);
    model.weights() = {1.0, 2.0, 3.0};

    ml::Dataset1D data{
        {
            {0.0, 2.0},
            {1.0, 5.0},
            {-1.0, 0.0}
        }
    };

    ml::MSELoss loss;
    assert(std::abs(loss.value(4.0, 1.0) - 9.0) < 1e-12);
    assert(std::abs(loss.gradient(4.0, 1.0) - 6.0) < 1e-12);

    ml::SerialBackend backend;
    ml::TrainResult result = backend.compute_loss_and_gradient(model, data, loss);

    assert(std::abs(result.loss - 2.0) < 1e-12);
    assert(result.gradient.size() == 3);
    assert(std::abs(result.gradient[0] - 4.0 / 3.0) < 1e-12);
    assert(std::abs(result.gradient[1] + 2.0 / 3.0) < 1e-12);
    assert(std::abs(result.gradient[2] - 2.0) < 1e-12);
    assert(model.weights() == std::vector<double>({1.0, 2.0, 3.0}));

    bool thrown = false;
    try {
        backend.compute_loss_and_gradient(model, ml::Dataset1D{}, loss);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown);

    std::vector<double> weights{1.0, -2.0};
    ml::SGD optimizer(0.1, 0.01);
    optimizer.step(weights, {0.5, -0.25});
    assert(std::abs(weights[0] - 0.949) < 1e-12);
    assert(std::abs(weights[1] + 1.973) < 1e-12);

    thrown = false;
    try {
        static_cast<void>(ml::SGD(0.0));
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown);

    thrown = false;
    try {
        static_cast<void>(ml::SGD(0.1, -0.01));
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown);

    thrown = false;
    try {
        optimizer.step(weights, {1.0});
    } catch (const std::invalid_argument&) {
        thrown = true;
    }
    assert(thrown);

    std::cout << "test_backend passed\n";
}
