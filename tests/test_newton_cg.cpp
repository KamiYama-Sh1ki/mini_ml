#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ml/opt/optimizer.hpp>

namespace {

ml::TrainResult ridge_quadratic(const std::vector<double>& w) {
    const double dx = w[0] - 1.0;
    const double dy = w[1] - 0.5;
    return ml::TrainResult{0.5 * (dx * dx + 50.0 * dy * dy), {dx, 50.0 * dy}};
}

double sigmoid(double z) {
    return z >= 0.0 ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z));
}

ml::TrainResult logistic(const std::vector<double>& w) {
    const std::vector<std::vector<double>> x{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {2.0, -1.0}};
    const std::vector<int> t{0, 1, 1, 0, 1};
    ml::TrainResult result{0.0, {0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < x.size(); ++i) {
        double z = w[0];
        for (std::size_t j = 0; j < 2; ++j) z += w[j + 1] * x[i][j];
        result.loss += (z > 0.0 ? z + std::log1p(std::exp(-z)) : std::log1p(std::exp(z))) - t[i] * z;
        const double residual = sigmoid(z) - t[i];
        result.gradient[0] += residual;
        for (std::size_t j = 0; j < 2; ++j) result.gradient[j + 1] += residual * x[i][j];
    }
    const double inverse = 1.0 / static_cast<double>(x.size());
    result.loss *= inverse;
    for (double& g : result.gradient) g *= inverse;
    return result;
}

std::vector<double> analytic_hessian_vector(const std::vector<double>& w, const std::vector<double>& v) {
    const std::vector<std::vector<double>> x{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}, {2.0, -1.0}};
    std::vector<double> augmented(3);
    std::vector<double> hv(3, 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        for (int j = 0; j < 3; ++j) augmented[j] = j == 0 ? 1.0 : x[i][static_cast<std::size_t>(j) - 1];
        double z = 0.0;
        double dot = 0.0;
        for (int j = 0; j < 3; ++j) {
            z += w[static_cast<std::size_t>(j)] * augmented[static_cast<std::size_t>(j)];
            dot += v[static_cast<std::size_t>(j)] * augmented[static_cast<std::size_t>(j)];
        }
        const double weight = sigmoid(z) * (1.0 - sigmoid(z)) / static_cast<double>(x.size());
        for (int j = 0; j < 3; ++j) hv[static_cast<std::size_t>(j)] += weight * dot * augmented[static_cast<std::size_t>(j)];
    }
    return hv;
}

}  // namespace

int main() {
    {
        ml::NewtonCG optimizer;
        std::vector<double> w{0.0, 0.0};
        double previous = ridge_quadratic(w).loss;
        for (int step = 0; step < 10; ++step) {
            optimizer.step(w, ridge_quadratic);
            const double loss = ridge_quadratic(w).loss;
            assert(loss <= previous + 1e-12);
            previous = loss;
        }
        assert(std::abs(w[0] - 1.0) < 1e-8);
        assert(std::abs(w[1] - 0.5) < 1e-8);
    }

    {
        const std::vector<double> w{0.3, -0.7, 0.2};
        const ml::TrainResult at_w = logistic(w);
        const std::vector<double> v{0.1, -0.4, 0.9};
        const double epsilon = 1.0e-6;
        std::vector<double> shifted = w;
        for (std::size_t i = 0; i < w.size(); ++i) shifted[i] += epsilon * v[i];
        const ml::TrainResult at_shifted = logistic(shifted);
        std::vector<double> finite_difference(w.size());
        for (std::size_t i = 0; i < w.size(); ++i) finite_difference[i] = (at_shifted.gradient[i] - at_w.gradient[i]) / epsilon;
        const std::vector<double> analytic = analytic_hessian_vector(w, v);
        double scale = 0.0;
        double error = 0.0;
        for (std::size_t i = 0; i < w.size(); ++i) {
            error += (finite_difference[i] - analytic[i]) * (finite_difference[i] - analytic[i]);
            scale += analytic[i] * analytic[i];
        }
        assert(std::sqrt(error / scale) < 1e-5);
    }

    {
        ml::NewtonCG optimizer(0.1, 20);
        std::vector<double> w{0.0, 0.0, 0.0};
        double previous = logistic(w).loss;
        for (int step = 0; step < 50; ++step) {
            optimizer.step(w, logistic);
            const double loss = logistic(w).loss;
            assert(loss <= previous + 1e-12);
            previous = loss;
        }
        const ml::TrainResult result = logistic(w);
        assert(result.loss < 0.7);
        double norm = 0.0;
        for (double g : result.gradient) norm += g * g;
        assert(norm < 1e-10);
    }

    {
        bool thrown = false;
        try {
            static_cast<void>(ml::NewtonCG(0.0));
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);

        ml::NewtonCG optimizer;
        std::vector<double> w{0.0, 0.0};
        thrown = false;
        try {
            optimizer.step(w, nullptr);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);
    }

    std::cout << "test_newton_cg passed\n";
}
