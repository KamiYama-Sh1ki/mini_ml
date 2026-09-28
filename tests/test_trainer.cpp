#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <ml/core/random.hpp>
#include <ml/opt/loss.hpp>
#include <ml/opt/optimizer.hpp>
#include <ml/runtime/serial.hpp>
#include <ml/runtime/trainer.hpp>

namespace {

ml::Dataset1D make_data() {
    return ml::Dataset1D{
        {
            {-1.0, 2.0},
            {-0.5, 1.75},
            {0.0, 1.0},
            {0.5, 2.75},
            {1.0, 6.0}
        }
    };
}

std::vector<double> ridge_closed_form(const ml::Dataset1D& data, double lambda) {
    const std::size_t n = data.size();
    const std::size_t m = 3;
    std::vector<double> a(m * m, 0.0);
    std::vector<double> b(m, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t k = 0; k < m; ++k) {
            double sum = 0.0;
            for (const auto& sample : data.samples) {
                double xj = 1.0;
                for (std::size_t s = 0; s < j; ++s) xj *= sample.x;
                double xk = 1.0;
                for (std::size_t s = 0; s < k; ++s) xk *= sample.x;
                sum += xj * xk;
            }
            a[j * m + k] = sum / static_cast<double>(n);
        }
        double sum = 0.0;
        for (const auto& sample : data.samples) {
            double xj = 1.0;
            for (std::size_t s = 0; s < j; ++s) xj *= sample.x;
            sum += sample.y * xj;
        }
        b[j] = sum / static_cast<double>(n);
    }
    for (std::size_t j = 1; j < m; ++j) a[j * m + j] += lambda;

    for (std::size_t column = 0; column < m; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < m; ++row) {
            if (std::abs(a[row * m + column]) > std::abs(a[pivot * m + column])) pivot = row;
        }
        for (std::size_t k = 0; k < m; ++k) std::swap(a[column * m + k], a[pivot * m + k]);
        std::swap(b[column], b[pivot]);
        for (std::size_t row = column + 1; row < m; ++row) {
            const double factor = a[row * m + column] / a[column * m + column];
            for (std::size_t k = column; k < m; ++k) a[row * m + k] -= factor * a[column * m + k];
            b[row] -= factor * b[column];
        }
    }
    std::vector<double> weights(m, 0.0);
    for (std::size_t i = m; i-- > 0;) {
        double value = b[i];
        for (std::size_t k = i + 1; k < m; ++k) value -= a[i * m + k] * weights[k];
        weights[i] = value / a[i * m + i];
    }
    return weights;
}

}  // namespace

int main() {
    const ml::Dataset1D data = make_data();
    const double lambda = 0.05;
    const std::vector<double> expected = ridge_closed_form(data, lambda);

    ml::MSELoss loss;
    ml::SerialBackend backend;
    const double decay = lambda;
    auto make_objective = [&backend, &loss, decay](const ml::Dataset1D& dataset) {
        return ml::Objective([&backend, &loss, dataset, decay](const std::vector<double>& w) {
            ml::TrainResult result = backend.compute_loss_and_gradient(w, dataset, loss);
            for (std::size_t i = 1; i < w.size(); ++i) {
                result.loss += 0.5 * decay * w[i] * w[i];
                result.gradient[i] += decay * w[i];
            }
            return result;
        });
    };
    ml::Objective full_objective = make_objective(data);
    std::vector<ml::Objective> sample_objectives;
    for (const auto& sample : data.samples) {
        sample_objectives.push_back(make_objective(ml::Dataset1D{{sample}}));
    }

    {
        ml::ConjugateGradient optimizer;
        ml::Trainer trainer(full_objective, {}, optimizer, ml::Sampling::full_batch);
        std::vector<double> weights(3, 0.0);
        trainer.train(weights, 50, 50, ml::TrainHooks{});
        for (std::size_t i = 0; i < weights.size(); ++i) {
            assert(std::abs(weights[i] - expected[i]) < 1e-8);
        }
    }

    {
        ml::SGD optimizer(0.01);
        ml::Trainer trainer(full_objective, sample_objectives, optimizer, ml::Sampling::per_sample);
        std::vector<double> weights(3, 0.0);
        std::size_t epoch_calls = 0;
        std::size_t checkpoint_calls = 0;
        ml::TrainHooks hooks;
        hooks.on_epoch = [&](std::size_t epoch, const ml::TrainResult&) {
            assert(epoch == epoch_calls + 1);
            ++epoch_calls;
        };
        hooks.on_checkpoint = [&](std::size_t, const std::vector<double>&) {
            ++checkpoint_calls;
        };
        ml::RandomGenerator rng(7);
        trainer.train(weights, 10, 5, hooks, &rng);
        assert(epoch_calls == 10);
        assert(checkpoint_calls == 2);
        assert(full_objective(weights).loss < full_objective(std::vector<double>(3, 0.0)).loss);
    }

    {
        bool thrown = false;
        try {
            ml::SGD optimizer(0.01);
            ml::Trainer trainer(full_objective, {}, optimizer, ml::Sampling::per_sample);
            static_cast<void>(trainer);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }
        assert(thrown);
    }

    std::cout << "test_trainer passed\n";
}
