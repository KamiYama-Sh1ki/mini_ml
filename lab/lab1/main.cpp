#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include <lab/common/lab_args.hpp>
#include <lab/common/lab_output.hpp>
#include <ml/core/random.hpp>
#include <ml/data/sampling.hpp>
#include <ml/opt/loss.hpp>
#include <ml/opt/optimizer.hpp>
#include <ml/runtime/serial.hpp>
#include <ml/runtime/trainer.hpp>

namespace {

constexpr int curve_point_count = 400;

struct Options {
    std::string optimizer = "sgd";
    std::uint64_t seed = 42;
    int sample_count = 9;
    std::size_t degree = 15;
    double noise_stddev = 0.1;
    std::size_t epochs = 10000;
    double learning_rate = 0.01;
    double weight_decay = 0.0;
    std::size_t checkpoint_interval = 1000;
};

double true_curve(double x) {
    return std::sin(std::numbers::pi_v<double> * x);
}

double l2_penalty(const std::vector<double>& weights, double weight_decay) {
    double squared_norm = 0.0;
    for (std::size_t i = 1; i < weights.size(); ++i) squared_norm += weights[i] * weights[i];
    return 0.5 * weight_decay * squared_norm;
}

std::unique_ptr<ml::Optimizer> make_optimizer(const Options& options) {
    if (options.optimizer == "cg") return std::make_unique<ml::ConjugateGradient>();
    return std::make_unique<ml::SGD>(options.learning_rate);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        Options options;
        lab::ArgParser args;
        args.add(lab::arg("--optimizer", options.optimizer, {"sgd", "cg"}, "sgd (per-sample) or cg (full-batch)"));
        args.add(lab::arg("--seed", options.seed, true, "random seed"));
        args.add(lab::arg("--samples", options.sample_count, false, "number of noisy samples", {.config_key = "sample_count"}));
        args.add(lab::arg("--degree", options.degree, true, "polynomial degree, including zero"));
        args.add(lab::arg("--noise", options.noise_stddev, true, "noise standard deviation, >= 0", {.config_key = "noise_stddev"}));
        args.add(lab::arg("--epochs", options.epochs, false, "number of training epochs"));
        args.add(lab::arg("--learning-rate", options.learning_rate, false, "positive learning rate, used by sgd only", {.config_key = "learning_rate", .in_summary = false}));
        args.add(lab::arg("--weight-decay", options.weight_decay, true, "L2 weight decay, >= 0, w0 exempt", {.config_key = "weight_decay"}));
        args.add(lab::arg("--save-every|--checkpoint-interval", options.checkpoint_interval, false, "save a checkpoint every N epochs", {.config_key = "checkpoint_interval"}));
        args.add(lab::constant("curve_point_count", curve_point_count));
        if (args.parse(argc, argv)) {
            args.print_usage(std::cout, argv[0]);
            return 0;
        }

        lab::RunOutput run("lab/lab1/output");
        ml::RandomGenerator rng(options.seed);
        ml::Dataset1D data =
            ml::sample_curve_noisy(true_curve, -1.0, 1.0, options.sample_count, options.noise_stddev, true, rng);
        std::vector<double> weights(options.degree + 1, 0.0);

        {
            auto samples_file = run.csv("samples.csv");
            samples_file << "x,y\n";
            for (const auto& sample : data.samples) samples_file << sample.x << ',' << sample.y << '\n';
            auto config_file = run.csv("config.csv");
            args.write_config(config_file);
        }

        ml::MSELoss loss;
        ml::SerialBackend backend;
        const double decay = options.weight_decay;
        auto make_objective = [&backend, &loss, decay](const ml::Dataset1D& dataset) {
            return ml::Objective([&backend, &loss, dataset, decay](const std::vector<double>& w) {
                ml::TrainResult result = backend.compute_loss_and_gradient(w, dataset, loss);
                result.loss += l2_penalty(w, decay);
                for (std::size_t i = 1; i < w.size(); ++i) result.gradient[i] += decay * w[i];
                return result;
            });
        };
        ml::Objective full_objective = make_objective(data);
        std::vector<ml::Objective> sample_objectives;
        for (const auto& sample : data.samples) sample_objectives.push_back(make_objective(ml::Dataset1D{{sample}}));

        auto optimizer = make_optimizer(options);
        const ml::Sampling sampling = options.optimizer == "cg" ? ml::Sampling::full_batch : ml::Sampling::per_sample;
        ml::Trainer trainer(full_objective, std::move(sample_objectives), *optimizer, sampling);

        ml::TrainResult result;
        {
            auto loss_file = run.csv("loss.csv");
            auto checkpoints_file = run.csv("checkpoints.csv");
            loss_file << "epoch,loss,l2_penalty,objective_loss\n";
            checkpoints_file << "epoch,loss,l2_penalty,objective_loss";
            for (std::size_t i = 0; i < weights.size(); ++i) checkpoints_file << ",w" << i;
            for (std::size_t i = 0; i < weights.size(); ++i) checkpoints_file << ",gradient" << i;
            checkpoints_file << '\n';

            ml::TrainHooks hooks;
            hooks.on_epoch = [&](std::size_t epoch, const ml::TrainResult& epoch_result) {
                const double penalty = l2_penalty(weights, decay);
                loss_file << epoch << ',' << epoch_result.loss - penalty << ',' << penalty << ','
                          << epoch_result.loss << '\n';
            };
            hooks.on_checkpoint = [&](std::size_t epoch, const std::vector<double>& epoch_weights) {
                const ml::TrainResult epoch_result = full_objective(epoch_weights);
                const double penalty = l2_penalty(epoch_weights, decay);
                checkpoints_file << epoch << ',' << epoch_result.loss - penalty << ',' << penalty << ','
                                 << epoch_result.loss;
                for (double weight : epoch_weights) checkpoints_file << ',' << weight;
                for (double value : epoch_result.gradient) checkpoints_file << ',' << value;
                checkpoints_file << '\n';
            };
            result = trainer.train(weights, options.epochs, options.checkpoint_interval, hooks, &rng);
        }

        {
            auto curve_file = run.csv("curve.csv");
            curve_file << "x,y_true,y_pred\n";
            for (int i = 0; i < curve_point_count; ++i) {
                const double x = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(curve_point_count - 1);
                double y_pred = 0.0;
                for (auto it = weights.rbegin(); it != weights.rend(); ++it) y_pred = y_pred * x + *it;
                curve_file << x << ',' << true_curve(x) << ',' << y_pred << '\n';
            }
        }

        const double final_penalty = l2_penalty(weights, decay);
        std::cout << "final loss: " << result.loss - final_penalty << '\n'
                  << "final L2 penalty: " << final_penalty << '\n'
                  << "final objective loss: " << result.loss << '\n';
        args.print_values(std::cout);
        if (options.optimizer == "sgd") std::cout << "learning rate: " << options.learning_rate << '\n';
        std::cout << "CSV output: " << std::filesystem::absolute(run.directory()).string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "lab1 failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
