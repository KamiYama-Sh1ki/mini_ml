#include <bits/stdc++.h>

#include <lab/common/lab_args.hpp>
#include <lab/common/lab_hooks.hpp>
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
    const double squared_norm = std::accumulate(std::next(weights.begin()), weights.end(), 0.0,
                                                [](double sum, double weight) { return sum + weight * weight; });
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
        run.write_samples("samples.csv", data.samples);
        {
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
        auto penalty = [&decay](const std::vector<double>& w) { return l2_penalty(w, decay); };
        auto hooks = lab::csv_hooks(run, weights, weights.size(), full_objective, penalty);
        const ml::TrainResult result =
            trainer.train(weights, options.epochs, options.checkpoint_interval, hooks, &rng);

        run.write_curve("curve.csv", true_curve,
                        [&weights](double x) {
                            double y = 0.0;
                            for (double coefficient : std::views::reverse(weights)) y = y * x + coefficient;
                            return y;
                        },
                        curve_point_count, -1.0, 1.0);

        const double final_penalty = penalty(weights);
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
