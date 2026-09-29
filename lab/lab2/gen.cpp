#include <bits/stdc++.h>

#include <lab/common/lab_args.hpp>
#include <lab/common/lab_hooks.hpp>
#include <lab/common/lab_output.hpp>
#include <ml/core/random.hpp>
#include <ml/opt/optimizer.hpp>
#include <ml/runtime/trainer.hpp>

namespace {

struct Sample {
    std::vector<double> x;
    int label;
};

double sigmoid(double z) {
    if (z >= 0.0) return 1.0 / (1.0 + std::exp(-z));
    const double e = std::exp(z);
    return e / (1.0 + e);
}

double softplus(double z) {
    return z > 30.0 ? z : std::log1p(std::exp(z));
}

double dot_augmented(const std::vector<double>& w, const std::vector<double>& x) {
    double sum = w[0];
    for (std::size_t i = 0; i < x.size(); ++i) sum += w[i + 1] * x[i];
    return sum;
}

double penalty_value(const std::vector<double>& w, double decay) {
    double squared = 0.0;
    for (std::size_t i = 1; i < w.size(); ++i) squared += w[i] * w[i];
    return 0.5 * decay * squared;
}

ml::TrainResult evaluate(const std::vector<double>& w, const std::vector<Sample>& data, double decay) {
    ml::TrainResult result;
    result.gradient.assign(w.size(), 0.0);
    for (const Sample& sample : data) {
        const double z = dot_augmented(w, sample.x);
        result.loss += softplus(z) - sample.label * z;
        const double residual = sigmoid(z) - sample.label;
        result.gradient[0] += residual;
        for (std::size_t i = 0; i < sample.x.size(); ++i) result.gradient[i + 1] += residual * sample.x[i];
    }
    const double inverse = 1.0 / static_cast<double>(data.size());
    result.loss *= inverse;
    for (double& g : result.gradient) g *= inverse;
    result.loss += penalty_value(w, decay);
    for (std::size_t i = 1; i < w.size(); ++i) result.gradient[i] += decay * w[i];
    return result;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        int dim = 2;
        int per_class = 60;
        double sigma = 0.3;
        std::uint64_t seed = 42;
        std::size_t epochs = 200;
        double learning_rate = 0.1;
        double weight_decay = 0.0;
        std::size_t checkpoint_interval = 50;
        lab::ArgParser args;
        args.add(lab::arg("--dim", dim, false, "feature dimension"));
        args.add(lab::arg("--samples", per_class, false, "samples per class", {.config_key = "per_class"}));
        args.add(lab::arg("--sigma", sigma, true, "gaussian noise stddev"));
        args.add(lab::arg("--seed", seed, true, "random seed"));
        args.add(lab::arg("--epochs", epochs, false, "number of training epochs"));
        args.add(lab::arg("--learning-rate", learning_rate, false, "sgd learning rate", {.config_key = "learning_rate", .in_summary = false}));
        args.add(lab::arg("--weight-decay", weight_decay, true, "L2 weight decay, >= 0, w0 exempt", {.config_key = "weight_decay"}));
        args.add(lab::arg("--save-every", checkpoint_interval, false, "checkpoint interval", {.config_key = "checkpoint_interval"}));
        if (args.parse(argc, argv)) {
            args.print_usage(std::cout, argv[0]);
            return 0;
        }

        lab::RunOutput run("lab/lab2/output");
        ml::RandomGenerator rng(seed);

        std::vector<std::vector<double>> centers(2, std::vector<double>(dim));
        for (std::vector<double>& center : centers)
            for (int i = 0; i < dim; ++i) center[i] = rng.uniform_double(-1.0, 1.0);

        std::vector<Sample> data;
        for (int label = 0; label < 2; ++label)
            for (int i = 0; i < per_class; ++i) {
                std::vector<double> x(dim);
                for (int j = 0; j < dim; ++j) x[j] = centers[label][j] + rng.normal(0.0, sigma);
                data.push_back(Sample{std::move(x), label});
            }

        {
            auto samples_file = run.csv("samples.csv");
            samples_file << "label";
            for (int i = 0; i < dim; ++i) samples_file << ",x" << i;
            samples_file << '\n';
            for (const Sample& sample : data) {
                samples_file << sample.label;
                for (double coordinate : sample.x) samples_file << ',' << coordinate;
                samples_file << '\n';
            }
            auto centers_file = run.csv("centers.csv");
            centers_file << "label";
            for (int i = 0; i < dim; ++i) centers_file << ",x" << i;
            centers_file << '\n';
            for (int label = 0; label < 2; ++label) {
                centers_file << label;
                for (int i = 0; i < dim; ++i) centers_file << ',' << centers[label][i];
                centers_file << '\n';
            }
            auto config_file = run.csv("config.csv");
            args.write_config(config_file);
        }

        const double decay = weight_decay;
        auto make_objective = [&data, decay](const std::vector<Sample>& subset) {
            return ml::Objective([subset, decay](const std::vector<double>& w) { return evaluate(w, subset, decay); });
        };
        ml::Objective full_objective = make_objective(data);
        std::vector<ml::Objective> sample_objectives;
        for (const Sample& sample : data) sample_objectives.push_back(make_objective({sample}));

        std::vector<double> weights(dim + 1, 0.0);
        ml::SGD optimizer(learning_rate);
        ml::Trainer trainer(full_objective, std::move(sample_objectives), optimizer, ml::Sampling::per_sample);
        auto penalty = [&decay](const std::vector<double>& w) { return penalty_value(w, decay); };
        auto hooks = lab::csv_hooks(run, weights, weights.size(), full_objective, penalty);
        const ml::TrainResult result = trainer.train(weights, epochs, checkpoint_interval, hooks, &rng);

        std::size_t correct = 0;
        for (const Sample& sample : data)
            if ((sigmoid(dot_augmented(weights, sample.x)) >= 0.5 ? 1 : 0) == sample.label) ++correct;

        std::cout << "final loss: " << result.loss - penalty(weights) << '\n'
                  << "accuracy: " << static_cast<double>(correct) / static_cast<double>(data.size()) << '\n';
        args.print_values(std::cout);
        std::cout << "learning rate: " << learning_rate << '\n'
                  << "CSV output: " << std::filesystem::absolute(run.directory()).string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "lab2 gen failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
