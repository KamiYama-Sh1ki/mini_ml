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

std::vector<Sample> load_samples(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("failed to open " + path.string());
    std::string line;
    std::getline(file, line);
    std::vector<Sample> data;
    while (std::getline(file, line)) {
        std::istringstream row(line);
        std::string value;
        std::getline(row, value, ',');
        Sample sample;
        sample.label = std::stoi(value);
        while (std::getline(row, value, ',')) sample.x.push_back(std::stod(value));
        data.push_back(std::move(sample));
    }
    if (data.empty()) throw std::runtime_error("no samples in " + path.string());
    return data;
}

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
        std::string data_path;
        std::uint64_t seed = 42;
        std::size_t epochs = 200;
        double learning_rate = 0.1;
        double weight_decay = 0.0;
        std::size_t checkpoint_interval = 50;
        lab::ArgParser args;
        args.add(lab::arg("--data", data_path, "path to samples.csv (default: latest under lab/lab2/data)"));
        args.add(lab::arg("--seed", seed, true, "random seed"));
        args.add(lab::arg("--epochs", epochs, false, "number of training epochs"));
        args.add(lab::arg("--learning-rate", learning_rate, false, "sgd learning rate", {.config_key = "learning_rate", .in_summary = false}));
        args.add(lab::arg("--weight-decay", weight_decay, true, "L2 weight decay, >= 0, w0 exempt", {.config_key = "weight_decay"}));
        args.add(lab::arg("--save-every", checkpoint_interval, false, "checkpoint interval", {.config_key = "checkpoint_interval"}));
        if (args.parse(argc, argv)) {
            args.print_usage(std::cout, argv[0]);
            return 0;
        }

        std::filesystem::path resolved = data_path;
        if (resolved.empty()) {
            const std::filesystem::path base = "lab/lab2/data";
            std::vector<std::filesystem::path> candidates;
            for (const auto& entry : std::filesystem::directory_iterator(base))
                if (entry.is_directory() && std::filesystem::exists(entry.path() / "samples.csv"))
                    candidates.push_back(entry.path());
            if (candidates.empty()) throw std::runtime_error("no generated data under " + base.string() + ", run lab2_gen first");
            resolved = *std::max_element(candidates.begin(), candidates.end()) / "samples.csv";
        }

        lab::RunOutput run("lab/lab2/output");
        const std::vector<Sample> data = load_samples(resolved);
        const std::size_t weight_count = data.front().x.size() + 1;
        std::vector<double> weights(weight_count, 0.0);
        {
            auto config_file = run.csv("config.csv");
            args.write_config(config_file);
        }

        const double decay = weight_decay;
        auto make_objective = [decay](const std::vector<Sample>& subset) {
            return ml::Objective([subset, decay](const std::vector<double>& w) { return evaluate(w, subset, decay); });
        };
        ml::Objective full_objective = make_objective(data);
        std::vector<ml::Objective> sample_objectives;
        for (const Sample& sample : data) sample_objectives.push_back(make_objective({sample}));

        ml::RandomGenerator rng(seed);
        ml::SGD optimizer(learning_rate);
        ml::Trainer trainer(full_objective, std::move(sample_objectives), optimizer, ml::Sampling::per_sample);
        auto penalty = [&decay](const std::vector<double>& w) { return penalty_value(w, decay); };
        auto hooks = lab::csv_hooks(run, weights, weights.size(), full_objective, penalty);
        const ml::TrainResult result = trainer.train(weights, epochs, checkpoint_interval, hooks, &rng);

        std::size_t correct = 0;
        for (const Sample& sample : data)
            if ((sigmoid(dot_augmented(weights, sample.x)) >= 0.5 ? 1 : 0) == sample.label) ++correct;

        std::cout << "data: " << std::filesystem::absolute(resolved).string() << '\n'
                  << "final loss: " << result.loss - penalty(weights) << '\n'
                  << "accuracy: " << static_cast<double>(correct) / static_cast<double>(data.size()) << '\n';
        args.print_values(std::cout);
        std::cout << "learning rate: " << learning_rate << '\n'
                  << "CSV output: " << std::filesystem::absolute(run.directory()).string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "lab2 failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
