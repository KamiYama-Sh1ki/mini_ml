#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <lab/common/lab_args.hpp>
#include <lab/common/lab_training.hpp>
#include <ml/model/logistic_regression.hpp>
#include <ml/opt/nonlinear_conjugate_gradient.hpp>

int main(int argc, char* argv[]) {
    try {
        std::string data_path;
        std::string split_directory;
        std::string output_root = "lab/lab2/output";
        std::string optimizer_name = "sgd";
        std::uint64_t seed = 42;
        std::size_t epochs = 300;
        std::size_t save_every = 1;
        int standardize = 0;
        double learning_rate = 0.01;
        double decay = 0.0;
        double test_ratio = 0.2;
        double tolerance = 1e-7;

        lab::ArgParser args;
        args.add(lab::arg("--optimizer", optimizer_name, "sgd, nonlinear cg (Wolfe), or newton"));
        args.add(lab::arg("--data", data_path, "CSV dataset (default: latest generated dataset)"));
        args.add(lab::arg("--split-dir", split_directory, "directory containing shared split indices", {.config_key = "split_dir"}));
        args.add(lab::arg("--output-root", output_root, "output base directory", {.config_key = "output_root"}));
        args.add(lab::arg("--seed", seed, true, "random seed"));
        args.add(lab::arg("--epochs", epochs, false, "maximum epochs"));
        args.add(lab::arg("--weight-decay", decay, true, "L2 penalty, bias exempt", {.config_key = "weight_decay"}));
        args.add(lab::arg("--test-ratio", test_ratio, true, "held-out fraction", {.config_key = "test_ratio"}));
        args.add(lab::arg("--standardize", standardize, true, "1: fit scaling on training data; 0: raw features"));
        args.add(lab::arg("--tolerance", tolerance, false, "full-batch gradient tolerance"));
        args.add(lab::arg("--save-every", save_every, false, "checkpoint interval", {.config_key = "checkpoint_interval"}));

        for (int i = 1; i + 1 < argc; ++i)
            if (std::string_view(argv[i]) == "--optimizer") optimizer_name = argv[i + 1];
        if (optimizer_name == "sgd")
            args.add(lab::arg("--learning-rate", learning_rate, false, "SGD learning rate", {.config_key = "learning_rate"}));

        if (args.parse(argc, argv)) {
            args.print_usage(std::cout, argv[0]);
            return 0;
        }
        if (optimizer_name != "sgd" && optimizer_name != "cg" && optimizer_name != "newton")
            throw std::invalid_argument("optimizer must be sgd, cg, or newton");
        if (standardize != 0 && standardize != 1) throw std::invalid_argument("standardize must be 0 or 1");
        if (!(test_ratio >= 0.0 && test_ratio < 1.0)) throw std::invalid_argument("test ratio must be in [0,1)");

        const auto source = data_path.empty() ? lab::latest_data("lab/lab2/data") : std::filesystem::path(data_path);
        const auto full = lab::load_dataset(std::filesystem::absolute(source));
        const auto data = lab::prepare_experiment(full, test_ratio, seed, split_directory, standardize == 1);

        lab::RunOutput run(output_root);
        lab::save_experiment(run, data);
        auto config = run.csv("config.csv");
        args.write_config(config);
        config.close();

        ml::LogisticRegression model(data.train.features());
        auto optimizer = lab::logistic_optimizer(optimizer_name, learning_rate);
        const bool per_sample = optimizer_name == "sgd";
        const auto training = lab::train_logistic(model, data.train, *optimizer, per_sample, decay,
                                                  epochs, save_every, tolerance, seed, run);
        lab::save_training(run, training);

        auto weights = run.csv("weights.csv");
        weights << "index,weight\n";
        for (std::size_t j = 0; j < model.weights().size(); ++j)
            weights << j << ',' << model.weights()[j] << '\n';

        auto metrics = run.csv("metrics.csv");
        lab::write_metrics_header(metrics);
        for (const auto& [name, subset] : std::vector<std::pair<std::string, const lab::Dataset*>>{
                 {"train", &data.train}, {"test", &data.test}}) {
            if (subset->size() == 0) continue;
            const auto result = lab::classification_metrics(model.logits(subset->X), subset->t);
            lab::write_metrics(metrics, name, result);
            std::cout << name << " accuracy: " << result.confusion.accuracy()
                      << " log loss: " << result.log_loss << '\n';
        }
        if (data.test.size() == 0) std::cout << "test evaluation: unavailable (no held-out samples)\n";
        std::cout << "training status: " << training.status << " epochs: " << training.epochs << '\n'
                  << "CSV output: " << std::filesystem::absolute(run.directory()).string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "lab2 failed: " << error.what() << '\n';
        return 1;
    }
}
