#include <bits/stdc++.h>
#include <lab/common/lab_args.hpp>
#include <lab/common/lab_experiment.hpp>

namespace {

struct ClassModel {
    double prior = 0.5;
    ml::Vector mean;
    ml::Vector variance;
};

ClassModel estimate_class(const lab::Dataset& data, double label) {
    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < data.size(); ++i) if (data.t[i] == label) rows.push_back(i);
    if (rows.size() < 2) throw std::invalid_argument("NB requires two training samples per class");
    ClassModel model;
    model.prior = static_cast<double>(rows.size()) / data.size();
    model.mean.resize(data.features());
    model.variance.resize(data.features());
    for (std::size_t j = 0; j < data.features(); ++j) {
        ml::Vector column(rows.size());
        for (std::size_t k = 0; k < rows.size(); ++k) column[k] = data.X(rows[k], j);
        model.mean[j] = ml::mean_of(column);
        double squared = 0.0;
        for (double value : column) squared += (value - model.mean[j]) * (value - model.mean[j]);
        model.variance[j] = std::max(1e-9, squared / column.size());
    }
    return model;
}

double log_density(const ClassModel& model, const ml::Vector& x) {
    double value = std::log(model.prior);
    for (std::size_t j = 0; j < x.size(); ++j)
        value -= 0.5 * (std::log(2.0 * std::numbers::pi * model.variance[j]) +
            (x[j] - model.mean[j]) * (x[j] - model.mean[j]) / model.variance[j]);
    return value;
}

ml::Vector logits(const std::array<ClassModel, 2>& models, const lab::Dataset& data) {
    ml::Vector result(data.size());
    for (std::size_t i = 0; i < data.size(); ++i)
        result[i] = log_density(models[1], data.X.row(i)) - log_density(models[0], data.X.row(i));
    return result;
}

}

int main(int argc, char* argv[]) {
    try {
        std::string data_path;
        std::string split_directory;
        std::string output_root = "lab/lab2/output";
        std::uint64_t seed = 42;
        double test_ratio = 0.2;
        int standardize = 0;
        lab::ArgParser args;
        args.add(lab::arg("--data", data_path, "CSV dataset (default: latest generated data)"));
        args.add(lab::arg("--split-dir", split_directory, "shared train/test indices", {.config_key = "split_dir"}));
        args.add(lab::arg("--output-root", output_root, "output base directory", {.config_key = "output_root"}));
        args.add(lab::arg("--seed", seed, true, "random seed"));
        args.add(lab::arg("--test-ratio", test_ratio, true, "held-out fraction", {.config_key = "test_ratio"}));
        args.add(lab::arg("--standardize", standardize, true, "fit scaling on training data: 0 or 1"));
        if (args.parse(argc, argv)) {
            args.print_usage(std::cout, argv[0]);
            return 0;
        }
        if (standardize != 0 && standardize != 1) throw std::invalid_argument("standardize must be 0 or 1");
        data_path = std::filesystem::absolute(data_path.empty() ? lab::latest_data("lab/lab2/data") : std::filesystem::path(data_path)).string();
        const auto data = lab::prepare_experiment(lab::load_dataset(data_path), test_ratio, seed, split_directory, standardize == 1);
        const auto start = std::chrono::steady_clock::now();
        const std::array<ClassModel, 2> models{estimate_class(data.train, 0.0), estimate_class(data.train, 1.0)};
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        lab::RunOutput run(output_root);
        lab::save_experiment(run, data);
        auto params = run.csv("nb_params.csv");
        params << "label,prior";
        for (std::size_t j = 0; j < data.train.features(); ++j) params << ",mean" << j;
        for (std::size_t j = 0; j < data.train.features(); ++j) params << ",var" << j;
        params << '\n';
        for (std::size_t label = 0; label < models.size(); ++label) {
            params << label << ',' << models[label].prior;
            for (double value : models[label].mean) params << ',' << value;
            for (double value : models[label].variance) params << ',' << value;
            params << '\n';
        }
        auto config = run.csv("config.csv");
        args.write_config(config);
        auto metrics = run.csv("metrics.csv");
        lab::write_metrics_header(metrics);
        for (const auto& [name, subset] : std::vector<std::pair<std::string, const lab::Dataset*>>{{"train", &data.train}, {"test", &data.test}}) {
            if (subset->size() == 0) continue;
            const auto result = lab::classification_metrics(logits(models, *subset), subset->t);
            lab::write_metrics(metrics, name, result);
            std::cout << name << " accuracy: " << result.confusion.accuracy() << " log loss: " << result.log_loss << '\n';
        }
        auto timing = run.csv("training_summary.csv");
        timing << "elapsed_seconds,status\n" << elapsed << ",closed_form\n";
        std::cout << "CSV output: " << std::filesystem::absolute(run.directory()).string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "lab2 nb failed: " << error.what() << '\n';
        return 1;
    }
}
