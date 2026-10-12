#pragma once

#include <array>
#include <numeric>
#include <lab/common/lab_data.hpp>
#include <lab/common/lab_logistic.hpp>
#include <lab/common/lab_metrics.hpp>
#include <lab/common/lab_output.hpp>

namespace lab {

struct SplitIndices { std::vector<std::size_t> train; std::vector<std::size_t> test; };

inline void shuffle_indices(std::vector<std::size_t>& indices, ml::RandomGenerator& rng) {
    for (std::size_t i = indices.size(); i > 1; --i)
        std::swap(indices[i - 1], indices[static_cast<std::size_t>(rng.uniform_int(0, static_cast<std::int64_t>(i) - 1))]);
}

inline SplitIndices stratified_indices(const Dataset& data, double ratio, std::uint64_t seed) {
    if (!(ratio >= 0.0 && ratio < 1.0)) throw std::invalid_argument("test ratio must be in [0,1)");
    ml::RandomGenerator rng(seed * 6364136223846793005ULL + 1442695040888963407ULL);
    SplitIndices result;
    for (double label : {0.0, 1.0}) {
        std::vector<std::size_t> group;
        for (std::size_t i = 0; i < data.size(); ++i) if (data.t[i] == label) group.push_back(i);
        if (group.size() < 2 || (ratio > 0.0 && group.size() < 3)) throw std::invalid_argument("each class needs at least two training samples and a held-out sample");
        shuffle_indices(group, rng);
        const std::size_t count = ratio == 0.0 ? 0 : std::clamp(static_cast<std::size_t>(ratio * group.size()), std::size_t{1}, group.size() - 2);
        result.test.insert(result.test.end(), group.begin(), group.begin() + count);
        result.train.insert(result.train.end(), group.begin() + count, group.end());
    }
    shuffle_indices(result.train, rng);
    shuffle_indices(result.test, rng);
    return result;
}

inline void write_indices(std::ostream& file, const std::vector<std::size_t>& indices) {
    file << "index\n";
    for (auto index : indices) file << index << '\n';
}

inline std::vector<std::size_t> read_indices(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open split indices: " + path.string());
    std::string line;
    std::getline(file, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "index") throw std::invalid_argument("split indices require an index header");
    std::vector<std::size_t> result;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.front() == '-') throw std::invalid_argument("invalid split index");
        std::size_t consumed = 0;
        const auto index = std::stoull(line, &consumed);
        if (consumed != line.size()) throw std::invalid_argument("invalid split index");
        result.push_back(static_cast<std::size_t>(index));
    }
    return result;
}

inline SplitIndices get_indices(const Dataset& data, double ratio, std::uint64_t seed, const std::string& directory) {
    auto result = directory.empty() ? stratified_indices(data, ratio, seed)
        : SplitIndices{read_indices(std::filesystem::path(directory) / "train_indices.csv"), read_indices(std::filesystem::path(directory) / "test_indices.csv")};
    std::vector<bool> seen(data.size(), false);
    for (const auto* group : {&result.train, &result.test}) {
        std::array<std::size_t, 2> counts{};
        for (auto index : *group) {
            if (index >= data.size() || seen[index]) throw std::invalid_argument("split indices overlap or exceed dataset size");
            seen[index] = true;
            if (data.t[index] != 0.0 && data.t[index] != 1.0) throw std::invalid_argument("labels must be 0 or 1");
            ++counts[static_cast<std::size_t>(data.t[index])];
        }
        if (group == &result.train && (counts[0] < 2 || counts[1] < 2)) throw std::invalid_argument("training split requires two samples per class");
        if (group == &result.test && !group->empty() && (counts[0] == 0 || counts[1] == 0)) throw std::invalid_argument("test split requires both classes");
    }
    if (!std::all_of(seen.begin(), seen.end(), [](bool x) { return x; })) throw std::invalid_argument("split indices must partition the complete dataset");
    return result;
}

inline Dataset select_rows(const Dataset& data, const std::vector<std::size_t>& indices) {
    Dataset result{ml::Matrix(indices.size(), data.features()), ml::Vector(indices.size())};
    for (std::size_t i = 0; i < indices.size(); ++i) { result.X.set_row(i, data.X.row(indices[i])); result.t[i] = data.t[indices[i]]; }
    return result;
}

struct PreparedExperiment { SplitIndices indices; Dataset raw_train; Dataset raw_test; Dataset train; Dataset test; Standardizer standardizer; };

inline PreparedExperiment prepare_experiment(const Dataset& full, double ratio, std::uint64_t seed, const std::string& split_directory, bool standardize) {
    PreparedExperiment result;
    result.indices = get_indices(full, ratio, seed, split_directory);
    result.raw_train = select_rows(full, result.indices.train);
    result.raw_test = select_rows(full, result.indices.test);
    result.train = result.raw_train;
    result.test = result.raw_test;
    result.standardizer = standardize ? fit_standardizer(result.train) : Standardizer{ml::Vector(full.features(), 0.0), ml::Vector(full.features(), 1.0)};
    apply(result.standardizer, result.train);
    apply(result.standardizer, result.test);
    return result;
}

inline void save_experiment(const RunOutput& run, const PreparedExperiment& data) {
    auto train_indices = run.csv("train_indices.csv"); auto test_indices = run.csv("test_indices.csv");
    write_indices(train_indices, data.indices.train); write_indices(test_indices, data.indices.test);
    auto train_samples = run.csv("train_samples.csv"); auto test_samples = run.csv("test_samples.csv");
    write_dataset(train_samples, data.raw_train); write_dataset(test_samples, data.raw_test);
    auto scaling = run.csv("standardizer.csv"); scaling << "feature,mean,scale\n";
    for (std::size_t j = 0; j < data.standardizer.mean.size(); ++j) scaling << j << ',' << data.standardizer.mean[j] << ',' << data.standardizer.scale[j] << '\n';
}

struct ClassificationMetrics { ConfusionMatrix confusion; double log_loss = 0.0; double precision = 0.0; double recall = 0.0; double f1 = 0.0; };

inline ClassificationMetrics classification_metrics(const ml::Vector& logits, const ml::Vector& truth) {
    if (logits.size() != truth.size() || truth.empty()) throw std::invalid_argument("metrics require nonempty matching vectors");
    ml::Vector prediction(logits.size()); ClassificationMetrics result;
    for (std::size_t i = 0; i < logits.size(); ++i) { if (!std::isfinite(logits[i])) throw std::runtime_error("non-finite prediction"); prediction[i] = logits[i] >= 0.0 ? 1.0 : 0.0; result.log_loss += softplus(logits[i]) - truth[i] * logits[i]; }
    result.log_loss /= truth.size(); result.confusion = confusion(prediction, truth);
    const double tp = result.confusion.true_positive, fp = result.confusion.false_positive, fn = result.confusion.false_negative;
    result.precision = tp + fp > 0 ? tp / (tp + fp) : 0.0; result.recall = tp + fn > 0 ? tp / (tp + fn) : 0.0; result.f1 = 2 * tp + fp + fn > 0 ? 2 * tp / (2 * tp + fp + fn) : 0.0;
    return result;
}

inline void write_metrics_header(std::ostream& file) { file << "split,samples,accuracy,log_loss,precision,recall,f1,tn,fp,fn,tp\n"; }
inline void write_metrics(std::ostream& file, const std::string& split, const ClassificationMetrics& result) {
    const auto& c = result.confusion; file << split << ',' << c.total() << ',' << c.accuracy() << ',' << result.log_loss << ',' << result.precision << ',' << result.recall << ',' << result.f1 << ',' << c.true_negative << ',' << c.false_positive << ',' << c.false_negative << ',' << c.true_positive << '\n';
}

}
