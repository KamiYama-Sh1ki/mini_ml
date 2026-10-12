#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ml/core/linalg.hpp>
#include <ml/core/random.hpp>

namespace lab {

struct Dataset {
    ml::Matrix X;
    ml::Vector t;
    std::size_t size() const noexcept { return X.rows; }
    std::size_t features() const noexcept { return X.cols; }
};

inline void write_dataset(std::ostream& file, const Dataset& data) {
    file << "label";
    for (std::size_t j = 0; j < data.features(); ++j) file << ",x" << j;
    file << '\n';
    for (std::size_t i = 0; i < data.size(); ++i) {
        file << data.t[i];
        for (std::size_t j = 0; j < data.features(); ++j) file << ',' << data.X(i, j);
        file << '\n';
    }
}

inline Dataset load_dataset(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("failed to open " + path.string());
    std::string line;
    std::getline(file, line);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.starts_with("\xEF\xBB\xBF")) line.erase(0, 3);
    if (!line.starts_with("label,")) throw std::invalid_argument("dataset header must start with label,");
    std::vector<std::vector<double>> rows;
    std::vector<double> labels;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.back() == ',') throw std::invalid_argument("empty row or missing feature");
        std::istringstream row(line);
        std::string value;
        std::getline(row, value, ',');
        auto parse = [](const std::string& text) {
            std::size_t consumed = 0;
            const double result = std::stod(text, &consumed);
            if (consumed != text.size() || !std::isfinite(result)) throw std::invalid_argument("dataset values must be finite numbers");
            return result;
        };
        const double label = parse(value);
        if (label != 0.0 && label != 1.0) throw std::invalid_argument("dataset labels must be 0 or 1");
        labels.push_back(label);
        std::vector<double> features;
        while (std::getline(row, value, ',')) features.push_back(parse(value));
        if (features.empty() || (!rows.empty() && features.size() != rows.front().size()))
            throw std::invalid_argument("dataset rows must have the same nonzero feature count");
        rows.push_back(std::move(features));
    }
    if (rows.empty()) throw std::runtime_error("no samples in " + path.string());
    Dataset data;
    data.X = ml::Matrix(rows.size(), rows.front().size());
    for (std::size_t i = 0; i < rows.size(); ++i) data.X.set_row(i, ml::Vector::from(std::move(rows[i])));
    data.t = ml::Vector::from(std::move(labels));
    return data;
}

inline std::filesystem::path latest_data(const std::filesystem::path& base) {
    std::vector<std::filesystem::path> candidates;
    for (const auto& entry : std::filesystem::directory_iterator(base))
        if (entry.is_directory() && std::filesystem::exists(entry.path() / "samples.csv"))
            candidates.push_back(entry.path() / "samples.csv");
    if (candidates.empty()) throw std::runtime_error("no generated data under " + base.string());
    return *std::max_element(candidates.begin(), candidates.end());
}

inline std::pair<Dataset, Dataset> train_test_split(const Dataset& data, double test_ratio,
                                                    ml::RandomGenerator& rng) {
    std::vector<std::size_t> order(data.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    for (std::size_t i = order.size(); i > 1; --i) {
        const std::size_t j = static_cast<std::size_t>(rng.uniform_int(0, static_cast<std::int64_t>(i) - 1));
        std::swap(order[i - 1], order[j]);
    }
    const std::size_t test_count = static_cast<std::size_t>(test_ratio * static_cast<double>(data.size()));
    const std::size_t train_count = data.size() - test_count;
    Dataset train;
    Dataset test;
    train.X = ml::Matrix(train_count, data.features());
    test.X = ml::Matrix(test_count, data.features());
    train.t = ml::Vector(train_count);
    test.t = ml::Vector(test_count);
    for (std::size_t i = 0; i < order.size(); ++i) {
        Dataset& target = i < train_count ? train : test;
        const std::size_t index = i < train_count ? i : i - train_count;
        target.X.set_row(index, data.X.row(order[i]));
        target.t[index] = data.t[order[i]];
    }
    return {std::move(train), std::move(test)};
}

struct Standardizer {
    ml::Vector mean;
    ml::Vector scale;

    ml::Vector transform(const ml::Vector& x) const {
        ml::Vector result(x.size());
        for (std::size_t j = 0; j < x.size(); ++j) result[j] = (x[j] - mean[j]) / scale[j];
        return result;
    }
};

inline Standardizer fit_standardizer(const Dataset& data) {
    Standardizer standardizer;
    standardizer.mean.resize(data.features());
    standardizer.scale.resize(data.features());
    for (std::size_t j = 0; j < data.features(); ++j) {
        ml::Vector column(data.size());
        for (std::size_t i = 0; i < data.size(); ++i) column[i] = data.X(i, j);
        standardizer.mean[j] = ml::mean_of(column);
        const double scale = ml::stddev_of(column);
        standardizer.scale[j] = scale > 1e-12 ? scale : 1.0;
    }
    return standardizer;
}

inline void apply(Standardizer& standardizer, Dataset& data) {
    for (std::size_t i = 0; i < data.size(); ++i)
        data.X.set_row(i, standardizer.transform(data.X.row(i)));
}

}  // namespace lab
