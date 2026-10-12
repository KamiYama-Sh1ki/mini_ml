#include <bits/stdc++.h>

#include <lab/common/lab_args.hpp>
#include <lab/common/lab_data.hpp>
#include <lab/common/lab_output.hpp>
#include <ml/core/linalg.hpp>
#include <ml/core/random.hpp>

namespace {

using ml::operator+;
using ml::operator-;
using ml::operator*;

std::vector<ml::Vector> roll_centers(int dim, double center_min, double center_max,
                                     ml::RandomGenerator& rng) {
    std::vector<ml::Vector> centers;
    for (int label = 0; label < 2; ++label) {
        ml::Vector center(dim);
        for (int i = 0; i < dim; ++i) center[i] = rng.uniform_double(center_min, center_max);
        centers.push_back(std::move(center));
    }
    return centers;
}

ml::Vector roll_direction(int dim, ml::RandomGenerator& rng) {
    ml::Vector direction(dim);
    for (int i = 0; i < dim; ++i) direction[i] = rng.normal(0.0, 1.0);
    const double norm = std::sqrt(ml::dot(direction, direction));
    return (1.0 / norm) * direction;
}

lab::Dataset sample_gaussian(const std::vector<ml::Vector>& centers, int per_class, double sigma,
                        double elongation, const std::vector<ml::Vector>& directions,
                        ml::RandomGenerator& rng) {
    lab::Dataset data;
    data.X = ml::Matrix(2 * static_cast<std::size_t>(per_class), centers.front().size());
    for (int label = 0; label < 2; ++label)
        for (int i = 0; i < per_class; ++i) {
            ml::Vector z(centers[label].size());
            for (double& coordinate : z) coordinate = rng.normal(0.0, 1.0);
            const ml::Vector x =
                centers[label] + sigma * (z + (elongation - 1.0) * ml::dot(z, directions[label]) * directions[label]);
            data.X.set_row(label * per_class + i, x);
            data.t.push_back(label);
        }
    return data;
}

void write_labeled_vectors(std::ostream& file, const std::vector<ml::Vector>& vectors, std::string_view prefix) {
    file << "label";
    for (std::size_t j = 0; j < vectors.front().size(); ++j) file << "," << prefix << j;
    file << '\n';
    for (std::size_t label = 0; label < vectors.size(); ++label) {
        file << label;
        for (double coordinate : vectors[label]) file << ',' << coordinate;
        file << '\n';
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        int dim = 2;
        int per_class = 60;
        double sigma = 0.3;
        double elongation = 1.0;
        double center_min = -1.0;
        double center_max = 1.0;
        std::string covariance = "class-specific";
        std::uint64_t seed = 42;
        lab::ArgParser args;
        args.add(lab::arg("--dim", dim, false, "feature dimension"));
        args.add(lab::arg("--samples", per_class, false, "samples per class", {.config_key = "per_class"}));
        args.add(lab::arg("--sigma", sigma, true, "gaussian noise stddev"));
        args.add(lab::arg("--elong", elongation, true, "cluster stretch along a random orientation (1 = round)"));
        args.add(lab::arg("--min", center_min, "center roll lower bound", {.config_key = "center_min"}));
        args.add(lab::arg("--max", center_max, "center roll upper bound", {.config_key = "center_max"}));
        args.add(lab::arg("--seed", seed, true, "random seed"));
        args.add(lab::arg("--covariance", covariance, "shared or class-specific class directions"));
        if (args.parse(argc, argv)) {
            args.print_usage(std::cout, argv[0]);
            return 0;
        }
        if (center_min >= center_max) throw std::invalid_argument("--min must be less than --max");
        if (dim <= 0 || per_class <= 0 || sigma <= 0.0 || elongation < 1.0)
            throw std::invalid_argument("dim and samples must be positive, sigma > 0 and elong >= 1");
        if (covariance != "shared" && covariance != "class-specific")
            throw std::invalid_argument("covariance must be shared or class-specific");

        lab::RunOutput run("lab/lab2/data");
        ml::RandomGenerator rng(seed);
        const std::vector<ml::Vector> centers = roll_centers(dim, center_min, center_max, rng);
        std::vector<ml::Vector> directions = {roll_direction(dim, rng), roll_direction(dim, rng)};
        if (covariance == "shared") directions[1] = directions[0];
        const lab::Dataset data = sample_gaussian(centers, per_class, sigma, elongation, directions, rng);

        {
            auto samples_file = run.csv("samples.csv");
            lab::write_dataset(samples_file, data);
            auto centers_file = run.csv("centers.csv");
            write_labeled_vectors(centers_file, centers, "x");
            auto directions_file = run.csv("directions.csv");
            write_labeled_vectors(directions_file, directions, "u");
            auto config_file = run.csv("config.csv");
            args.write_config(config_file);
        }

        std::cout << "samples: " << data.size() << '\n'
                  << "center distance: " << std::sqrt(ml::dot(centers[0] - centers[1], centers[0] - centers[1])) << '\n';
        args.print_values(std::cout);
        std::cout << "CSV output: " << std::filesystem::absolute(run.directory()).string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "lab2 gen failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
