#include <bits/stdc++.h>

#include <lab/common/lab_args.hpp>
#include <lab/common/lab_output.hpp>
#include <ml/core/random.hpp>

namespace {

struct Sample {
    std::vector<double> x;
    int label;
};

std::vector<std::vector<double>> roll_centers(int dim, ml::RandomGenerator& rng) {
    std::vector<std::vector<double>> centers(2, std::vector<double>(dim));
    for (std::vector<double>& center : centers)
        for (int i = 0; i < dim; ++i) center[i] = rng.uniform_double(-1.0, 1.0);
    return centers;
}

std::vector<Sample> sample_gaussian(const std::vector<std::vector<double>>& centers, int per_class, double sigma,
                                    ml::RandomGenerator& rng) {
    std::vector<Sample> data;
    for (int label = 0; label < 2; ++label)
        for (int i = 0; i < per_class; ++i) {
            std::vector<double> x(centers[label].size());
            for (std::size_t j = 0; j < x.size(); ++j) x[j] = centers[label][j] + rng.normal(0.0, sigma);
            data.push_back(Sample{std::move(x), label});
        }
    return data;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        int dim = 2;
        int per_class = 60;
        double sigma = 0.3;
        std::uint64_t seed = 42;
        lab::ArgParser args;
        args.add(lab::arg("--dim", dim, false, "feature dimension"));
        args.add(lab::arg("--samples", per_class, false, "samples per class", {.config_key = "per_class"}));
        args.add(lab::arg("--sigma", sigma, true, "gaussian noise stddev"));
        args.add(lab::arg("--seed", seed, true, "random seed"));
        if (args.parse(argc, argv)) {
            args.print_usage(std::cout, argv[0]);
            return 0;
        }

        lab::RunOutput run("lab/lab2/data");
        ml::RandomGenerator rng(seed);
        const std::vector<std::vector<double>> centers = roll_centers(dim, rng);
        const std::vector<Sample> data = sample_gaussian(centers, per_class, sigma, rng);

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

        double distance = 0.0;
        for (int i = 0; i < dim; ++i) distance += (centers[0][i] - centers[1][i]) * (centers[0][i] - centers[1][i]);
        std::cout << "samples: " << data.size() << '\n'
                  << "center distance: " << std::sqrt(distance) << '\n';
        args.print_values(std::cout);
        std::cout << "CSV output: " << std::filesystem::absolute(run.directory()).string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "lab2 gen failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
