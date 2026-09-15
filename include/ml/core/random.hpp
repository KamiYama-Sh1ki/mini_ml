//random.hpp
#pragma once

#include <cstdint>
#include <random>
#include <stdexcept>

namespace ml {

class RandomGenerator {
public:
    explicit RandomGenerator(std::uint64_t seed = 42)
        : engine_(seed) {}

    double uniform_double(double left, double right) {
        if (left > right) {
            throw std::invalid_argument(
                "RandomGenerator::uniform: left must be <= right"
            );
        }

        std::uniform_real_distribution<double> dist(left, right);
        return dist(engine_);
    }

    std::int64_t uniform_int(std::int64_t left,std::int64_t right) {
        if (left > right) {
            throw std::invalid_argument(
                "RandomGenerator::uniform_int: left must be <= right"
            );
        }

        std::uniform_int_distribution<std::int64_t> dist(left,right);
        return dist(engine_);
    }

    double normal(double mean, double stddev) {
        if (stddev < 0.0) {
            throw std::invalid_argument(
                "RandomGenerator::normal: stddev must be >= 0"
            );
        }
        if (stddev == 0.0) return mean;

        std::normal_distribution<double> dist(mean,stddev);

        return dist(engine_);
    }

    void reseed(std::uint64_t seed) {
        engine_.seed(seed);
    }

private:
    std::mt19937_64 engine_;
};

} // namespace ml