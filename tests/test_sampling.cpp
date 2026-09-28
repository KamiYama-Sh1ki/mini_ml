#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <ml/data/sampling.hpp>

int main() {
    auto f = [](double x) { return x * x; };

    {
        auto data = ml::sample_curve(f, 0.0, 4.0, 5);

        assert(data.size() == 5);

        assert(std::abs(data[0].x - 0.0) < 1e-12);
        assert(std::abs(data[1].x - 1.0) < 1e-12);
        assert(std::abs(data[2].x - 2.0) < 1e-12);
        assert(std::abs(data[3].x - 3.0) < 1e-12);
        assert(std::abs(data[4].x - 4.0) < 1e-12);

        for (const auto& sample : data.samples) {
            assert(std::abs(sample.y - sample.x * sample.x) < 1e-12);
        }
    }

    {
        auto data = ml::sample_curve(f, 2.0, 6.0, 1);

        assert(data.size() == 1);
        assert(std::abs(data[0].x - 4.0) < 1e-12);
        assert(std::abs(data[0].y - 16.0) < 1e-12);
    }

    {
        ml::RandomGenerator rng(42);
        auto data = ml::sample_curve_random(f, -3.0, 5.0, 1000, false, rng);

        assert(data.size() == 1000);

        for (const auto& sample : data.samples) {
            assert(sample.x >= -3.0);
            assert(sample.x <= 5.0);
            assert(std::abs(sample.y - sample.x * sample.x) < 1e-12);
        }
    }

    {
        ml::RandomGenerator rng(42);
        auto data = ml::sample_curve_random(f, -3.0, 5.0, 1000, true, rng);

        for (std::size_t i = 1; i < data.size(); ++i) {
            assert(data[i - 1].x <= data[i].x);
        }
    }

    {
        ml::RandomGenerator rng1(42);
        ml::RandomGenerator rng2(42);

        auto a = ml::sample_curve_random(f, -1.0, 1.0, 100, false, rng1);
        auto b = ml::sample_curve_random(f, -1.0, 1.0, 100, false, rng2);

        assert(a.size() == b.size());

        for (std::size_t i = 0; i < a.size(); ++i) {
            assert(a[i].x == b[i].x);
            assert(a[i].y == b[i].y);
        }
    }

    {
        ml::RandomGenerator rng(42);
        auto data = ml::sample_curve_noisy(f, -2.0, 2.0, 100, 0.0, false, rng);

        for (const auto& sample : data.samples) {
            assert(std::abs(sample.y - sample.x * sample.x) < 1e-12);
        }
    }

    {
        bool thrown = false;

        try {
            ml::sample_curve(f, 0.0, 1.0, 0);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }

        assert(thrown);
    }

    {
        bool thrown = false;

        try {
            ml::sample_curve(f, 1.0, -1.0, 10);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }

        assert(thrown);
    }

    {
        ml::RandomGenerator rng(42);
        bool thrown = false;

        try {
            ml::sample_curve_noisy(f, -1.0, 1.0, 10, -0.1, false, rng);
        } catch (const std::invalid_argument&) {
            thrown = true;
        }

        assert(thrown);
    }

    std::cout << "test_sampling passed\n";
}
