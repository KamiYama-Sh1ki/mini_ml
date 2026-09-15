#pragma once

#include <algorithm>
#include <stdexcept>
#include <ml/core/dataset.hpp>
#include <ml/core/random.hpp>

namespace ml {

template<class F>
Dataset1D sample_curve(F&& func, double left, double right, int count) {
    if (count <= 0) throw std::invalid_argument("sample_curve: count must be > 0");
    if (left > right) throw std::invalid_argument("sample_curve: left must be <= right");

    Dataset1D res;
    res.samples.reserve(count);

    if (count == 1) {
        double x = (left + right) / 2.0;
        res.push_back({x, func(x)});
        return res;
    }

    for (int i = 0; i < count; ++i) {
        double x = left + (right - left) * static_cast<double>(i) / static_cast<double>(count - 1);
        res.push_back({x, func(x)});
    }

    return res;
}

template<class F>
Dataset1D sample_curve_random(F&& func, double left, double right, int count, bool keep_order, RandomGenerator& rng) {
    if (count <= 0) throw std::invalid_argument("sample_curve_random: count must be > 0");
    if (left > right) throw std::invalid_argument("sample_curve_random: left must be <= right");

    Dataset1D res;
    res.samples.reserve(count);

    for (int i = 0; i < count; ++i) {
        double x = rng.uniform_double(left, right);
        res.push_back({x, func(x)});
    }

    if (keep_order) {
        std::sort(res.samples.begin(), res.samples.end(), [](const Sample1D& a, const Sample1D& b) {
            return a.x < b.x;
        });
    }

    return res;
}

template<class F>
Dataset1D sample_curve_noisy(F&& func, double left, double right, int count, double stddev, bool keep_order, RandomGenerator& rng) {
    if (count <= 0) throw std::invalid_argument("sample_curve_noisy: count must be > 0");
    if (left > right) throw std::invalid_argument("sample_curve_noisy: left must be <= right");
    if (stddev < 0.0) throw std::invalid_argument("sample_curve_noisy: stddev must be >= 0");

    Dataset1D res;
    res.samples.reserve(count);

    for (int i = 0; i < count; ++i) {
        double x = rng.uniform_double(left, right);
        double y = func(x) + rng.normal(0.0, stddev);
        res.push_back({x, y});
    }

    if (keep_order) {
        std::sort(res.samples.begin(), res.samples.end(), [](const Sample1D& a, const Sample1D& b) {
            return a.x < b.x;
        });
    }

    return res;
}

}