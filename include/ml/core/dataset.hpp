//dataset.hpp
#pragma once

#include <vector>
#include <cstddef>
#include <ml/core/sample.hpp>

namespace ml {

struct Dataset1D {
    std::vector<Sample1D> samples;

    std::size_t size() const noexcept {
        return samples.size();
    }

    void resize(std::size_t size_) {
        samples.resize(size_);
        return ;
    }

    void reserve(std::size_t size_) {
        samples.reserve(size_);
        return ;
    }

    void push_back(const Sample1D &o) {
        samples.push_back(o);
    }

    bool empty() const noexcept {
        return samples.empty();
    }

    Sample1D& operator[] (std::size_t index) {
        return samples[index];
    }

    const Sample1D& operator[] (std::size_t index) const  {
        return samples[index];
    }
};

}
