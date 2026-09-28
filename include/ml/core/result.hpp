#pragma once

#include <vector>

namespace ml {

struct TrainResult {
    double loss{};
    std::vector<double> gradient;
};

}
