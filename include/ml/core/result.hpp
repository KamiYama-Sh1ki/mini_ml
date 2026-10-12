#pragma once

#include <ml/core/linalg.hpp>

namespace ml {

struct TrainResult {
    double loss{};
    Vector gradient;
};

}
