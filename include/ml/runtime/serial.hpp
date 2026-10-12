#pragma once

#include <ml/core/dataset.hpp>
#include <ml/core/linalg.hpp>
#include <ml/core/result.hpp>
#include <ml/opt/loss.hpp>

namespace ml {

class SerialBackend {
public:
    TrainResult compute_loss_and_gradient(const Vector& weights, const Dataset1D& data, const MSELoss& loss) const;
};

}
