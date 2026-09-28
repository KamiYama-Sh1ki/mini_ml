#pragma once

#include <ml/core/dataset.hpp>
#include <ml/core/result.hpp>
#include <ml/opt/loss.hpp>

namespace ml {

class SerialBackend {
public:
    // Evaluates the averaged MSE loss and its gradient at arbitrary weights
    // using Horner evaluation; the model object is not needed.
    TrainResult compute_loss_and_gradient(const std::vector<double>& weights, const Dataset1D& data, const MSELoss& loss) const;
};

}
