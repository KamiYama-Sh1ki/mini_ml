#pragma once

#include <ml/core/dataset.hpp>
#include <ml/core/result.hpp>
#include <ml/model/regression.hpp>
#include <ml/ops/loss.hpp>

namespace ml {

class SerialBackend {
public:
    TrainResult compute_loss_and_gradient(const PolynomialRegression& model, const Dataset1D& data, const MSELoss& loss) const;
};

}
