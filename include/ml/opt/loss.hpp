#pragma once

namespace ml {

class MSELoss {
public:
    // 0.5 * (prediction - target)^2 convention, so the gradient is the raw
    // residual and reported losses sit on the J(w) = mean(0.5 e^2) scale.
    double value(double prediction, double target) const noexcept {
        double difference = prediction - target;
        return 0.5 * difference * difference;
    }

    double gradient(double prediction, double target) const noexcept {
        return prediction - target;
    }
};

}
