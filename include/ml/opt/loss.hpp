#pragma once

namespace ml {

class MSELoss {
public:
    double value(double prediction, double target) const noexcept {
        double difference = prediction - target;
        return 0.5 * difference * difference;
    }

    double gradient(double prediction, double target) const noexcept {
        return prediction - target;
    }
};

}
