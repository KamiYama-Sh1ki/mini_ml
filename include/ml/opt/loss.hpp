#pragma once

namespace ml {

class MSELoss {
public:
    double value(double prediction, double target) const noexcept {
        double difference = prediction - target;
        return difference * difference;
    }

    double gradient(double prediction, double target) const noexcept {
        return 2.0 * (prediction - target);
    }
};

}
