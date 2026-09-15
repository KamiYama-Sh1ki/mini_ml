#include <stdexcept>

#include <ml/runtime/serial.hpp>

namespace ml {

TrainResult SerialBackend::compute_loss_and_gradient(const PolynomialRegression& model, const Dataset1D& data, const MSELoss& loss) const {
    if (data.empty()) throw std::invalid_argument("SerialBackend: dataset must not be empty");

    TrainResult result;
    result.gradient.assign(model.weights().size(), 0.0);

    for (const auto& sample : data.samples) {
        double prediction = model.predict(sample.x);
        result.loss += loss.value(prediction, sample.y);

        double prediction_gradient = loss.gradient(prediction, sample.y);
        double x_power = 1.0;
        for (double& parameter_gradient : result.gradient) {
            parameter_gradient += prediction_gradient * x_power;
            x_power *= sample.x;
        }
    }

    double inverse_size = 1.0 / static_cast<double>(data.size());
    result.loss *= inverse_size;
    for (double& parameter_gradient : result.gradient) parameter_gradient *= inverse_size;
    return result;
}

}
