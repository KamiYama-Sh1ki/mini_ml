#include <iostream>
#include <lab/common/lab_training.hpp>

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        double largest_error = 0.0;
        lab::Dataset data{ml::Matrix(8, 1, 100.0), ml::Vector{0, 1, 1, 1, 1, 0, 0, 0}};
        for (std::size_t i = 4; i < 8; ++i) data.X(i, 0) = -100.0;
        for (double decay : {0.0, 0.01}) {
            const ml::Vector point{0.2, -0.003};
            const auto value = lab::evaluate_logistic(point, data, decay);
            for (std::size_t j = 0; j < point.size(); ++j) {
                auto plus = point;
                auto minus = point;
                plus[j] += 1e-6;
                minus[j] -= 1e-6;
                const double fd = (lab::evaluate_logistic(plus, data, decay).loss - lab::evaluate_logistic(minus, data, decay).loss) / 2e-6;
                largest_error = std::max(largest_error, std::abs(fd - value.gradient[j]));
                require(std::abs(fd - value.gradient[j]) < 1e-6, "production logistic gradient mismatch");
            }
        }
        for (double scale : {0.01, 1.0, 100.0}) {
            auto scaled = data;
            for (std::size_t i = 0; i < scaled.size(); ++i) scaled.X(i, 0) *= scale / 100.0;
            lab::NonlinearConjugateGradient optimizer;
            auto objective = lab::make_logistic_objective(scaled, 0.0);
            ml::Vector w{0.0, 0.0};
            double previous = objective(w).loss;
            for (int iteration = 0; iteration < 100; ++iteration) {
                optimizer.step(w, objective);
                const double loss = objective(w).loss;
                require(std::isfinite(loss) && loss <= previous + 1e-12, "nonlinear CG accepted an increasing step");
                previous = loss;
            }
            require(std::abs(previous - 0.5623351446188083) < 1e-9, "nonlinear CG did not reach finite optimum");
            std::cout << "CG scale=" << scale << " final_loss=" << previous << '\n';
        }
        require(lab::l2_penalty(ml::Vector{100, 2}, 0.1) == 0.2, "bias must be exempt from L2");
        require(std::isfinite(lab::softplus(1000)) && lab::sigmoid(-1000) == 0.0, "extreme logit stability");
        const auto split = lab::get_indices(data, 0.25, 7, "");
        require(split.train.size() == 6 && split.test.size() == 2, "stratified split size");
        const auto first = lab::prepare_experiment(data, 0.25, 7, "", true);
        auto changed = data;
        for (auto index : split.test) changed.X(index, 0) += 1e6;
        const auto second = lab::prepare_experiment(changed, 0.25, 7, "", true);
        require(first.standardizer.mean == second.standardizer.mean && first.standardizer.scale == second.standardizer.scale,
                "test data leaked into scaling");
        std::cout << "gradient max_error=" << largest_error << '\n';
        std::cout << "PASS: production gradients, three CG feature scales, bias penalty, stable sigmoid, stratification, no scaling leakage\n";
    } catch (const std::exception& error) {
        std::cerr << "verification failed: " << error.what() << '\n';
        return 1;
    }
}
