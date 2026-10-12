#pragma once

#include <chrono>
#include <memory>
#include <lab/common/lab_experiment.hpp>
#include <lab/common/lab_nonlinear_cg.hpp>

namespace lab {

inline std::unique_ptr<ml::Optimizer> logistic_optimizer(const std::string& name, double learning_rate) {
    if (name == "cg") return std::make_unique<NonlinearConjugateGradient>();
    if (name == "newton") return std::make_unique<ml::NewtonCG>();
    return std::make_unique<ml::SGD>(learning_rate);
}

struct TrainingSummary {
    std::size_t epochs = 0;
    std::size_t objective_calls = 0;
    std::size_t sample_updates = 0;
    double elapsed_seconds = 0.0;
    double gradient_norm = 0.0;
    double weight_norm = 0.0;
    double objective_loss = 0.0;
    std::string status;
};

inline TrainingSummary train_logistic(LogisticRegression& model, const Dataset& data, ml::Optimizer& optimizer,
                                      bool per_sample, double decay, std::size_t epochs, std::size_t save_every,
                                      double tolerance, std::uint64_t seed, const RunOutput& run) {
    if (epochs == 0 || save_every == 0) throw std::invalid_argument("epochs and save interval must be positive");
    TrainingSummary summary;
    auto full = make_logistic_objective(data, decay);
    ml::Objective counted = [&full, &summary](const ml::Vector& w) {
        ++summary.objective_calls;
        return full(w);
    };
    std::vector<ml::Objective> samples;
    if (per_sample) {
        for (std::size_t i = 0; i < data.size(); ++i) {
            Dataset single{ml::Matrix(1, data.features()), ml::Vector{data.t[i]}};
            single.X.set_row(0, data.X.row(i));
            auto objective = make_logistic_objective(std::move(single), decay);
            samples.push_back([objective = std::move(objective), &summary](const ml::Vector& w) {
                ++summary.objective_calls;
                return objective(w);
            });
        }
    }
    std::vector<std::size_t> order(data.size());
    std::iota(order.begin(), order.end(), 0);
    ml::RandomGenerator rng(seed);
    auto loss = run.csv("loss.csv");
    auto checkpoints = run.csv("checkpoints.csv");
    loss << "epoch,loss,l2_penalty,objective_loss,gradient_norm,weight_norm,elapsed_seconds,objective_calls,sample_updates\n";
    checkpoints << "epoch,loss,l2_penalty,objective_loss";
    for (std::size_t j = 0; j < model.weights().size(); ++j) checkpoints << ",w" << j;
    for (std::size_t j = 0; j < model.weights().size(); ++j) checkpoints << ",gradient" << j;
    checkpoints << '\n';
    const auto start = std::chrono::steady_clock::now();
    auto record = [&](std::size_t epoch) {
        const auto value = full(model.weights());
        if (!std::isfinite(value.loss)) throw std::runtime_error("training produced a non-finite loss");
        summary.epochs = epoch;
        summary.elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        summary.gradient_norm = std::sqrt(ml::dot(value.gradient, value.gradient));
        summary.weight_norm = std::sqrt(ml::dot(model.weights(), model.weights()));
        summary.objective_loss = value.loss;
        const double penalty = l2_penalty(model.weights(), decay);
        loss << epoch << ',' << value.loss - penalty << ',' << penalty << ',' << value.loss << ','
             << summary.gradient_norm << ',' << summary.weight_norm << ',' << summary.elapsed_seconds << ','
             << summary.objective_calls << ',' << summary.sample_updates << '\n';
        const bool converged = !per_sample && summary.gradient_norm <= tolerance;
        if (epoch % save_every == 0 || epoch == epochs || converged) {
            checkpoints << epoch << ',' << value.loss - penalty << ',' << penalty << ',' << value.loss;
            for (double w : model.weights()) checkpoints << ',' << w;
            for (double g : value.gradient) checkpoints << ',' << g;
            checkpoints << '\n';
        }
        return converged;
    };
    bool converged = record(0);
    for (std::size_t epoch = 1; epoch <= epochs && !converged; ++epoch) {
        if (per_sample) {
            shuffle_indices(order, rng);
            for (auto index : order) optimizer.step(model.weights(), samples[index]);
            summary.sample_updates += order.size();
        } else {
            optimizer.step(model.weights(), counted);
        }
        converged = record(epoch);
    }
    summary.status = converged ? "gradient_tolerance" : "epoch_limit";
    if (auto* cg = dynamic_cast<NonlinearConjugateGradient*>(&optimizer); cg && cg->failed_searches() > 0 && !converged)
        summary.status = "line_search_failed";
    return summary;
}

inline void save_training(const RunOutput& run, const TrainingSummary& summary) {
    auto file = run.csv("training_summary.csv");
    file << "epochs,objective_calls,sample_updates,elapsed_seconds,gradient_norm,weight_norm,objective_loss,status\n"
         << summary.epochs << ',' << summary.objective_calls << ',' << summary.sample_updates << ','
         << summary.elapsed_seconds << ',' << summary.gradient_norm << ',' << summary.weight_norm << ','
         << summary.objective_loss << ',' << summary.status << '\n';
}

}
