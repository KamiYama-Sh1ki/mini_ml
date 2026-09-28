#pragma once

#include <cstddef>
#include <fstream>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <lab/common/lab_output.hpp>
#include <ml/core/result.hpp>
#include <ml/runtime/trainer.hpp>

namespace lab {

inline ml::TrainHooks csv_hooks(const RunOutput& run, const std::vector<double>& weights,
                                std::size_t weight_count, const ml::Objective& full_objective,
                                const std::function<double(const std::vector<double>&)>& penalty) {
    auto loss_file = std::make_shared<std::ofstream>(run.csv("loss.csv"));
    auto checkpoints_file = std::make_shared<std::ofstream>(run.csv("checkpoints.csv"));
    *loss_file << "epoch,loss,l2_penalty,objective_loss\n";
    *checkpoints_file << "epoch,loss,l2_penalty,objective_loss";
    for (std::size_t i = 0; i < weight_count; ++i) *checkpoints_file << ",w" << i;
    for (std::size_t i = 0; i < weight_count; ++i) *checkpoints_file << ",gradient" << i;
    *checkpoints_file << '\n';

    ml::TrainHooks hooks;
    hooks.on_epoch = [loss_file, &weights, penalty](std::size_t epoch, const ml::TrainResult& result) {
        const double value = penalty(weights);
        *loss_file << epoch << ',' << result.loss - value << ',' << value << ',' << result.loss << '\n';
    };
    hooks.on_checkpoint = [checkpoints_file, &full_objective,
                           penalty](std::size_t epoch, const std::vector<double>& epoch_weights) {
        const ml::TrainResult result = full_objective(epoch_weights);
        const double value = penalty(epoch_weights);
        *checkpoints_file << epoch << ',' << result.loss - value << ',' << value << ',' << result.loss;
        for (double weight : epoch_weights) *checkpoints_file << ',' << weight;
        for (double gradient : result.gradient) *checkpoints_file << ',' << gradient;
        *checkpoints_file << '\n';
    };
    return hooks;
}

}  // namespace lab
