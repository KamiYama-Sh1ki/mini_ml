#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ml/core/random.hpp>
#include <ml/core/result.hpp>
#include <ml/opt/optimizer.hpp>

namespace ml {

// Library-side training loop. One epoch means one optimizer step in
// full-batch mode, or one pass over all sample objectives (in shuffled
// order) in per-sample mode. Hooks decide what gets logged and where.
enum class Sampling {
    full_batch,
    per_sample,
};

struct TrainHooks {
    std::function<void(std::size_t epoch, const TrainResult&)> on_epoch;
    std::function<void(std::size_t epoch, const std::vector<double>&)> on_checkpoint;
};

class Trainer {
public:
    Trainer(const Objective& full_objective, std::vector<Objective> sample_objectives,
            Optimizer& optimizer, Sampling sampling)
        : full_objective_(full_objective), sample_objectives_(std::move(sample_objectives)),
          optimizer_(optimizer), sampling_(sampling) {
        if (!full_objective_) throw std::invalid_argument("Trainer: full objective must not be empty");
        if (sampling_ == Sampling::per_sample && sample_objectives_.empty()) {
            throw std::invalid_argument("Trainer: per-sample sampling requires sample objectives");
        }
    }

    TrainResult train(std::vector<double>& weights, std::size_t epochs, std::size_t checkpoint_interval,
                      const TrainHooks& hooks, RandomGenerator* shuffle_rng = nullptr) {
        if (checkpoint_interval == 0) throw std::invalid_argument("Trainer: checkpoint_interval must be > 0");
        if (sampling_ == Sampling::per_sample && shuffle_rng == nullptr) {
            throw std::invalid_argument("Trainer: per-sample sampling requires a shuffle generator");
        }

        std::vector<std::size_t> order;
        if (sampling_ == Sampling::per_sample) {
            order.resize(sample_objectives_.size());
            for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        }

        TrainResult result;
        for (std::size_t epoch = 1;; ++epoch) {
            if (sampling_ == Sampling::per_sample) {
                shuffle(order, *shuffle_rng);
                for (std::size_t index : order) {
                    optimizer_.step(weights, sample_objectives_[index]);
                }
            } else {
                optimizer_.step(weights, full_objective_);
            }

            result = full_objective_(weights);
            if (hooks.on_epoch) hooks.on_epoch(epoch, result);

            if (hooks.on_checkpoint && (epoch % checkpoint_interval == 0 || epoch == epochs)) {
                hooks.on_checkpoint(epoch, weights);
            }

            if (epoch == epochs) break;
        }
        return result;
    }

private:
    static void shuffle(std::vector<std::size_t>& order, RandomGenerator& rng) {
        for (std::size_t i = order.size(); i > 1; --i) {
            const std::size_t j = static_cast<std::size_t>(rng.uniform_int(0, static_cast<std::int64_t>(i) - 1));
            std::swap(order[i - 1], order[j]);
        }
    }

    Objective full_objective_;
    std::vector<Objective> sample_objectives_;
    Optimizer& optimizer_;
    Sampling sampling_;
};

}
