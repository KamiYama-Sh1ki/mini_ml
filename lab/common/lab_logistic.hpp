#pragma once

#include <utility>

#include <lab/common/lab_data.hpp>
#include <ml/model/logistic_regression.hpp>

namespace lab {

using LogisticRegression = ml::LogisticRegression;
using ml::l2_penalty;
using ml::sigmoid;
using ml::softplus;

inline ml::TrainResult evaluate_logistic(const ml::Vector& weights, const Dataset& data, double decay) {
    return ml::evaluate_logistic(weights, data.X, data.t, decay);
}

inline ml::Objective make_logistic_objective(Dataset data, double decay) {
    return ml::make_logistic_objective(std::move(data.X), std::move(data.t), decay);
}

}
