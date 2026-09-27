#include <chrono>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <ml/core/random.hpp>
#include <ml/data/sampling.hpp>
#include <ml/model/regression.hpp>
#include <ml/opt/loss.hpp>
#include <ml/opt/optimizer.hpp>
#include <ml/runtime/serial.hpp>

namespace {

constexpr std::uint64_t default_seed = 42;
constexpr int default_sample_count = 100;
constexpr std::size_t default_degree = 15;
constexpr double default_noise_stddev = 0.1;
constexpr std::size_t default_epochs = 10000;
constexpr double default_learning_rate = 0.01;
constexpr double default_weight_decay = 0.0;
constexpr std::size_t default_checkpoint_interval = 5;
constexpr int curve_point_count = 400;

struct Options {
    std::uint64_t seed = default_seed;
    int sample_count = default_sample_count;
    std::size_t degree = default_degree;
    double noise_stddev = default_noise_stddev;
    std::size_t epochs = default_epochs;
    double learning_rate = default_learning_rate;
    double weight_decay = default_weight_decay;
    std::size_t checkpoint_interval = default_checkpoint_interval;
    bool show_help = false;
};

template <typename UnsignedInteger>
UnsignedInteger parse_unsigned_integer(std::string_view option, std::string_view value, bool allow_zero) {
    UnsignedInteger parsed_value = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed_value);
    if (error != std::errc{} || end != value.data() + value.size() || (!allow_zero && parsed_value == 0)) {
        const std::string requirement = allow_zero ? "a non-negative integer" : "a positive integer";
        throw std::invalid_argument(std::string(option) + " requires " + requirement + ", got '" +
                                    std::string(value) + "'");
    }
    return parsed_value;
}

double parse_finite_number(std::string_view option, std::string_view value, bool allow_zero) {
    double parsed_value = 0.0;
    const auto [end, error] =
        std::from_chars(value.data(), value.data() + value.size(), parsed_value, std::chars_format::general);
    const bool invalid_sign = allow_zero ? parsed_value < 0.0 : parsed_value <= 0.0;
    if (error != std::errc{} || end != value.data() + value.size() || !std::isfinite(parsed_value) ||
        invalid_sign) {
        const std::string requirement = allow_zero ? "a finite non-negative number" : "a finite positive number";
        throw std::invalid_argument(std::string(option) + " requires " + requirement + ", got '" +
                                    std::string(value) + "'");
    }
    return parsed_value;
}

Options parse_options(int argc, char* argv[]) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option = argv[i];
        if (option == "-h" || option == "--help") {
            options.show_help = true;
            continue;
        }

        if (option != "--samples" && option != "--degree" && option != "--noise" &&
            option != "--epochs" && option != "--learning-rate" && option != "--weight-decay" &&
            option != "--seed" && option != "--save-every" && option != "--checkpoint-interval") {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
        if (++i >= argc) throw std::invalid_argument("missing value for " + std::string(option));

        const std::string_view value = argv[i];
        if (option == "--samples") {
            const std::size_t parsed_value = parse_unsigned_integer<std::size_t>(option, value, false);
            if (parsed_value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("--samples is too large: '" + std::string(value) + "'");
            }
            options.sample_count = static_cast<int>(parsed_value);
        } else if (option == "--degree") {
            options.degree = parse_unsigned_integer<std::size_t>(option, value, true);
            if (options.degree == std::numeric_limits<std::size_t>::max()) {
                throw std::invalid_argument("--degree is too large: '" + std::string(value) + "'");
            }
        } else if (option == "--noise") {
            options.noise_stddev = parse_finite_number(option, value, true);
        } else if (option == "--epochs") {
            options.epochs = parse_unsigned_integer<std::size_t>(option, value, false);
        } else if (option == "--learning-rate") {
            options.learning_rate = parse_finite_number(option, value, false);
        } else if (option == "--weight-decay") {
            options.weight_decay = parse_finite_number(option, value, true);
        } else if (option == "--seed") {
            options.seed = parse_unsigned_integer<std::uint64_t>(option, value, true);
        } else {
            options.checkpoint_interval = parse_unsigned_integer<std::size_t>(option, value, false);
        }
    }
    return options;
}

void print_usage(std::ostream& output, std::string_view program) {
    output << "Usage: " << program << " [options]\n"
           << "  --samples N             Number of noisy samples (default: " << default_sample_count << ")\n"
           << "  --degree N              Polynomial degree, including zero (default: " << default_degree << ")\n"
           << "  --noise X               Noise standard deviation, >= 0 (default: " << default_noise_stddev << ")\n"
           << "  --epochs N              Number of training epochs (default: " << default_epochs << ")\n"
           << "  --learning-rate X       Positive CG learning rate (default: " << default_learning_rate << ")\n"
           << "  --weight-decay X        L2 weight decay, >= 0 (default: " << default_weight_decay << ")\n"
           << "  --seed N                Random seed, >= 0 (default: " << default_seed << ")\n"
           << "  --save-every N          Save a checkpoint every N epochs (default: "
           << default_checkpoint_interval << ")\n"
           << "  --checkpoint-interval N Alias for --save-every\n"
           << "  -h, --help              Show this help\n";
}

double true_curve(double x) {
    return std::sin(std::numbers::pi_v<double> * x);
}

double l2_penalty(const ml::PolynomialRegression& model, double weight_decay) {
    double squared_norm = 0.0;
    for (const double weight : model.weights()) squared_norm += weight * weight;
    return 0.5 * weight_decay * squared_norm;
}

std::ofstream open_csv(const std::filesystem::path& path) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("failed to open " + path.string());
    file << std::setprecision(17);
    return file;
}

std::string format_start_time(const std::chrono::system_clock::time_point& start_time) {
    const auto start_time_seconds = std::chrono::floor<std::chrono::seconds>(start_time);
    const std::time_t start_time_value = std::chrono::system_clock::to_time_t(start_time_seconds);
    const std::tm* local_time_pointer = std::localtime(&start_time_value);
    if (local_time_pointer == nullptr) throw std::runtime_error("failed to convert start time to local time");

    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(start_time - start_time_seconds).count();

    std::ostringstream formatted_time;
    formatted_time << std::put_time(local_time_pointer, "%Y%m%d_%H%M%S_")
                   << std::setfill('0') << std::setw(3) << milliseconds;
    return formatted_time.str();
}

}

int main(int argc, char* argv[]) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.show_help) {
            print_usage(std::cout, argv[0]);
            return 0;
        }

        const auto start_time = std::chrono::system_clock::now();
        const std::filesystem::path output_directory =
            std::filesystem::path("lab/lab1/output") / format_start_time(start_time);
        if (!std::filesystem::create_directories(output_directory)) {
            throw std::runtime_error("output directory already exists: " + output_directory.string());
        }

        ml::RandomGenerator rng(options.seed);
        ml::Dataset1D data = ml::sample_curve_noisy(
            true_curve, -1.0, 1.0, options.sample_count, options.noise_stddev, true, rng);

        {
            auto samples_file = open_csv(output_directory / "samples.csv");
            samples_file << "x,y\n";
            for (const auto& sample : data.samples) samples_file << sample.x << ',' << sample.y << '\n';
        }

        ml::PolynomialRegression model(options.degree);
        ml::MSELoss loss;
        ml::SerialBackend backend;
        ml::ConjugateGradient optimizer(options.learning_rate, options.weight_decay);
        auto result = backend.compute_loss_and_gradient(model, data, loss);

        {
            auto config_file = open_csv(output_directory / "config.csv");
            config_file << "seed,sample_count,degree,noise_stddev,epochs,learning_rate,weight_decay,"
                           "checkpoint_interval,curve_point_count\n";
            config_file << options.seed << ',' << options.sample_count << ',' << options.degree << ','
                        << options.noise_stddev << ',' << options.epochs << ',' << options.learning_rate << ','
                        << options.weight_decay << ','
                        << options.checkpoint_interval << ',' << curve_point_count << '\n';
        }

        {
            auto loss_file = open_csv(output_directory / "loss.csv");
            auto checkpoints_file = open_csv(output_directory / "checkpoints.csv");
            loss_file << "epoch,loss,l2_penalty,objective_loss\n";

            checkpoints_file << "epoch,loss,l2_penalty,objective_loss";
            for (std::size_t i = 0; i < model.weights().size(); ++i) checkpoints_file << ",w" << i;
            for (std::size_t i = 0; i < model.weights().size(); ++i) checkpoints_file << ",gradient" << i;
            checkpoints_file << '\n';

            for (std::size_t epoch = 1;; ++epoch) {
                optimizer.step(model.weights(), result.gradient);
                result = backend.compute_loss_and_gradient(model, data, loss);
                const double penalty = l2_penalty(model, options.weight_decay);
                const double objective_loss = result.loss + penalty;
                // const double objective_loss = result.loss;
                loss_file << epoch << ',' << result.loss << ',' << penalty << ',' << objective_loss << '\n';

                if (epoch % options.checkpoint_interval == 0 || epoch == options.epochs) {
                    checkpoints_file << epoch << ',' << result.loss << ',' << penalty << ',' << objective_loss;
                    for (const double weight : model.weights()) checkpoints_file << ',' << weight;
                    for (const double gradient : result.gradient) checkpoints_file << ',' << gradient;
                    checkpoints_file << '\n';
                }

                if (epoch == options.epochs) break;
            }
        }

        const double final_loss = result.loss;
        const double final_l2_penalty = l2_penalty(model, options.weight_decay);
        const double final_objective_loss = final_loss + final_l2_penalty;

        {
            auto curve_file = open_csv(output_directory / "curve.csv");
            curve_file << "x,y_true,y_pred\n";
            for (int i = 0; i < curve_point_count; ++i) {
                double x = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(curve_point_count - 1);
                curve_file << x << ',' << true_curve(x) << ',' << model.predict(x) << '\n';
            }
        }

        std::cout << std::setprecision(10)
                  << "final loss: " << final_loss << '\n'
                  << "final L2 penalty: " << final_l2_penalty << '\n'
                  << "final objective loss: " << final_objective_loss << '\n'
                  << "seed: " << options.seed << '\n'
                  << "degree: " << options.degree << '\n'
                  << "sample count: " << options.sample_count << '\n'
                  << "noise stddev: " << options.noise_stddev << '\n'
                  << "epochs: " << options.epochs << '\n'
                  << "checkpoint interval: " << options.checkpoint_interval << '\n'
                  << "learning rate: " << options.learning_rate << '\n'
                  << "weight decay: " << options.weight_decay << '\n'
                  << "CSV output: " << std::filesystem::absolute(output_directory).string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "lab1-cg failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
