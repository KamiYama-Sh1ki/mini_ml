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
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <ml/core/random.hpp>
#include <ml/data/sampling.hpp>
#include <ml/model/regression.hpp>
#include <ml/ops/loss.hpp>
#include <ml/optimizer/sgd.hpp>
#include <ml/runtime/serial.hpp>

namespace {

constexpr std::uint64_t seed = 42;
constexpr int sample_count = 100;
constexpr std::size_t degree = 9;
constexpr double noise_stddev = 0.1;
constexpr std::size_t default_epochs = 1000;
constexpr double learning_rate = 0.1;
constexpr double weight_decay = 1e-3;
constexpr std::size_t default_checkpoint_interval = 200;
constexpr int curve_point_count = 400;

struct Options {
    std::size_t epochs = default_epochs;
    std::size_t checkpoint_interval = default_checkpoint_interval;
    bool show_help = false;
};

std::size_t parse_positive_integer(std::string_view option, std::string_view value) {
    std::size_t parsed_value = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed_value);
    if (error != std::errc{} || end != value.data() + value.size() || parsed_value == 0) {
        throw std::invalid_argument(std::string(option) + " requires a positive integer, got '" +
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

        if (option != "--epochs" && option != "--save-every" && option != "--checkpoint-interval") {
            throw std::invalid_argument("unknown option: " + std::string(option));
        }
        if (++i >= argc) throw std::invalid_argument("missing value for " + std::string(option));

        const std::size_t value = parse_positive_integer(option, argv[i]);
        if (option == "--epochs") {
            options.epochs = value;
        } else {
            options.checkpoint_interval = value;
        }
    }
    return options;
}

void print_usage(std::ostream& output, std::string_view program) {
    output << "Usage: " << program << " [--epochs N] [--save-every N]\n"
           << "  --epochs N              Number of training epochs (default: " << default_epochs << ")\n"
           << "  --save-every N          Save a checkpoint every N epochs (default: "
           << default_checkpoint_interval << ")\n"
           << "  --checkpoint-interval N Alias for --save-every\n"
           << "  -h, --help              Show this help\n";
}

double true_curve(double x) {
    return std::sin(std::numbers::pi_v<double> * x);
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

        ml::RandomGenerator rng(seed);
        ml::Dataset1D data = ml::sample_curve_noisy(true_curve, -1.0, 1.0, sample_count, noise_stddev, true, rng);

        {
            auto samples_file = open_csv(output_directory / "samples.csv");
            samples_file << "x,y\n";
            for (const auto& sample : data.samples) samples_file << sample.x << ',' << sample.y << '\n';
        }

        ml::PolynomialRegression model(degree);
        ml::MSELoss loss;
        ml::SerialBackend backend;
        ml::SGD optimizer(learning_rate, weight_decay);
        auto result = backend.compute_loss_and_gradient(model, data, loss);

        {
            auto config_file = open_csv(output_directory / "config.csv");
            config_file << "seed,sample_count,degree,noise_stddev,epochs,learning_rate,weight_decay,"
                           "checkpoint_interval,curve_point_count\n";
            config_file << seed << ',' << sample_count << ',' << degree << ',' << noise_stddev << ','
                        << options.epochs << ',' << learning_rate << ',' << weight_decay << ','
                        << options.checkpoint_interval << ',' << curve_point_count << '\n';
        }

        {
            auto loss_file = open_csv(output_directory / "loss.csv");
            auto checkpoints_file = open_csv(output_directory / "checkpoints.csv");
            loss_file << "epoch,loss\n";

            checkpoints_file << "epoch,loss";
            for (std::size_t i = 0; i < model.weights().size(); ++i) checkpoints_file << ",w" << i;
            for (std::size_t i = 0; i < model.weights().size(); ++i) checkpoints_file << ",gradient" << i;
            checkpoints_file << '\n';

            for (std::size_t epoch = 1;; ++epoch) {
                optimizer.step(model.weights(), result.gradient);
                result = backend.compute_loss_and_gradient(model, data, loss);
                loss_file << epoch << ',' << result.loss << '\n';

                if (epoch % options.checkpoint_interval == 0 || epoch == options.epochs) {
                    checkpoints_file << epoch << ',' << result.loss;
                    for (const double weight : model.weights()) checkpoints_file << ',' << weight;
                    for (const double gradient : result.gradient) checkpoints_file << ',' << gradient;
                    checkpoints_file << '\n';
                }

                if (epoch == options.epochs) break;
            }
        }

        const double final_loss = result.loss;

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
                  << "degree: " << degree << '\n'
                  << "sample count: " << sample_count << '\n'
                  << "epochs: " << options.epochs << '\n'
                  << "checkpoint interval: " << options.checkpoint_interval << '\n'
                  << "learning rate: " << learning_rate << '\n'
                  << "weight decay: " << weight_decay << '\n'
                  << "CSV output: " << std::filesystem::absolute(output_directory).string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "lab1 failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
