#pragma once

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace lab {

inline std::string timestamp(std::chrono::system_clock::time_point time) {
    const auto seconds = std::chrono::floor<std::chrono::seconds>(time);
    const std::time_t value = std::chrono::system_clock::to_time_t(seconds);
    const std::tm* local = std::localtime(&value);
    if (local == nullptr) throw std::runtime_error("failed to convert time to local time");
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(time - seconds).count();
    std::ostringstream stream;
    stream << std::put_time(local, "%Y%m%d_%H%M%S_") << std::setfill('0') << std::setw(3) << milliseconds;
    return stream.str();
}

class RunOutput {
public:
    explicit RunOutput(std::filesystem::path base) : directory_(base / timestamp(std::chrono::system_clock::now())) {
        if (!std::filesystem::create_directories(directory_)) {
            throw std::runtime_error("output directory already exists: " + directory_.string());
        }
    }

    const std::filesystem::path& directory() const noexcept { return directory_; }

    std::ofstream csv(std::string_view name) const {
        std::ofstream file(directory_ / std::string(name));
        if (!file) throw std::runtime_error("failed to open " + (directory_ / std::string(name)).string());
        file << std::setprecision(17);
        return file;
    }

    template <typename Samples>
    void write_samples(std::string_view name, const Samples& samples) const {
        auto file = csv(name);
        file << "x,y\n";
        for (const auto& sample : samples) file << sample.x << ',' << sample.y << '\n';
    }

    template <typename Truth, typename Predict>
    void write_curve(std::string_view name, Truth truth, Predict predict, int points, double left,
                     double right) const {
        auto file = csv(name);
        file << "x,y_true,y_pred\n";
        for (int i = 0; i < points; ++i) {
            const double x = left + (right - left) * static_cast<double>(i) / static_cast<double>(points - 1);
            file << x << ',' << truth(x) << ',' << predict(x) << '\n';
        }
    }

private:
    std::filesystem::path directory_;
};

}  // namespace lab
