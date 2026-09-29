#pragma once

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lab {

struct ArgOptions {
    std::string_view config_key = {};
    bool in_summary = true;
};

struct ArgSpec {
    std::string_view names;
    std::string_view help;
    bool in_summary = true;
    std::string config_key;
    std::function<void(std::string_view)> set;
    std::function<std::string()> value;
};

namespace detail {

inline std::string format_number(double value) {
    std::ostringstream stream;
    stream.precision(17);
    stream << value;
    return stream.str();
}

inline bool matches(std::string_view names, std::string_view option) {
    std::size_t begin = 0;
    while (begin <= names.size()) {
        const std::size_t end = names.find('|', begin);
        const std::string_view name = names.substr(begin, end - begin);
        if (name == option) return true;
        if (end == std::string_view::npos) break;
        begin = end + 1;
    }
    return false;
}

inline std::string_view first_name(std::string_view names) {
    const std::size_t end = names.find('|');
    return names.substr(0, end);
}

inline std::string default_config_key(std::string_view names) {
    std::string_view name = first_name(names);
    if (name.substr(0, 2) == "--") name.remove_prefix(2);
    return std::string(name);
}

}  // namespace detail

inline ArgSpec arg(std::string_view names, std::string& target,
                   std::initializer_list<std::string_view> choices, std::string_view help,
                   ArgOptions options = {}) {
    return ArgSpec{
        .names = names,
        .help = help,
        .in_summary = options.in_summary,
        .config_key = options.config_key.empty() ? detail::default_config_key(names) : std::string(options.config_key),
        .set = [&target, choices, names](std::string_view value) {
            for (std::string_view choice : choices) {
                if (value == choice) {
                    target = std::string(value);
                    return;
                }
            }
            throw std::invalid_argument(std::string(names) + " requires one of its listed values, got '" +
                                        std::string(value) + "'");
        },
        .value = [&target] { return target; },
    };
}

template <typename Unsigned>
ArgSpec arg(std::string_view names, Unsigned& target, bool allow_zero, std::string_view help,
            ArgOptions options = {}) {
    return ArgSpec{
        .names = names,
        .help = help,
        .in_summary = options.in_summary,
        .config_key = options.config_key.empty() ? detail::default_config_key(names) : std::string(options.config_key),
        .set = [&target, names, allow_zero](std::string_view value) {
            Unsigned parsed = 0;
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
            const bool invalid = error != std::errc{} || end != value.data() + value.size() ||
                                 (!allow_zero && parsed == 0);
            if (invalid) {
                const char* requirement = allow_zero ? "a non-negative integer" : "a positive integer";
                throw std::invalid_argument(std::string(names) + " requires " + requirement + ", got '" +
                                            std::string(value) + "'");
            }
            target = parsed;
        },
        .value = [&target] { return std::to_string(target); },
    };
}

inline ArgSpec arg(std::string_view names, double& target, bool allow_zero, std::string_view help,
                   ArgOptions options = {}) {
    return ArgSpec{
        .names = names,
        .help = help,
        .in_summary = options.in_summary,
        .config_key = options.config_key.empty() ? detail::default_config_key(names) : std::string(options.config_key),
        .set = [&target, names, allow_zero](std::string_view value) {
            double parsed = 0.0;
            const auto [end, error] =
                std::from_chars(value.data(), value.data() + value.size(), parsed, std::chars_format::general);
            const bool invalid = error != std::errc{} || end != value.data() + value.size() ||
                                 !std::isfinite(parsed) || (allow_zero ? parsed < 0.0 : parsed <= 0.0);
            if (invalid) {
                const char* requirement = allow_zero ? "a finite non-negative number" : "a finite positive number";
                throw std::invalid_argument(std::string(names) + " requires " + requirement + ", got '" +
                                            std::string(value) + "'");
            }
            target = parsed;
        },
        .value = [&target] { return detail::format_number(target); },
    };
}

inline ArgSpec arg(std::string_view names, std::string& target, std::string_view help,
                   ArgOptions options = {}) {
    return ArgSpec{
        .names = names,
        .help = help,
        .in_summary = options.in_summary,
        .config_key = options.config_key.empty() ? detail::default_config_key(names) : std::string(options.config_key),
        .set = [&target](std::string_view value) { target = std::string(value); },
        .value = [&target] { return target; },
    };
}

inline ArgSpec arg(std::string_view names, double& target, std::string_view help, ArgOptions options = {}) {
    return ArgSpec{
        .names = names,
        .help = help,
        .in_summary = options.in_summary,
        .config_key = options.config_key.empty() ? detail::default_config_key(names) : std::string(options.config_key),
        .set = [&target, names](std::string_view value) {
            double parsed = 0.0;
            const auto [end, error] =
                std::from_chars(value.data(), value.data() + value.size(), parsed, std::chars_format::general);
            if (error != std::errc{} || end != value.data() + value.size() || !std::isfinite(parsed)) {
                throw std::invalid_argument(std::string(names) + " requires a finite number, got '" +
                                            std::string(value) + "'");
            }
            target = parsed;
        },
        .value = [&target] { return detail::format_number(target); },
    };
}

template <typename Value>
ArgSpec constant(std::string_view key, Value value) {
    return ArgSpec{
        .names = {},
        .help = {},
        .in_summary = false,
        .config_key = std::string(key),
        .set = nullptr,
        .value = [value] { return std::to_string(value); },
    };
}

class ArgParser {
public:
    void add(ArgSpec spec) { specs_.push_back(std::move(spec)); }

    bool parse(int argc, char* argv[]) const {
        for (int i = 1; i < argc; ++i) {
            const std::string_view option = argv[i];
            if (option == "-h" || option == "--help") return true;
            const ArgSpec* matched = nullptr;
            for (const ArgSpec& spec : specs_) {
                if (!spec.names.empty() && detail::matches(spec.names, option)) {
                    matched = &spec;
                    break;
                }
            }
            if (matched == nullptr) throw std::invalid_argument("unknown option: " + std::string(option));
            if (++i >= argc) throw std::invalid_argument("missing value for " + std::string(option));
            matched->set(argv[i]);
        }
        return false;
    }

    void print_usage(std::ostream& output, std::string_view program) const {
        output << "Usage: " << program << " [options]\n";
        for (const ArgSpec& spec : specs_) {
            if (spec.names.empty() || spec.help.empty()) continue;
            output << "  " << spec.names << "    " << spec.help << " (default: " << spec.value() << ")\n";
        }
        output << "  -h, --help              Show this help\n";
    }

    void print_values(std::ostream& output) const {
        for (const ArgSpec& spec : specs_) {
            if (spec.names.empty() || !spec.in_summary) continue;
            output << spec.config_key << ": " << spec.value() << '\n';
        }
    }

    void write_config(std::ostream& output) const {
        bool first = true;
        for (const ArgSpec& spec : specs_) {
            if (spec.config_key.empty()) continue;
            if (!first) output << ',';
            output << spec.config_key;
            first = false;
        }
        output << '\n';
        first = true;
        for (const ArgSpec& spec : specs_) {
            if (spec.config_key.empty()) continue;
            if (!first) output << ',';
            output << spec.value();
            first = false;
        }
        output << '\n';
    }

private:
    std::vector<ArgSpec> specs_;
};

}  // namespace lab
