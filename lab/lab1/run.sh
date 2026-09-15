#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd -- "$script_directory/../.." && pwd)"
build_directory="${MINI_ML_BUILD_DIR:-$HOME/build/mini_ml}"
python_executable="${MINI_ML_PYTHON:-$HOME/miniconda3/envs/mini_ml/bin/python}"

usage() {
    printf '%s\n' \
        "Usage: $0 [options] [checkpoint epochs...]" \
        "  --optimizer NAME      Optimizer binary to run: sgd or cg (default: sgd)" \
        "  --samples N           Number of noisy samples (default: 100)" \
        "  --degree N            Polynomial degree, including zero (default: 9)" \
        "  --noise X             Noise standard deviation, >= 0 (default: 0.1)" \
        "  --epochs N            Train for N epochs (default: 1000)" \
        "  --learning-rate X     Positive SGD learning rate (default: 0.1)" \
        "  --weight-decay X      L2 weight decay, >= 0 (default: 0.001)" \
        "  --seed N              Random seed, >= 0 (default: 42)" \
        "  --save-every N        Save a C++ checkpoint every N epochs (default: 200)" \
        "  --plot-every N        Plot every N epochs (N must be a multiple of --save-every)" \
        "  --plot-every default" \
        "                        Plot at the C++ checkpoint frequency (default)" \
        "  checkpoint epochs" \
        "                        Plot these exact saved epochs instead of using a frequency" \
        "  -h, --help            Show this help"
}

is_non_negative_number() {
    local value="$1"
    local number_pattern='^(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$'
    [[ "$value" =~ $number_pattern ]]
}

is_positive_number() {
    local value="$1"
    local mantissa="${value%%[eE]*}"
    is_non_negative_number "$value" && [[ "$mantissa" =~ [1-9] ]]
}

training_arguments=()
plot_epochs=()
plot_every="default"
plot_every_set=false
optimizer="sgd"
lab1_target=lab1_polynomial

while (( $# > 0 )); do
    case "$1" in
        --optimizer)
            if (( $# < 2 )); then
                printf 'missing value for %s\n' "$1" >&2
                usage >&2
                exit 2
            fi
            case "$2" in
                sgd) lab1_target=lab1_polynomial ;;
                cg) lab1_target=lab1_polynomial_cg ;;
                *)
                    printf '%s requires sgd or cg, got %q\n' "$1" "$2" >&2
                    usage >&2
                    exit 2
                    ;;
            esac
            optimizer="$2"
            shift 2
            ;;
        --samples|--epochs|--save-every|--checkpoint-interval)
            if (( $# < 2 )); then
                printf 'missing value for %s\n' "$1" >&2
                usage >&2
                exit 2
            fi
            if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                printf '%s requires a positive integer, got %q\n' "$1" "$2" >&2
                exit 2
            fi
            training_arguments+=("$1" "$2")
            shift 2
            ;;
        --degree|--seed)
            if (( $# < 2 )); then
                printf 'missing value for %s\n' "$1" >&2
                usage >&2
                exit 2
            fi
            if [[ ! "$2" =~ ^[0-9]+$ ]]; then
                printf '%s requires a non-negative integer, got %q\n' "$1" "$2" >&2
                exit 2
            fi
            training_arguments+=("$1" "$2")
            shift 2
            ;;
        --noise|--weight-decay)
            if (( $# < 2 )); then
                printf 'missing value for %s\n' "$1" >&2
                usage >&2
                exit 2
            fi
            if ! is_non_negative_number "$2"; then
                printf '%s requires a non-negative number, got %q\n' "$1" "$2" >&2
                exit 2
            fi
            training_arguments+=("$1" "$2")
            shift 2
            ;;
        --learning-rate)
            if (( $# < 2 )); then
                printf 'missing value for %s\n' "$1" >&2
                usage >&2
                exit 2
            fi
            if ! is_positive_number "$2"; then
                printf '%s requires a positive number, got %q\n' "$1" "$2" >&2
                exit 2
            fi
            training_arguments+=("$1" "$2")
            shift 2
            ;;
        --plot-every)
            if (( $# < 2 )); then
                printf 'missing value for %s\n' "$1" >&2
                usage >&2
                exit 2
            fi
            if [[ "$2" != "default" && ! "$2" =~ ^[1-9][0-9]*$ ]]; then
                printf '%s requires a positive integer or default, got %q\n' "$1" "$2" >&2
                exit 2
            fi
            plot_every="$2"
            plot_every_set=true
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            printf 'unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ ! "$1" =~ ^[1-9][0-9]*$ ]]; then
                printf 'checkpoint epoch must be a positive integer, got %q\n' "$1" >&2
                exit 2
            fi
            plot_epochs+=("$1")
            shift
            ;;
    esac
done

if [[ "$plot_every_set" == true ]] && (( ${#plot_epochs[@]} > 0 )); then
    printf '%s\n' 'checkpoint epochs and --plot-every cannot be used together' >&2
    exit 2
fi

cd "$project_directory"
cmake -S "$project_directory" -B "$build_directory"
cmake --build "$build_directory" --target "$lab1_target"
"$build_directory/$lab1_target" "${training_arguments[@]}"

if (( ${#plot_epochs[@]} > 0 )); then
    "$python_executable" "$script_directory/plot.py" "${plot_epochs[@]}"
else
    "$python_executable" "$script_directory/plot.py" --plot-every "$plot_every"
fi
