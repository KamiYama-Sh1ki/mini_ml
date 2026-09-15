#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(cd -- "$script_directory/../.." && pwd)"
build_directory="${MINI_ML_BUILD_DIR:-$HOME/build/mini_ml}"
python_executable="${MINI_ML_PYTHON:-$HOME/miniconda3/envs/mini_ml/bin/python}"

usage() {
    printf '%s\n' \
        "Usage: $0 [options] [checkpoint epochs...]" \
        "  --epochs N       Train for N epochs" \
        "  --save-every N   Save a C++ checkpoint every N epochs" \
        "  --plot-every N   Plot every N epochs (N must be a multiple of --save-every)" \
        "  --plot-every default" \
        "                   Plot at the C++ checkpoint frequency (default)" \
        "  checkpoint epochs" \
        "                   Plot these exact saved epochs instead of using a frequency" \
        "  -h, --help       Show this help"
}

training_arguments=()
plot_epochs=()
plot_every="default"
plot_every_set=false

while (( $# > 0 )); do
    case "$1" in
        --epochs|--save-every)
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
cmake --build "$build_directory" --target lab1_polynomial
"$build_directory/lab1_polynomial" "${training_arguments[@]}"

if (( ${#plot_epochs[@]} > 0 )); then
    "$python_executable" "$script_directory/plot.py" "${plot_epochs[@]}"
else
    "$python_executable" "$script_directory/plot.py" --plot-every "$plot_every"
fi
