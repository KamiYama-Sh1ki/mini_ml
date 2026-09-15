#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_directory="$(dirname -- "$script_directory")"
build_directory="${MINI_ML_BUILD_DIR:-$HOME/build/mini_ml}"
python_executable="${MINI_ML_PYTHON:-$HOME/miniconda3/envs/mini_ml/bin/python}"

if (( $# == 0 )); then set -- 200 400 600; fi

cd "$project_directory"
cmake -S "$project_directory" -B "$build_directory"
cmake --build "$build_directory" --target lab1_polynomial
"$build_directory/lab1_polynomial"
"$python_executable" "$script_directory/plot.py" "$@"
