#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_executable="${MINI_ML_PYTHON:-$HOME/miniconda3/envs/mini_ml/bin/python}"

exec "$python_executable" "$script_directory/compare.py" "$@"
