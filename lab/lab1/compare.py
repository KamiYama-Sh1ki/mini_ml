import argparse
import csv
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_DIRECTORY = SCRIPT_DIRECTORY.parents[1]
OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "output"

# One-factor-at-a-time experiments.  Each group deliberately has its own
# baseline and epoch count so that changing one group does not accidentally
# change the meaning of another one.
EXPERIMENT_MATRIX = {
    "degree": {
        "title": "Polynomial degree",
        "variable": "degree",
        "symbol": "degree",
        "values": (1, 3, 5, 7, 9, 12, 15, 20),
        "fixed": {
            "sample_count": 50,
            "noise_stddev": 0.2,
            "learning_rate": 0.1,
            "weight_decay": 0.0,
        },
        "epochs": 50_000,
    },
    "samples": {
        "title": "Sample count",
        "variable": "sample_count",
        "symbol": "N",
        "values": (20, 50, 100, 300),
        "fixed": {
            "degree": 15,
            "noise_stddev": 0.2,
            "learning_rate": 0.1,
            "weight_decay": 0.0,
        },
        "epochs": 50_000,
    },
    "learning_rate": {
        "title": "Learning rate",
        "variable": "learning_rate",
        "symbol": "lr",
        "values": (0.005, 0.01, 0.05, 0.1, 0.3, 0.5),
        "fixed": {
            "sample_count": 100,
            "degree": 9,
            "noise_stddev": 0.1,
            "weight_decay": 1e-3,
        },
        "epochs": 1_000,
    },
    "l2": {
        "title": "L2 regularization",
        "variable": "weight_decay",
        "symbol": "lambda",
        "values": (0.0, 1e-4, 1e-3, 1e-2, 0.1),
        "fixed": {
            "sample_count": 30,
            "degree": 15,
            "noise_stddev": 0.2,
            "learning_rate": 0.1,
        },
        "epochs": 100_000,
    },
}

PARAMETER_OPTIONS = {
    "sample_count": "--samples",
    "degree": "--degree",
    "noise_stddev": "--noise",
    "learning_rate": "--learning-rate",
    "weight_decay": "--weight-decay",
    "seed": "--seed",
    "epochs": "--epochs",
    "save_every": "--save-every",
}

RUN_FIELDS = (
    "run_id",
    "experiment",
    "case",
    "parameter",
    "parameter_value",
    "status",
    "return_code",
    "elapsed_seconds",
    "run_dir",
    "command",
    "error",
)

COMPARISON_FIELDS = (
    "run_id",
    "experiment",
    "case",
    "parameter",
    "parameter_value",
    "seed",
    "sample_count",
    "degree",
    "noise_stddev",
    "epochs",
    "learning_rate",
    "weight_decay",
    "save_every",
    "final_epoch",
    "final_training_mse",
    "dense_r2",
    "dense_rmse",
    "dense_mae",
    "weight_l2_norm",
    "l2_penalty",
    "regularized_objective",
    "run_dir",
    "status",
    "error",
)


@dataclass(frozen=True)
class ExperimentCase:
    run_id: str
    experiment: str
    title: str
    parameter: str
    symbol: str
    parameter_value: object
    parameters: dict

    @property
    def label(self):
        return f"{self.symbol}={format_number(self.parameter_value)}"


def positive_integer(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_integer(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_arguments():
    default_build = Path(
        os.environ.get("MINI_ML_BUILD_DIR", str(Path.home() / "build" / "mini_ml"))
    )
    parser = argparse.ArgumentParser(
        description=(
            "Build mini_ml once, run the C++ polynomial trainer for an OFAT "
            "parameter matrix, and summarize its CSV outputs."
        )
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=tuple(EXPERIMENT_MATRIX),
        default=tuple(EXPERIMENT_MATRIX),
        help="experiment groups to run (default: all four groups)",
    )
    parser.add_argument(
        "--epochs",
        type=positive_integer,
        help="override the independently configured epoch count for every group",
    )
    parser.add_argument(
        "--save-every",
        type=positive_integer,
        help="checkpoint interval (default: only the final checkpoint of each case)",
    )
    parser.add_argument(
        "--seed",
        type=non_negative_integer,
        default=42,
        help="random seed shared by all cases (default: 42)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=default_build,
        help=f"CMake build directory (default: {default_build})",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="use an existing lab1_polynomial executable without invoking CMake",
    )
    parser.add_argument(
        "--timeout",
        type=positive_integer,
        default=300,
        help="maximum seconds allowed for each C++ case (default: 300)",
    )
    parser.add_argument(
        "--max-cases",
        type=positive_integer,
        help="run only the first N selected cases (useful for a quick smoke test)",
    )
    return parser.parse_args()


def format_number(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.12g}"


def create_comparison_directory():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    candidate = OUTPUT_DIRECTORY / f"comparison_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = OUTPUT_DIRECTORY / f"comparison_{timestamp}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def build_cases(arguments):
    cases = []
    for group_name in dict.fromkeys(arguments.groups):
        specification = EXPERIMENT_MATRIX[group_name]
        for index, value in enumerate(specification["values"], start=1):
            parameters = dict(specification["fixed"])
            parameters[specification["variable"]] = value
            parameters["seed"] = arguments.seed
            parameters["epochs"] = arguments.epochs or specification["epochs"]
            parameters["save_every"] = arguments.save_every or parameters["epochs"]
            cases.append(
                ExperimentCase(
                    run_id=f"{group_name}_{index:02d}",
                    experiment=group_name,
                    title=specification["title"],
                    parameter=specification["variable"],
                    symbol=specification["symbol"],
                    parameter_value=value,
                    parameters=parameters,
                )
            )
    if arguments.max_cases is not None:
        cases = cases[: arguments.max_cases]
    return cases


def configure_and_build(build_directory):
    configure = subprocess.run(
        ["cmake", "-S", str(PROJECT_DIRECTORY), "-B", str(build_directory)],
        cwd=PROJECT_DIRECTORY,
        text=True,
        capture_output=True,
        check=False,
    )
    if configure.returncode != 0:
        detail = configure.stderr.strip() or configure.stdout.strip()
        raise RuntimeError(f"CMake configure failed:\n{detail}")

    build = subprocess.run(
        ["cmake", "--build", str(build_directory), "--target", "lab1_polynomial"],
        cwd=PROJECT_DIRECTORY,
        text=True,
        capture_output=True,
        check=False,
    )
    if build.returncode != 0:
        detail = build.stderr.strip() or build.stdout.strip()
        raise RuntimeError(f"CMake build failed:\n{detail}")


def cpp_command(executable, parameters):
    command = [str(executable)]
    for name in (
        "sample_count",
        "degree",
        "noise_stddev",
        "learning_rate",
        "weight_decay",
        "seed",
        "epochs",
        "save_every",
    ):
        command.extend((PARAMETER_OPTIONS[name], format_number(parameters[name])))
    return command


def output_path_from_stdout(stdout):
    prefix = "CSV output: "
    matches = [line[len(prefix) :] for line in stdout.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("C++ output did not contain exactly one 'CSV output:' path")
    return Path(matches[0]).resolve()


def read_csv_rows(path, required_columns):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        missing = [column for column in required_columns if column not in fields]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return fields, rows


def finite_float(value, description):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {description}: {value!r}")
    return parsed


def integer_value(value, description):
    try:
        return int(value, 10)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid integer {description}: {value!r}") from error


def read_configuration(run_directory):
    path = run_directory / "config.csv"
    columns = (
        "seed",
        "sample_count",
        "degree",
        "noise_stddev",
        "epochs",
        "learning_rate",
        "weight_decay",
        "checkpoint_interval",
    )
    _, rows = read_csv_rows(path, columns)
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one row")
    row = rows[0]
    return {
        "seed": integer_value(row["seed"], "seed in config.csv"),
        "sample_count": integer_value(row["sample_count"], "sample_count in config.csv"),
        "degree": integer_value(row["degree"], "degree in config.csv"),
        "noise_stddev": finite_float(row["noise_stddev"], "noise_stddev in config.csv"),
        "epochs": integer_value(row["epochs"], "epochs in config.csv"),
        "learning_rate": finite_float(row["learning_rate"], "learning_rate in config.csv"),
        "weight_decay": finite_float(row["weight_decay"], "weight_decay in config.csv"),
        "save_every": integer_value(
            row["checkpoint_interval"], "checkpoint_interval in config.csv"
        ),
    }


def verify_configuration(actual, expected):
    integer_names = ("seed", "sample_count", "degree", "epochs", "save_every")
    for name in integer_names:
        if actual[name] != expected[name]:
            raise ValueError(
                f"config mismatch for {name}: expected {expected[name]}, got {actual[name]}"
            )
    for name in ("noise_stddev", "learning_rate", "weight_decay"):
        if not math.isclose(actual[name], expected[name], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"config mismatch for {name}: expected {expected[name]}, got {actual[name]}"
            )


def read_final_training_loss(run_directory, expected_epoch):
    path = run_directory / "loss.csv"
    _, rows = read_csv_rows(path, ("epoch", "loss"))
    row = rows[-1]
    epoch = integer_value(row["epoch"], "final epoch in loss.csv")
    if epoch != expected_epoch:
        raise ValueError(
            f"loss.csv ends at epoch {epoch}, expected epoch {expected_epoch}"
        )
    return epoch, finite_float(row["loss"], "final loss in loss.csv")


def read_final_weights(run_directory, expected_epoch):
    path = run_directory / "checkpoints.csv"
    fields, rows = read_csv_rows(path, ("epoch", "loss"))
    weight_columns = sorted(
        (name for name in fields if name.startswith("w") and name[1:].isdigit()),
        key=lambda name: int(name[1:]),
    )
    if weight_columns != [f"w{index}" for index in range(len(weight_columns))]:
        raise ValueError(f"{path} has non-contiguous weight columns")
    if not weight_columns:
        raise ValueError(f"{path} contains no weights")
    final_rows = [
        row
        for row in rows
        if integer_value(row["epoch"], "epoch in checkpoints.csv") == expected_epoch
    ]
    if len(final_rows) != 1:
        raise ValueError(
            f"checkpoints.csv must contain exactly one row for final epoch {expected_epoch}"
        )
    return [
        finite_float(final_rows[0][column], f"{column} in checkpoints.csv")
        for column in weight_columns
    ]


def read_dense_metrics(run_directory):
    path = run_directory / "curve.csv"
    _, rows = read_csv_rows(path, ("y_true", "y_pred"))
    true_values = [finite_float(row["y_true"], "y_true in curve.csv") for row in rows]
    predictions = [finite_float(row["y_pred"], "y_pred in curve.csv") for row in rows]
    errors = [prediction - truth for truth, prediction in zip(true_values, predictions)]
    mse = sum(error * error for error in errors) / len(errors)
    mae = sum(abs(error) for error in errors) / len(errors)
    mean = sum(true_values) / len(true_values)
    total = sum((truth - mean) ** 2 for truth in true_values)
    residual = sum(error * error for error in errors)
    r_squared = 1.0 - residual / total if total else float("nan")
    metrics = (r_squared, math.sqrt(mse), mae)
    if not all(math.isfinite(metric) for metric in metrics):
        raise ValueError("non-finite dense-curve regression metric")
    return metrics


def read_curve(run_directory):
    path = run_directory / "curve.csv"
    _, rows = read_csv_rows(path, ("x", "y_true", "y_pred"))
    x_values = [finite_float(row["x"], "x in curve.csv") for row in rows]
    true_values = [finite_float(row["y_true"], "y_true in curve.csv") for row in rows]
    predictions = [finite_float(row["y_pred"], "y_pred in curve.csv") for row in rows]
    return x_values, true_values, predictions


def collect_metrics(case, run_directory):
    configuration = read_configuration(run_directory)
    verify_configuration(configuration, case.parameters)
    final_epoch, training_mse = read_final_training_loss(
        run_directory, configuration["epochs"]
    )
    weights = read_final_weights(run_directory, configuration["epochs"])
    dense_r2, dense_rmse, dense_mae = read_dense_metrics(run_directory)
    squared_norm = sum(weight * weight for weight in weights)
    weight_norm = math.sqrt(squared_norm)
    # SGD adds lambda*w to the MSE gradient, which is the derivative of
    # (lambda/2)*||w||^2.
    l2_penalty = 0.5 * configuration["weight_decay"] * squared_norm
    objective = training_mse + l2_penalty
    if not all(
        math.isfinite(value)
        for value in (weight_norm, l2_penalty, objective)
    ):
        raise ValueError("non-finite weight norm or regularized objective")
    return {
        "seed": configuration["seed"],
        "sample_count": configuration["sample_count"],
        "degree": configuration["degree"],
        "noise_stddev": configuration["noise_stddev"],
        "epochs": configuration["epochs"],
        "learning_rate": configuration["learning_rate"],
        "weight_decay": configuration["weight_decay"],
        "save_every": configuration["save_every"],
        "final_epoch": final_epoch,
        "final_training_mse": training_mse,
        "dense_r2": dense_r2,
        "dense_rmse": dense_rmse,
        "dense_mae": dense_mae,
        "weight_l2_norm": weight_norm,
        "l2_penalty": l2_penalty,
        "regularized_objective": objective,
    }


def concise_error(text, limit=800):
    compact = " | ".join(line.strip() for line in text.splitlines() if line.strip())
    return compact[-limit:] if compact else "unknown error"


def run_case(case, executable, timeout):
    command = cpp_command(executable, case.parameters)
    started = time.monotonic()
    completed = None
    run_directory = ""
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_DIRECTORY,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            detail = completed.stderr or completed.stdout
            raise RuntimeError(
                f"C++ exited with code {completed.returncode}: {concise_error(detail)}"
            )
        run_directory = output_path_from_stdout(completed.stdout)
        metrics = collect_metrics(case, run_directory)
        return {
            "status": "ok",
            "return_code": completed.returncode,
            "elapsed_seconds": elapsed,
            "run_dir": str(run_directory),
            "command": " ".join(command),
            "error": "",
            **metrics,
        }
    except subprocess.TimeoutExpired as error:
        elapsed = time.monotonic() - started
        return {
            "status": "failed",
            "return_code": "timeout",
            "elapsed_seconds": elapsed,
            "run_dir": "",
            "command": " ".join(command),
            "error": f"timed out after {timeout} seconds",
        }
    except Exception as error:
        elapsed = time.monotonic() - started
        return {
            "status": "failed",
            "return_code": getattr(locals().get("completed", None), "returncode", ""),
            "elapsed_seconds": elapsed,
            "run_dir": str(run_directory),
            "command": " ".join(command),
            "error": concise_error(str(error)),
        }


def common_result_fields(case, result):
    return {
        "run_id": case.run_id,
        "experiment": case.experiment,
        "case": case.label,
        "parameter": case.parameter,
        "parameter_value": format_number(case.parameter_value),
        **result,
    }


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_text(value):
    return f"{value:.6g}" if isinstance(value, (int, float)) else "-"


def plot_comparison(path, cases, comparison_rows):
    successful = {row["run_id"]: row for row in comparison_rows if row["status"] == "ok"}
    selected_groups = list(dict.fromkeys(case.experiment for case in cases))
    if not selected_groups or not successful:
        figure, axes = plt.subplots(figsize=(9, 4))
        axes.axis("off")
        axes.text(0.5, 0.5, "No successful experiment cases", ha="center", va="center")
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return

    figure, axes = plt.subplots(
        len(selected_groups), 4, figsize=(18, 4.1 * len(selected_groups)), squeeze=False
    )
    metric_columns = (
        (("final_training_mse", "regularized_objective"), "Training objective"),
        (("dense_rmse", "dense_mae"), "Dense-curve error"),
        (("dense_r2",), "Dense-curve R-squared"),
        (("weight_l2_norm",), "Weight L2 norm"),
    )
    display_names = {
        "final_training_mse": "training MSE",
        "regularized_objective": "MSE + L2 penalty",
        "dense_rmse": "dense RMSE",
        "dense_mae": "dense MAE",
        "dense_r2": "dense R²",
        "weight_l2_norm": "||w||₂",
    }

    for row_index, group in enumerate(selected_groups):
        group_cases = [case for case in cases if case.experiment == group]
        labels = [case.label for case in group_cases]
        x_positions = list(range(len(group_cases)))
        for column_index, (metric_names, title) in enumerate(metric_columns):
            axis = axes[row_index][column_index]
            for metric_name in metric_names:
                values = [
                    successful[case.run_id][metric_name]
                    if case.run_id in successful
                    else float("nan")
                    for case in group_cases
                ]
                axis.plot(
                    x_positions,
                    values,
                    marker="o",
                    linewidth=2,
                    label=display_names[metric_name],
                )
            axis.set_xticks(x_positions, labels, rotation=20, ha="right")
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
            if len(metric_names) > 1:
                axis.legend(fontsize=8)
            if column_index == 0:
                axis.set_ylabel(EXPERIMENT_MATRIX[group]["title"])

    figure.suptitle("mini_ml Lab 1: one-factor-at-a-time comparison", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_fitted_curves(path, cases, comparison_rows):
    successful = {
        row["run_id"]: row for row in comparison_rows if row["status"] == "ok"
    }
    selected_groups = list(dict.fromkeys(case.experiment for case in cases))
    column_count = 2
    row_count = max(1, math.ceil(len(selected_groups) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(14, 5 * row_count),
        squeeze=False,
    )

    for axis, group in zip(axes.flat, selected_groups):
        group_cases = [case for case in cases if case.experiment == group]
        plotted_truth = False
        for case in group_cases:
            result = successful.get(case.run_id)
            if result is None:
                continue
            x_values, true_values, predictions = read_curve(
                Path(result["run_dir"])
            )
            if not plotted_truth:
                axis.plot(
                    x_values,
                    true_values,
                    color="black",
                    linestyle="--",
                    linewidth=2.4,
                    label="true sin(pi*x)",
                )
                plotted_truth = True
            axis.plot(x_values, predictions, linewidth=1.6, label=case.label)
        axis.set_title(EXPERIMENT_MATRIX[group]["title"])
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.grid(True, alpha=0.3)
        if plotted_truth:
            axis.legend(fontsize=8, ncols=2)
        else:
            axis.text(
                0.5,
                0.5,
                "No successful cases",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    for axis in list(axes.flat)[len(selected_groups) :]:
        axis.axis("off")

    figure.suptitle("mini_ml Lab 1: fitted-curve comparison", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def markdown_table(rows):
    lines = [
        "| 参数 | 训练 MSE | 稠密曲线 R² | 稠密曲线 RMSE | MAE | ||w||₂ | L2 惩罚 | 正则化目标 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {mse} | {r2} | {rmse} | {mae} | {norm} | {penalty} | {objective} |".format(
                case=row["case"],
                mse=metric_text(row["final_training_mse"]),
                r2=metric_text(row["dense_r2"]),
                rmse=metric_text(row["dense_rmse"]),
                mae=metric_text(row["dense_mae"]),
                norm=metric_text(row["weight_l2_norm"]),
                penalty=metric_text(row["l2_penalty"]),
                objective=metric_text(row["regularized_objective"]),
            )
        )
    return lines


def write_report(path, cases, comparison_rows, build_error=""):
    successful = [row for row in comparison_rows if row["status"] == "ok"]
    failed = [row for row in comparison_rows if row["status"] != "ok"]
    lines = [
        "# mini_ml 实验一参数比较",
        "",
        "本报告由 `compare.py` 批量调用 C++ `lab1_polynomial` 生成。Python 只负责实验编排、读取 CSV、统计和绘图，没有参与模型训练。",
        "",
        "采用单因素控制变量法（OFAT）：在每一组内只改变表中所示参数，其他参数保持不变，并统一使用随机种子 42（或命令行指定的种子）。",
        "",
        "## 指标说明",
        "",
        "- 训练 MSE：最后一轮在含噪训练样本上的均方误差。",
        "- 稠密曲线指标：在 `curve.csv` 的无噪声正弦函数网格上计算，用于衡量泛化和过拟合。",
        "- L2 惩罚：`lambda / 2 * ||w||²`；正则化目标为训练 MSE 与该惩罚之和。",
        "- 同组中训练 MSE 更低但稠密 RMSE 更高，是过拟合的直接迹象之一。",
        "",
    ]
    if build_error:
        lines.extend(("## 构建失败", "", f"```text\n{build_error}\n```", ""))

    for group in dict.fromkeys(case.experiment for case in cases):
        group_rows = [row for row in successful if row["experiment"] == group]
        lines.extend((f"## {EXPERIMENT_MATRIX[group]['title']}", ""))
        if not group_rows:
            lines.extend(("本组没有成功完成的实验。", ""))
            continue
        example = group_rows[0]
        configuration_names = {
            "sample_count": "N",
            "degree": "degree",
            "noise_stddev": "noise",
            "learning_rate": "lr",
            "weight_decay": "lambda",
            "epochs": "epochs",
        }
        fixed_configuration = ", ".join(
            f"{label}={format_number(example[name])}"
            for name, label in configuration_names.items()
            if name != EXPERIMENT_MATRIX[group]["variable"]
        )
        lines.extend((f"固定配置：`{fixed_configuration}`。", ""))
        lines.extend(markdown_table(group_rows))
        best_dense = min(group_rows, key=lambda row: row["dense_rmse"])
        best_train = min(group_rows, key=lambda row: row["final_training_mse"])
        lines.extend(
            (
                "",
                f"本组稠密曲线 RMSE 最低的是 **{best_dense['case']}**（{best_dense['dense_rmse']:.6g}）；"
                f"训练 MSE 最低的是 **{best_train['case']}**（{best_train['final_training_mse']:.6g}）。",
            )
        )
        if best_dense["run_id"] != best_train["run_id"]:
            lines.append("二者并非同一组参数，说明只比较训练误差会掩盖泛化差异，应结合稠密曲线指标判断过拟合。")
        lines.append("")

    lines.extend(("## 运行概况", "", f"成功：{len(successful)}；失败：{len(failed)}。", ""))
    if failed:
        lines.extend(("失败项：", ""))
        for row in failed:
            lines.append(f"- `{row['run_id']}`：{row['error']}")
        lines.append("")
    lines.extend(
        (
            "原始实验目录与完整参数见 `runs.csv`，逐项指标见 `comparison.csv`，汇总图见 `comparison.png`。",
            "各组最终拟合曲线见 `fits.png`。",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def initial_failed_row(case, error):
    return {
        "run_id": case.run_id,
        "experiment": case.experiment,
        "case": case.label,
        "parameter": case.parameter,
        "parameter_value": format_number(case.parameter_value),
        **case.parameters,
        "status": "failed",
        "run_dir": "",
        "error": error,
    }


def persist_results(summary_directory, cases, run_rows, comparison_rows, build_error=""):
    write_csv(summary_directory / "runs.csv", RUN_FIELDS, run_rows)
    write_csv(
        summary_directory / "comparison.csv", COMPARISON_FIELDS, comparison_rows
    )
    plot_comparison(summary_directory / "comparison.png", cases, comparison_rows)
    plot_fitted_curves(summary_directory / "fits.png", cases, comparison_rows)
    write_report(
        summary_directory / "report.md", cases, comparison_rows, build_error=build_error
    )


def main():
    arguments = parse_arguments()
    cases = build_cases(arguments)
    summary_directory = create_comparison_directory()
    build_directory = arguments.build_dir.expanduser().resolve()
    executable = build_directory / "lab1_polynomial"
    run_rows = []
    comparison_rows = []

    print(f"comparison output: {summary_directory}")
    try:
        if not arguments.skip_build:
            print("configuring and building lab1_polynomial ...")
            configure_and_build(build_directory)
        if not executable.is_file():
            raise RuntimeError(f"C++ executable does not exist: {executable}")
    except Exception as error:
        message = concise_error(str(error), limit=4000)
        for case in cases:
            failed = initial_failed_row(case, message)
            comparison_rows.append(failed)
            run_rows.append(failed)
        persist_results(
            summary_directory, cases, run_rows, comparison_rows, build_error=message
        )
        print(f"build failed: {message}", file=sys.stderr)
        return 1

    total = len(cases)
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{total}] {case.experiment}: {case.label}")
        result = run_case(case, executable, arguments.timeout)
        row = common_result_fields(case, result)
        run_rows.append(row)
        comparison_rows.append({**case.parameters, **row})
        # Keep usable tabular results if a later case fails or is interrupted;
        # the more expensive plot and report are rendered once at the end.
        write_csv(summary_directory / "runs.csv", RUN_FIELDS, run_rows)
        write_csv(
            summary_directory / "comparison.csv", COMPARISON_FIELDS, comparison_rows
        )
        if result["status"] == "ok":
            print(
                f"  RMSE={result['dense_rmse']:.6g}, R²={result['dense_r2']:.6g}, "
                f"{result['elapsed_seconds']:.2f}s"
            )
        else:
            print(f"  failed: {result['error']}", file=sys.stderr)

    persist_results(summary_directory, cases, run_rows, comparison_rows)
    failed_count = sum(row["status"] != "ok" for row in comparison_rows)
    print(f"completed: {total - failed_count} succeeded, {failed_count} failed")
    print(f"report: {summary_directory / 'report.md'}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
