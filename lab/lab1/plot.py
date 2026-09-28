import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt

WINDOWS_CJK_FONTS = (
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)


def load_cjk_font():
    for path in WINDOWS_CJK_FONTS:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=path).get_name()
    return None


REQUIRED_FILES = (
    "samples.csv",
    "curve.csv",
    "loss.csv",
    "checkpoints.csv",
    "config.csv",
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot saved checkpoint weights from a mini_ml lab1 run. Positional "
            "epochs select exact checkpoints; otherwise all checkpoints are plotted "
            "by default."
        )
    )
    parser.add_argument(
        "epochs",
        metavar="EPOCH",
        nargs="*",
        type=int,
        help="saved checkpoint epochs to plot, for example: 2000 4000 6000",
    )
    parser.add_argument(
        "--plot-every",
        metavar="N|default",
        help=(
            "plot every N epochs, or 'default' to use the C++ checkpoint interval "
            "(default: plot every saved checkpoint)"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="run directory containing the CSV files (default: latest run in lab/lab1/output)",
    )
    return parser, parser.parse_args()


def has_required_files(directory):
    return directory.is_dir() and all(
        (directory / filename).is_file() for filename in REQUIRED_FILES
    )


def select_run_directory(explicit_directory):
    if explicit_directory is not None:
        run_directory = explicit_directory.expanduser().resolve()
        missing = [
            filename
            for filename in REQUIRED_FILES
            if not (run_directory / filename).is_file()
        ]
        if missing:
            raise ValueError(
                f"run directory '{run_directory}' is missing: {', '.join(missing)}"
            )
        return run_directory

    output_directory = Path(__file__).resolve().parent / "output"
    if not output_directory.is_dir():
        raise ValueError(f"output directory does not exist: '{output_directory}'")

    run_directories = [
        path for path in output_directory.iterdir() if has_required_files(path)
    ]
    if not run_directories:
        required = ", ".join(REQUIRED_FILES)
        raise ValueError(
            f"no completed run directory found under '{output_directory}'; "
            f"a run must contain {required}"
        )

    # Run directories use a lexicographically sortable start-time name.
    return max(run_directories, key=lambda path: path.name)


def read_rows(path, required_columns):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        missing = [name for name in required_columns if name not in fieldnames]
        if missing:
            raise ValueError(f"'{path}' is missing columns: {', '.join(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"'{path}' contains no data rows")
    return fieldnames, rows


def parse_number(row, column, path, row_number):
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid {column!r} value in '{path}' at CSV row {row_number}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"non-finite {column!r} value in '{path}' at CSV row {row_number}"
        )
    return value


def read_columns(path, names):
    _, rows = read_rows(path, names)
    columns = [[] for _ in names]
    for row_number, row in enumerate(rows, start=2):
        for values, name in zip(columns, names):
            values.append(parse_number(row, name, path, row_number))
    return columns


def read_loss_history(path):
    fieldnames, rows = read_rows(path, ("epoch", "loss"))
    epochs = []
    losses = []
    objective_losses = [] if "objective_loss" in fieldnames else None
    for row_number, row in enumerate(rows, start=2):
        epochs.append(parse_number(row, "epoch", path, row_number))
        losses.append(parse_number(row, "loss", path, row_number))
        if objective_losses is not None:
            objective_losses.append(
                parse_number(row, "objective_loss", path, row_number)
            )
    return epochs, losses, objective_losses


def parse_epoch(row, path, row_number):
    value = parse_number(row, "epoch", path, row_number)
    if not value.is_integer():
        raise ValueError(
            f"non-integer 'epoch' value in '{path}' at CSV row {row_number}"
        )
    return int(value)


def parse_positive_integer(row, column, path, row_number):
    value = parse_number(row, column, path, row_number)
    if not value.is_integer() or value <= 0:
        raise ValueError(
            f"{column!r} in '{path}' must be a positive integer "
            f"(CSV row {row_number})"
        )
    return int(value)


def read_config(path):
    _, rows = read_rows(path, ("epochs", "checkpoint_interval"))
    if len(rows) != 1:
        raise ValueError(f"'{path}' must contain exactly one configuration row")

    row = rows[0]
    return {
        "epochs": parse_positive_integer(row, "epochs", path, 2),
        "checkpoint_interval": parse_positive_integer(
            row, "checkpoint_interval", path, 2
        ),
    }


def read_checkpoints(path):
    fieldnames, rows = read_rows(path, ("epoch", "loss"))
    weight_columns = sorted(
        (
            name
            for name in fieldnames
            if name.startswith("w") and name[1:].isdigit()
        ),
        key=lambda name: int(name[1:]),
    )
    if not weight_columns:
        raise ValueError(f"'{path}' contains no weight columns (expected w0, w1, ...)")

    expected_columns = [f"w{index}" for index in range(len(weight_columns))]
    if weight_columns != expected_columns:
        raise ValueError(
            f"'{path}' weight columns must be contiguous from w0; found: "
            f"{', '.join(weight_columns)}"
        )

    checkpoints = {}
    for row_number, row in enumerate(rows, start=2):
        epoch = parse_epoch(row, path, row_number)
        if epoch in checkpoints:
            raise ValueError(f"duplicate epoch {epoch} in '{path}'")
        checkpoints[epoch] = {
            "loss": parse_number(row, "loss", path, row_number),
            "weights": [
                parse_number(row, name, path, row_number) for name in weight_columns
            ],
        }
    return checkpoints


def parse_plot_every(value, checkpoint_interval):
    if value is None or value == "default":
        return None

    try:
        plot_interval = int(value)
    except ValueError as error:
        raise ValueError(
            "--plot-every must be a positive integer or 'default'"
        ) from error

    if plot_interval <= 0:
        raise ValueError("--plot-every must be a positive integer or 'default'")
    if plot_interval % checkpoint_interval != 0:
        raise ValueError(
            f"--plot-every ({plot_interval}) must be an integer multiple of the "
            f"C++ checkpoint interval ({checkpoint_interval})"
        )
    return plot_interval


def select_checkpoints_by_frequency(checkpoints, config, plot_every):
    configured_epochs = config["epochs"]
    checkpoint_interval = config["checkpoint_interval"]
    invalid_epochs = sorted(
        epoch for epoch in checkpoints if epoch <= 0 or epoch > configured_epochs
    )
    if invalid_epochs:
        raise ValueError(
            "checkpoint epoch(s) fall outside the configured training range: "
            + ", ".join(str(epoch) for epoch in invalid_epochs)
        )
    if configured_epochs not in checkpoints:
        raise ValueError(
            f"final epoch {configured_epochs} from config.csv is not saved in "
            "checkpoints.csv"
        )

    plot_interval = parse_plot_every(plot_every, checkpoint_interval)
    if plot_interval is None:
        selected_epochs = sorted(checkpoints)
    else:
        selected_epochs = list(
            range(plot_interval, configured_epochs + 1, plot_interval)
        )
        if configured_epochs not in selected_epochs:
            selected_epochs.append(configured_epochs)

        unavailable = [epoch for epoch in selected_epochs if epoch not in checkpoints]
        if unavailable:
            raise ValueError(
                "checkpoint(s) required by --plot-every are missing: "
                + ", ".join(str(epoch) for epoch in unavailable)
            )

    return [(epoch, checkpoints[epoch]) for epoch in selected_epochs]


def polynomial_predictions(x_values, weights):
    predictions = []
    for x_value in x_values:
        prediction = 0.0
        for weight in reversed(weights):
            prediction = prediction * x_value + weight
        predictions.append(prediction)
    return predictions


def regression_metrics(true_values, predicted_values):
    if len(true_values) != len(predicted_values) or not true_values:
        raise ValueError("regression metrics require equally sized, non-empty data")

    errors = [
        predicted - actual
        for actual, predicted in zip(true_values, predicted_values)
    ]
    mse = sum(error * error for error in errors) / len(errors)
    mae = sum(abs(error) for error in errors) / len(errors)
    true_mean = sum(true_values) / len(true_values)
    total_variation = sum((value - true_mean) ** 2 for value in true_values)
    residual_variation = sum(error * error for error in errors)
    r_squared = (
        1.0 - residual_variation / total_variation
        if total_variation != 0.0
        else (1.0 if residual_variation == 0.0 else float("nan"))
    )
    return r_squared, math.sqrt(mse), mae


def plot_fit(run_directory, sample_x, sample_y, curve_x, true_y, predictions):
    font_name = load_cjk_font()
    if font_name:
        plt.rcParams["font.family"] = font_name
        plt.rcParams["axes.unicode_minus"] = False
    figure, axes = plt.subplots(figsize=(9, 6))
    axes.scatter(sample_x, sample_y, s=22, alpha=0.65, label="含噪声的训练样本")
    axes.plot(curve_x, true_y, linewidth=2, label="真实曲线 sin(πx)")
    for label, predicted_y in predictions:
        r_squared, _, _ = regression_metrics(true_y, predicted_y)
        axes.plot(curve_x, predicted_y, linewidth=2, label=f"{label} (R²={r_squared:.4f})")
    # Truncate the y-axis around the data and the true curve so exploding
    # fits stay readable instead of stretching the scale to their extremes.
    data_min = min(min(sample_y), min(true_y))
    data_max = max(max(sample_y), max(true_y))
    data_range = data_max - data_min
    axes.set_ylim(data_min - 0.5 * data_range, data_max + 0.5 * data_range)
    axes.set_title("多项式拟合随训练轮数的演化")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(run_directory / "fit.png", dpi=150)
    plt.close(figure)


def plot_loss(
    run_directory,
    loss_epochs,
    losses,
    objective_losses,
    selected_checkpoints,
):
    figure, axes = plt.subplots(figsize=(9, 6))
    axes.plot(loss_epochs, losses, linewidth=1.5, label="Training MSE")
    if objective_losses is not None and objective_losses != losses:
        axes.plot(
            loss_epochs,
            objective_losses,
            linewidth=1.3,
            linestyle="--",
            label="Objective (MSE + L2 penalty)",
        )
    if selected_checkpoints:
        selected_epochs = [epoch for epoch, _ in selected_checkpoints]
        selected_losses = [checkpoint["loss"] for _, checkpoint in selected_checkpoints]
        axes.scatter(
            selected_epochs,
            selected_losses,
            s=42,
            zorder=3,
            label="Selected checkpoints",
        )
        for epoch, checkpoint in selected_checkpoints:
            axes.annotate(
                str(epoch),
                (epoch, checkpoint["loss"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axes.set_title("Training Loss")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Loss")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(run_directory / "loss.png", dpi=150)
    plt.close(figure)


def main():
    parser, arguments = parse_arguments()
    try:
        if arguments.epochs and arguments.plot_every is not None:
            raise ValueError(
                "positional EPOCH arguments cannot be used together with --plot-every"
            )

        run_directory = select_run_directory(arguments.run_dir)

        sample_x, sample_y = read_columns(
            run_directory / "samples.csv", ("x", "y")
        )
        curve_x, true_y, _ = read_columns(
            run_directory / "curve.csv", ("x", "y_true", "y_pred")
        )
        loss_epochs, losses, objective_losses = read_loss_history(
            run_directory / "loss.csv"
        )

        config = read_config(run_directory / "config.csv")
        checkpoints = read_checkpoints(run_directory / "checkpoints.csv")
        requested_epochs = list(dict.fromkeys(arguments.epochs))
        if requested_epochs:
            unavailable = [
                epoch for epoch in requested_epochs if epoch not in checkpoints
            ]
            if unavailable:
                available = ", ".join(str(epoch) for epoch in sorted(checkpoints))
                raise ValueError(
                    "requested epoch(s) are not saved checkpoints: "
                    f"{', '.join(str(epoch) for epoch in unavailable)}. "
                    f"Available checkpoint epochs: {available}"
                )
            selected_checkpoints = [
                (epoch, checkpoints[epoch]) for epoch in requested_epochs
            ]
            predictions = [
                (
                    f"Epoch {epoch}",
                    polynomial_predictions(curve_x, checkpoint["weights"]),
                )
                for epoch, checkpoint in selected_checkpoints
            ]
        else:
            selected_checkpoints = select_checkpoints_by_frequency(
                checkpoints, config, arguments.plot_every
            )
            predictions = [
                (
                    f"Epoch {epoch}",
                    polynomial_predictions(curve_x, checkpoint["weights"]),
                )
                for epoch, checkpoint in selected_checkpoints
            ]

        plot_fit(
            run_directory,
            sample_x,
            sample_y,
            curve_x,
            true_y,
            predictions,
        )
        plot_loss(
            run_directory,
            loss_epochs,
            losses,
            objective_losses,
            selected_checkpoints,
        )

        print(f"run directory: {run_directory}")
        print("Regression has no classification accuracy (acc).")
        print("Dense-curve regression metrics:")
        for label, predicted_y in predictions:
            r_squared, rmse, mae = regression_metrics(true_y, predicted_y)
            print(
                f"  {label}: R^2={r_squared:.8f}, "
                f"RMSE={rmse:.8f}, MAE={mae:.8f}"
            )
        print(f"generated: {run_directory / 'fit.png'}")
        print(f"generated: {run_directory / 'loss.png'}")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
