#!/usr/bin/env python
"""Plot train and test MSE against polynomial degree from several runs."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt

WINDOWS_CJK_FONTS = [
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


def load_cjk_font():
    for path in WINDOWS_CJK_FONTS:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=path).get_name()
    return None


def final_train_mse(run_dir):
    with (run_dir / "loss.csv").open(newline="") as file:
        return float(list(csv.DictReader(file))[-1]["loss"])


def test_mse(run_dir):
    with (run_dir / "curve.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    squared = sum(
        (float(row["y_true"]) - float(row["y_pred"])) ** 2 for row in rows
    )
    return squared / len(rows)


def config_value(run_dir, key):
    with (run_dir / "config.csv").open(newline="") as file:
        return int(next(csv.DictReader(file))[key])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--key", choices=("degree", "samples"), default="degree",
                        help="config.csv column to use as the x-axis value")
    parser.add_argument("--title", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    font_name = load_cjk_font()
    if font_name:
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    points = sorted((config_value(directory, "degree" if args.key == "degree" else "sample_count"),
                     directory) for directory in args.run_dirs)
    x_values = [value for value, _ in points]
    train = [final_train_mse(directory) for _, directory in points]
    test = [test_mse(directory) for _, directory in points]

    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.plot(x_values, train, marker="o", linewidth=1.8, label="训练 MSE")
    axis.plot(x_values, test, marker="s", linewidth=1.8, label="测试 MSE(对真实曲线)")
    axis.axhline(0.005, color="gray", linestyle=":", linewidth=1.2,
                 label="噪声下界 σ²/2 = 0.005")
    axis.set_yscale("log")
    axis.set_xlabel("多项式阶数 M" if args.key == "degree" else "训练样本数 N")
    axis.set_ylabel("MSE(对数坐标)")
    axis.set_xticks(x_values)
    default_title = "训练/测试误差随多项式阶数的变化" if args.key == "degree"         else "训练/测试误差随训练样本数的变化(M = 15, λ = 0)"
    axis.set_title(args.title or default_title, fontsize=13)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=10)
    figure.tight_layout()
    figure.savefig(args.output, dpi=150)

    print(f"generated: {args.output}")
    for value, directory in points:
        print(f"{args.key}={value}\ttrain={final_train_mse(directory):.6f}\ttest={test_mse(directory):.4f}\t({directory.name})")


if __name__ == "__main__":
    main()
