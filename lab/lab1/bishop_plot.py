#!/usr/bin/env python
"""Bishop-style side-by-side comparison of two polynomial fits."""

import argparse
import csv
import math
import sys
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


def read_samples(run_dir):
    with (run_dir / "samples.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    return [float(row["x"]) for row in rows], [float(row["y"]) for row in rows]


def read_curve(run_dir):
    with (run_dir / "curve.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    return (
        [float(row["x"]) for row in rows],
        [float(row["y_true"]) for row in rows],
        [float(row["y_pred"]) for row in rows],
    )


def read_config(run_dir):
    with (run_dir / "config.csv").open(newline="") as file:
        return next(csv.DictReader(file))


def mean_squared_error(reference, prediction):
    total = sum((ref - pred) ** 2 for ref, pred in zip(reference, prediction))
    return total / len(reference)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir_a", type=Path, help="first run directory")
    parser.add_argument("run_dir_b", type=Path, help="second run directory")
    parser.add_argument("--label-a", default=None, help="subplot title override for the first run")
    parser.add_argument("--label-b", default=None, help="subplot title override for the second run")
    parser.add_argument("--output", type=Path, default=None, help="output PNG path")
    return parser.parse_args()


def main():
    args = parse_args()
    font_name = load_cjk_font()
    if font_name:
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)

    for axis, run_dir, label in zip(
        axes, (args.run_dir_a, args.run_dir_b), (args.label_a, args.label_b)
    ):
        config = read_config(run_dir)
        sample_x, sample_y = read_samples(run_dir)
        curve_x, curve_true, curve_pred = read_curve(run_dir)
        test_mse = mean_squared_error(curve_true, curve_pred)

        axis.set_xlim(-1.05, 1.05)
        axis.set_ylim(-1.6, 1.6)
        axis.scatter(sample_x, sample_y, facecolors="none", edgecolors="dimgray", s=42,
                     zorder=3, label="含噪声的训练样本")
        axis.plot(curve_x, curve_true, color="tab:green", linestyle="--", linewidth=1.4,
                  label="真实曲线 sin(πx)")
        axis.plot(curve_x, curve_pred, color="tab:red", linewidth=1.8,
                  label="多项式拟合曲线")
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        title = label if label else f"M = {config['degree']}"
        axis.set_title(f"{title}    测试MSE = {test_mse:.3f}", fontsize=12)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="lower left", fontsize=9, framealpha=0.9)

    figure.suptitle("不同多项式阶数 M 的拟合结果对比(共轭梯度训练)", fontsize=13)
    figure.tight_layout(rect=[0, 0, 1, 0.93])

    output = args.output
    if output is None:
        output = args.run_dir_a.parent / "bishop_comparison.png"
    figure.savefig(output, dpi=150)
    print(f"generated: {output}")


if __name__ == "__main__":
    main()
