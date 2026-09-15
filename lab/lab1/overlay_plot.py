#!/usr/bin/env python
"""Overlay fitted curves from multiple runs that share the same samples."""

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


def read_columns(path, names):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return [[float(row[name]) for row in rows] for name in names]


def mean_squared_error(reference, prediction):
    return sum((a - b) ** 2 for a, b in zip(reference, prediction)) / len(reference)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="run directories (same samples)")
    parser.add_argument("--labels", nargs="+", default=None, help="legend label per run")
    parser.add_argument("--title", default="不同正则化强度 λ 的拟合结果对比")
    parser.add_argument("--output", type=Path, required=True, help="output PNG path")
    return parser.parse_args()


def main():
    args = parse_args()
    font_name = load_cjk_font()
    if font_name:
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False
    if args.labels and len(args.labels) != len(args.run_dirs):
        raise SystemExit("number of labels must match number of run directories")

    figure, axis = plt.subplots(figsize=(9, 6))
    sample_x, sample_y = read_columns(args.run_dirs[0] / "samples.csv", ("x", "y"))
    curve_x, curve_true = read_columns(args.run_dirs[0] / "curve.csv", ("x", "y_true"))

    for run_dir, label in zip(
        args.run_dirs, args.labels or [str(directory) for directory in args.run_dirs]
    ):
        _, _, curve_pred = read_columns(run_dir / "curve.csv", ("x", "y_true", "y_pred"))
        mse = mean_squared_error(curve_true, curve_pred)
        axis.plot(curve_x, curve_pred, linewidth=1.8, label=f"{label} (测试MSE={mse:.3f})")

    axis.scatter(sample_x, sample_y, facecolors="none", edgecolors="dimgray", s=42,
                 zorder=3, label="含噪声的训练样本")
    axis.plot(curve_x, curve_true, color="black", linestyle="--", linewidth=1.4,
              zorder=2, label="真实曲线 sin(πx)")
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.6, 1.6)
    axis.set_xlabel("x")
    axis.set_ylabel("t")
    axis.set_title(args.title, fontsize=13)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower left", fontsize=9, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(args.output, dpi=150)
    print(f"generated: {args.output}")


if __name__ == "__main__":
    main()
