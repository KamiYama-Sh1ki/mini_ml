#!/usr/bin/env python
"""Compare optimizer runs (sgd/cg/newton) on the same data: convergence and final boundary."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
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


def read_rows(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def config_value(path, key):
    return read_rows(path)[0][key]


def run_label(run):
    optimizer = config_value(run / "config.csv", "optimizer")
    if optimizer == "sgd":
        learning_rate = float(config_value(run / "config.csv", "learning_rate"))
        return f"SGD(学习率 {learning_rate:g})"
    return {"cg": "非线性 CG(Wolfe 线搜索)", "newton": "Newton-CG"}[optimizer]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--run-dir", nargs="+", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output = root / "output"
    font_name = load_cjk_font()
    if font_name:
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    samples = read_rows(Path(args.data_dir) / "samples.csv")
    x = np.array([[float(r["x0"]), float(r["x1"])] for r in samples])
    labels = np.array([int(r["label"]) for r in samples])
    runs = [(run_label(Path(p)), Path(p)) for p in args.run_dir]
    colors = ("tab:blue", "tab:green", "tab:orange")

    figure, axis = plt.subplots(figsize=(8, 5))
    for (label, run), color in zip(runs, colors):
        rows = read_rows(run / "loss.csv")
        axis.plot([int(r["epoch"]) for r in rows], [float(r["loss"]) for r in rows],
                  linewidth=1.8, color=color, label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("交叉熵损失")
    axis.set_title("三种优化器收敛对比(同一 2D 数据)", fontsize=12)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=10)
    figure.tight_layout()
    figure.savefig(output / "optimizer_compare_loss.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6.5))
    for label_value, color, marker in ((0, "tab:blue", "o"), (1, "tab:red", "s")):
        mask = labels == label_value
        axis.scatter(x[mask, 0], x[mask, 1], c=color, marker=marker, s=22, alpha=0.45,
                     edgecolors="none", label=f"类别 {label_value}")
    x_min, x_max = x[:, 0].min() - 0.3, x[:, 0].max() + 0.3
    xx = np.linspace(x_min, x_max, 200)
    for (label, run), color, style in zip(runs, colors, ("-", "--", "-.")):
        final = read_rows(run / "checkpoints.csv")[-1]
        w = [float(final[f"w{i}"]) for i in range(3)]
        if (run / "standardizer.csv").exists():
            scaling = read_rows(run / "standardizer.csv")
            for i, row in enumerate(scaling):
                w[i + 1] /= float(row["scale"])
                w[0] -= w[i + 1] * float(row["mean"])
        axis.plot(xx, -(w[0] + w[1] * xx) / w[2], color=color, linestyle=style, linewidth=2.0, label=label)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(x[:, 1].min() - 0.3, x[:, 1].max() + 0.3)
    axis.set_xlabel("x0")
    axis.set_ylabel("x1")
    axis.set_title("三种优化器最终决策边界对比", fontsize=12)
    axis.legend(loc="lower left", fontsize=9, framealpha=0.9)
    axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "optimizer_compare_boundary.png", dpi=150)
    plt.close(figure)

    print(f"generated: {output / 'optimizer_compare_loss.png'}")
    print(f"generated: {output / 'optimizer_compare_boundary.png'}")


if __name__ == "__main__":
    main()
