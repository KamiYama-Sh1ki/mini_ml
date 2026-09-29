#!/usr/bin/env python
"""Plot lab2 2D logistic regression: data, centers, decision boundary, loss."""

import argparse
import csv
import math
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


def select_directory(base, marker):
    runs = sorted(p for p in base.iterdir() if p.is_dir() and (p / marker).exists())
    if not runs:
        raise SystemExit(f"no run containing {marker} under {base}")
    return runs[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="directory with samples.csv/centers.csv")
    parser.add_argument("--run-dir", default=None, help="training directory with checkpoints.csv/loss.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_run = Path(args.data_dir) if args.data_dir else select_directory(root / "data", "samples.csv")
    train_run = Path(args.run_dir) if args.run_dir else select_directory(root / "output", "checkpoints.csv")
    font_name = load_cjk_font()
    if font_name:
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    samples = read_rows(data_run / "samples.csv")
    if "x2" in samples[0]:
        raise SystemExit("data is 3D; use plot3d.py instead")
    centers = read_rows(data_run / "centers.csv")
    checkpoints = read_rows(train_run / "checkpoints.csv")
    x1 = np.array([float(r["x0"]) for r in samples])
    x2 = np.array([float(r["x1"]) for r in samples])
    labels = np.array([int(r["label"]) for r in samples])
    final = checkpoints[-1]
    w0 = float(final["w0"])
    w1 = float(final["w1"])
    w2 = float(final["w2"])

    z = w0 + w1 * x1 + w2 * x2
    accuracy = np.mean((z >= 0.0).astype(int) == labels)

    x_min, x_max = x1.min() - 0.4, x1.max() + 0.4
    y_min, y_max = x2.min() - 0.4, x2.max() + 0.4
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_z = w0 + w1 * grid_x + w2 * grid_y
    grid_p = 1.0 / (1.0 + np.exp(-grid_z))

    figure, axis = plt.subplots(figsize=(8, 6.5))
    contour = axis.contourf(grid_x, grid_y, grid_p, levels=np.linspace(0.0, 1.0, 21), cmap="RdBu_r", alpha=0.35)
    figure.colorbar(contour, label="P(class=1)")
    for level, style in ((0.5, "-"), (0.25, "--"), (0.75, "--")):
        axis.contour(grid_x, grid_y, grid_p, levels=[level], colors="black", linestyles=style,
                     linewidths=2.0 if level == 0.5 else 1.0, alpha=0.9 if level == 0.5 else 0.55)
    for label, color, marker in ((0, "tab:blue", "o"), (1, "tab:red", "s")):
        mask = labels == label
        axis.scatter(x1[mask], x2[mask], c=color, marker=marker, s=26, alpha=0.75,
                     edgecolors="white", linewidths=0.4, label=f"类别 {label}")
    for row, color in zip(centers, ("tab:blue", "tab:red")):
        axis.scatter(float(row["x0"]), float(row["x1"]), c=color, marker="X", s=190,
                     edgecolors="black", linewidths=1.4, zorder=5, label=f"真实中心(类 {row['label']})")
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel("x0")
    axis.set_ylabel("x1")
    axis.set_title(f"逻辑回归决策边界(2D 高斯数据,训练精度 = {accuracy:.1%})", fontsize=12)
    axis.legend(loc="lower left", fontsize=9, framealpha=0.9)
    axis.grid(True, alpha=0.2)
    figure.tight_layout()
    figure.savefig(train_run / "boundary.png", dpi=150)
    plt.close(figure)

    loss_rows = read_rows(train_run / "loss.csv")
    epochs = [int(r["epoch"]) for r in loss_rows]
    losses = [float(r["loss"]) for r in loss_rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, losses, linewidth=1.8, color="tab:blue")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("交叉熵损失")
    axis.set_title("SGD 训练损失", fontsize=12)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(train_run / "loss.png", dpi=150)
    plt.close(figure)

    print(f"accuracy: {accuracy:.4f}")
    print(f"weights: w0={w0:.4f} w1={w1:.4f} w2={w2:.4f}")
    print(f"generated: {train_run / 'boundary.png'}")
    print(f"generated: {train_run / 'loss.png'}")


if __name__ == "__main__":
    main()
