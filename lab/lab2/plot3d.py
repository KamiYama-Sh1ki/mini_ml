#!/usr/bin/env python
"""Plot lab2 in 3D: sigmoid probability surface (2 features) or decision plane (3 features)."""

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


def select_directory(base, marker):
    runs = sorted(p for p in base.iterdir() if p.is_dir() and (p / marker).exists())
    if not runs:
        raise SystemExit(f"no run containing {marker} under {base}")
    return runs[-1]


def sigmoid_surface(axis, samples, labels, w, accuracy):
    x1 = np.array([float(r["x0"]) for r in samples])
    x2 = np.array([float(r["x1"]) for r in samples])
    grid_x, grid_y = np.meshgrid(np.linspace(x1.min() - 0.3, x1.max() + 0.3, 120),
                                 np.linspace(x2.min() - 0.3, x2.max() + 0.3, 120))
    grid_p = 1.0 / (1.0 + np.exp(-(w[0] + w[1] * grid_x + w[2] * grid_y)))

    axis.plot_surface(grid_x, grid_y, grid_p, cmap="coolwarm", alpha=0.55,
                      rstride=2, cstride=2, linewidth=0, antialiased=True)
    axis.contour(grid_x, grid_y, grid_p, levels=[0.5], colors="black", linewidths=2.0,
                 offset=0.0, zdir="z")

    for label, z, color, marker in ((0, 0.0, "tab:blue", "o"), (1, 1.0, "tab:red", "s")):
        mask = labels == label
        axis.scatter(x1[mask], x2[mask], z + (0.02 if z == 0.0 else -0.02), c=color, marker=marker,
                     s=26, edgecolors="black", linewidths=0.3, depthshade=False, label=f"类别 {label}")

    axis.set_zlim(-0.05, 1.05)
    axis.set_xlabel("x0")
    axis.set_ylabel("x1")
    axis.set_zlabel("P(类别=1)")
    axis.set_title(f"逻辑回归概率曲面 σ(w·x)(训练精度 = {accuracy:.1%})", fontsize=12)
    axis.view_init(elev=24, azim=-60)


def decision_plane(axis, samples, labels, w, accuracy):
    x = np.array([[float(r[f"x{i}"]) for i in range(3)] for r in samples])
    z = w[0] + x @ np.array(w[1:])
    probability = 1.0 / (1.0 + np.exp(-z))

    bounds = np.array([[x[:, i].min() - 0.4, x[:, i].max() + 0.4] for i in range(3)])
    grid_x, grid_y = np.meshgrid(np.linspace(bounds[0, 0], bounds[0, 1], 40),
                                 np.linspace(bounds[1, 0], bounds[1, 1], 40))
    if abs(w[3]) < 1e-9:
        raise SystemExit("w3 is zero; decision plane is vertical and cannot be drawn as z(x, y)")
    grid_z = -(w[0] + w[1] * grid_x + w[2] * grid_y) / w[3]
    axis.plot_surface(grid_x, grid_y, grid_z, color="dimgray", alpha=0.22, linewidth=0)

    for label, marker in ((0, "o"), (1, "s")):
        mask = labels == label
        axis.scatter(x[mask, 0], x[mask, 1], x[mask, 2], c=probability[mask], cmap="coolwarm",
                     vmin=0.0, vmax=1.0, marker=marker, s=30, edgecolors="black", linewidths=0.3,
                     depthshade=False, label=f"类别 {label}(颜色=预测概率)")

    axis.set_xlim(bounds[0])
    axis.set_ylim(bounds[1])
    axis.set_zlim(bounds[2])
    axis.set_xlabel("x0")
    axis.set_ylabel("x1")
    axis.set_zlabel("x2")
    axis.set_title(f"逻辑回归决策平面(3D 特征,训练精度 = {accuracy:.1%})", fontsize=12)
    axis.view_init(elev=22, azim=-58)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_run = Path(args.data_dir) if args.data_dir else select_directory(root / "data", "samples.csv")
    train_run = Path(args.run_dir) if args.run_dir else select_directory(root / "output", "checkpoints.csv")

    font_name = load_cjk_font()
    if font_name:
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    samples = read_rows(data_run / "samples.csv")
    labels = np.array([int(r["label"]) for r in samples])
    checkpoints = read_rows(train_run / "checkpoints.csv")
    weight_count = sum(1 for key in checkpoints[-1] if key.startswith("w") and key[1:].isdigit())
    w = [float(checkpoints[-1][f"w{i}"]) for i in range(weight_count)]

    features = len(samples[0]) - 1
    if features == 2:
        z = w[0] + w[1] * np.array([float(r["x0"]) for r in samples]) + w[2] * np.array(
            [float(r["x1"]) for r in samples])
    else:
        x = np.array([[float(r[f"x{i}"]) for i in range(features)] for r in samples])
        z = w[0] + x @ np.array(w[1:])
    accuracy = np.mean((z >= 0.0).astype(int) == labels)

    figure = plt.figure(figsize=(9.5, 7.5))
    axis = figure.add_subplot(projection="3d")
    if features == 2:
        sigmoid_surface(axis, samples, labels, w, accuracy)
    elif features == 3:
        decision_plane(axis, samples, labels, w, accuracy)
    else:
        raise SystemExit(f"cannot visualize {features}-feature data in 3D")
    axis.legend(loc="upper left", fontsize=9)
    figure.tight_layout()
    figure.savefig(train_run / "boundary3d.png", dpi=150)

    loss_rows = read_rows(train_run / "loss.csv")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot([int(r["epoch"]) for r in loss_rows], [float(r["loss"]) for r in loss_rows],
              linewidth=1.8, color="tab:blue")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("交叉熵损失")
    axis.set_title("训练损失", fontsize=12)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(train_run / "loss.png", dpi=150)

    print(f"accuracy: {accuracy:.4f}")
    print(f"generated: {train_run / 'boundary3d.png'}")


if __name__ == "__main__":
    main()
