#!/usr/bin/env python
"""Compare naive Bayes, logistic regression and the true Bayes boundary on lab2 data."""

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
    rows = read_rows(path)
    return rows[0][key]


def select_directory(base, marker):
    runs = sorted(p for p in base.iterdir() if p.is_dir() and (p / marker).exists())
    if not runs:
        raise SystemExit(f"no run containing {marker} under {base}")
    return runs[-1]


def gaussian_log_density(x, mean, covariance):
    shift = x - mean
    inverse = np.linalg.inv(covariance)
    sign, logdet = np.linalg.slogdet(covariance)
    return -0.5 * (np.einsum("...i,ij,...j->...", shift, inverse, shift) + logdet + x.shape[-1] * np.log(2 * np.pi))


def naive_bayes_log_ratio(grid, params):
    log_ratio = np.zeros(grid.shape[0])
    for row in params:
        mean = np.array([float(row[f"mean{i}"]) for i in range(2)])
        variance = np.array([float(row[f"var{i}"]) for i in range(2)])
        density = -0.5 * np.sum(np.log(2 * np.pi * variance) + (grid - mean) ** 2 / variance, axis=1)
        density += np.log(float(row["prior"]))
        log_ratio += density if row["label"] == "0" else -density
    return log_ratio


def true_bayes_log_ratio(grid, centers, directions, sigma, elongation):
    log_ratio = np.zeros(grid.shape[0])
    for row in centers:
        mean = np.array([float(row["x0"]), float(row["x1"])])
        direction = np.array([float(row[f"u{i}"]) for i in range(2)])
        covariance = sigma**2 * (np.eye(2) + (elongation**2 - 1.0) * np.outer(direction, direction))
        density = gaussian_log_density(grid, mean, covariance)
        log_ratio += density if row["label"] == "0" else -density
    return log_ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--lr-dir", default=None)
    parser.add_argument("--nb-dir", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_run = Path(args.data_dir) if args.data_dir else select_directory(root / "data", "samples.csv")
    lr_run = Path(args.lr_dir) if args.lr_dir else select_directory(root / "output", "checkpoints.csv")
    nb_run = Path(args.nb_dir) if args.nb_dir else select_directory(root / "output", "nb_params.csv")

    font_name = load_cjk_font()
    if font_name:
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    samples = read_rows(data_run / "samples.csv")
    x = np.array([[float(r["x0"]), float(r["x1"])] for r in samples])
    labels = np.array([int(r["label"]) for r in samples])

    lr_last = read_rows(lr_run / "checkpoints.csv")[-1]
    w = [float(lr_last[f"w{i}"]) for i in range(3)]
    if (lr_run / "standardizer.csv").exists():
        for i, row in enumerate(read_rows(lr_run / "standardizer.csv")):
            w[i + 1] /= float(row["scale"])
            w[0] -= w[i + 1] * float(row["mean"])
    lr_metrics = {row["split"]: row for row in read_rows(lr_run / "metrics.csv")} if (lr_run / "metrics.csv").exists() else {}
    lr_accuracy = float(lr_metrics.get("test", {"accuracy": np.mean(((w[0] + x @ np.array(w[1:])) >= 0.0).astype(int) == labels)})["accuracy"])

    nb_params = read_rows(nb_run / "nb_params.csv")
    if (nb_run / "standardizer.csv").exists():
        scaling = read_rows(nb_run / "standardizer.csv")
        for row in nb_params:
            for i, scale in enumerate(scaling):
                row[f"mean{i}"] = float(row[f"mean{i}"]) * float(scale["scale"]) + float(scale["mean"])
                row[f"var{i}"] = float(row[f"var{i}"]) * float(scale["scale"]) ** 2
    nb_ratio = naive_bayes_log_ratio(x, nb_params)
    nb_metrics = {row["split"]: row for row in read_rows(nb_run / "metrics.csv")} if (nb_run / "metrics.csv").exists() else {}
    nb_accuracy = float(nb_metrics.get("test", {"accuracy": np.mean((nb_ratio < 0.0).astype(int) == labels)})["accuracy"])

    sigma = float(config_value(data_run / "config.csv", "sigma"))
    elongation = float(config_value(data_run / "config.csv", "elong"))
    centers = read_rows(data_run / "centers.csv")
    directions = read_rows(data_run / "directions.csv")
    bayes_by_label = {row["label"]: row for row in directions}

    x_min, x_max = x[:, 0].min() - 0.3, x[:, 0].max() + 0.3
    y_min, y_max = x[:, 1].min() - 0.3, x[:, 1].max() + 0.3
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    bayes_ratio = true_bayes_log_ratio(grid, [
        {**row, "u0": bayes_by_label[row["label"]]["u0"], "u1": bayes_by_label[row["label"]]["u1"]}
        for row in centers
    ], None, sigma, elongation).reshape(grid_x.shape)
    nb_grid = naive_bayes_log_ratio(grid, nb_params).reshape(grid_x.shape)

    figure, axis = plt.subplots(figsize=(9, 7))
    for label, color, marker in ((0, "tab:blue", "o"), (1, "tab:red", "s")):
        mask = labels == label
        axis.scatter(x[mask, 0], x[mask, 1], c=color, marker=marker, s=20, alpha=0.4, edgecolors="none",
                     label=f"类别 {label}")

    axis.contour(grid_x, grid_y, bayes_ratio, levels=[0.0], colors="black", linewidths=2.2, linestyles="--")
    axis.contour(grid_x, grid_y, nb_grid, levels=[0.0], colors="tab:green", linewidths=2.2, linestyles="-.")
    xx = np.linspace(x_min, x_max, 200)
    axis.plot(xx, -(w[0] + w[1] * xx) / w[2], color="tab:purple", linewidth=2.2)

    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel("x0")
    axis.set_ylabel("x1")
    axis.set_title(f"相关特征下三种边界:真实贝叶斯(黑虚) / 朴素贝叶斯(绿点划,测试精度 {nb_accuracy:.1%}) "
                   f"/ 逻辑回归(紫,测试精度 {lr_accuracy:.1%})", fontsize=11)
    handles, _ = axis.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [
        Line2D([0], [0], color="black", linestyle="--", linewidth=2),
        Line2D([0], [0], color="tab:green", linestyle="-.", linewidth=2),
        Line2D([0], [0], color="tab:purple", linewidth=2),
    ]
    labels_legend = handles and [
        "真实贝叶斯最优(用生成参数解析计算)", "朴素贝叶斯(独立性假设)", "逻辑回归(判别式)"]
    axis.legend(handles, ["类别 0", "类别 1"] + labels_legend, loc="lower left", fontsize=9, framealpha=0.9)
    axis.grid(True, alpha=0.2)
    figure.tight_layout()
    output = root / "output" / "nb_vs_lr_boundary.png"
    figure.savefig(output, dpi=150)

    print(f"naive bayes accuracy: {nb_accuracy:.4f}")
    print(f"logistic regression accuracy: {lr_accuracy:.4f}")
    print(f"generated: {output}")


if __name__ == "__main__":
    main()
