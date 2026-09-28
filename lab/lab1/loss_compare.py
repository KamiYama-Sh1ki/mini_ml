#!/usr/bin/env python
"""Overlay training loss curves from multiple runs on a log scale."""

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


def read_loss_history(path):
    epochs = []
    losses = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            epoch = float(row["epoch"])
            loss = float(row["loss"])
            if loss > 0.0:
                epochs.append(epoch)
                losses.append(loss)
    return epochs, losses


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--title", default="训练损失收敛对比")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.labels) != len(args.run_dirs):
        raise SystemExit("number of labels must match number of run directories")
    font_name = load_cjk_font()
    if font_name:
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(figsize=(9, 5.5))
    for run_dir, label in zip(args.run_dirs, args.labels):
        epochs, losses = read_loss_history(run_dir / "loss.csv")
        axis.plot(epochs, losses, linewidth=1.8, label=label)

    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("训练 MSE(对数坐标)")
    axis.set_title(args.title, fontsize=13)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=10)
    figure.tight_layout()
    figure.savefig(args.output, dpi=150)
    print(f"generated: {args.output}")


if __name__ == "__main__":
    main()
