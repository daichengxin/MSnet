from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rt_utils import read_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    path = Path(args.input) if args.input else Path(config["paths"]["benchmark_root"]) / "benchmark_r2_summary.csv"
    frame = pd.read_csv(path)
    panels = [("AutoRT", "by_first_char"), ("DeepLC", "by_first_char"), ("GPTime", "by_first_char"), ("AutoRT", "by_species"), ("DeepLC", "by_species"), ("GPTime", "by_species")]
    labels = config["benchmark"]["first_character_groups"] + config["benchmark"]["species_groups"]
    figure, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    for axis, (software, split_type) in zip(axes.ravel(), panels):
        subset = frame[(frame["software"] == software) & (frame["split_type"] == split_type)]
        for name in labels:
            values = subset[subset["split_name"] == name].sort_values("train_size")
            if values.empty:
                continue
            axis.errorbar(values["train_size"], values["r2_mean"], yerr=values["r2_std"].fillna(0), marker="o", linewidth=2, capsize=3, label=name.replace("_", " "))
        axis.set_xscale("log")
        axis.set_ylim(0, 1)
        axis.set_title(f"{software}: {'By first character' if split_type == 'by_first_char' else 'By species'}", fontsize=15)
        axis.set_xlabel("Training size", fontsize=13)
        axis.set_ylabel(r"$R^2$", fontsize=13)
        axis.legend(loc="lower right", fontsize=9, ncol=1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
