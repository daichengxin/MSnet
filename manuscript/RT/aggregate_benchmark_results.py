from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rt_utils import read_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    root = Path(config["paths"]["benchmark_root"])
    files = sorted(root.glob("**/*metrics*.csv"))
    frames = [pd.read_csv(path) for path in files if path.stat().st_size]
    if not frames:
        raise FileNotFoundError("No benchmark metric files were found")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["status"] == "ok"].copy()
    ensemble = raw[raw["model_index"].astype(str).isin(["ensemble", "single"])].copy()
    if not ensemble.empty:
        raw = ensemble
    raw.to_csv(root / "benchmark_raw_metrics.csv", index=False)
    grouping = ["software", "split_type", "split_name", "round_id", "train_size"]
    per_round = raw.groupby(grouping, as_index=False).agg(r2=("r2", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"), model_count=("model_index", "count"))
    summary = per_round.groupby(["software", "split_type", "split_name", "train_size"], as_index=False).agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"), mae_mean=("mae", "mean"), mae_std=("mae", "std"), rounds=("round_id", "nunique")
    )
    summary.to_csv(root / "benchmark_r2_summary.csv", index=False)


if __name__ == "__main__":
    main()
