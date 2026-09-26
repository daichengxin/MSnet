from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rt_utils import progressively_smoothed, read_config


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(intercept)), float(-slope)


def tolerance(settings: dict, software: str, split_name: str) -> float:
    value = settings["selection_tolerance"][software]
    return float(value.get(split_name, value.get("default", 0.10))) if isinstance(value, dict) else float(value)


def metric_frames(paths: list[Path]) -> pd.DataFrame:
    data = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    value_column = "test_mae" if "test_mae" in data.columns else "mae"
    status = data["status"].eq("ok") | data["status"].isna()
    data = data[status].copy()
    data["mae_value"] = pd.to_numeric(data[value_column], errors="coerce")
    return data[data["mae_value"].notna()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    settings = config["scaling"]
    paths = [path for item in args.input for path in (sorted(Path(item).glob("*.csv")) if Path(item).is_dir() else [Path(item)])]
    raw = metric_frames(paths)
    selected = []
    curves = []
    for keys, part in raw.groupby(["software", "split_name", "train_size"]):
        ensemble = part[part["model_index"].astype(str).eq("ensemble")]
        source = ensemble if not ensemble.empty else part
        points = source.groupby("step", as_index=False).agg(mae=("mae_value", "mean")).sort_values("step")
        smooth = progressively_smoothed(points, settings["smoothing_start_step"], settings["smoothing_base_window"], settings["smoothing_step_fraction"])
        smooth["software"] = keys[0]
        smooth["split_name"] = keys[1]
        smooth["train_size"] = keys[2]
        curves.append(smooth)
        eligible = smooth[smooth["step"] >= settings["smoothing_start_step"]]
        choice = eligible[eligible["smoothed_mae"] <= eligible["smoothed_mae"].min() + tolerance(settings, keys[0], keys[1])].iloc[0]
        selected.append({"software": keys[0], "split_name": keys[1], "train_size": keys[2], "converged_mae": choice["smoothed_mae"], "convergence_step": choice["step"]})
    summary = pd.DataFrame(selected)
    fits = []
    for keys, part in summary.groupby(["software", "split_name"]):
        coefficient, exponent = fit_power_law(part["train_size"].to_numpy(float), part["converged_mae"].to_numpy(float))
        fits.append({"software": keys[0], "split_name": keys[1], "coefficient": coefficient, "exponent": exponent})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(curves, ignore_index=True).to_csv(output.with_name("scaling_smoothed_steps.csv"), index=False)
    summary.to_csv(output.with_name("scaling_converged_mae.csv"), index=False)
    pd.DataFrame(fits).to_csv(output, index=False)


if __name__ == "__main__":
    main()
