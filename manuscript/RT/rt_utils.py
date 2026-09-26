from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


METRIC_COLUMNS = [
    "software", "phase", "split_type", "split_name", "round_id", "seed",
    "train_size", "total_steps", "step", "model_index", "train_count",
    "test_count", "mae", "rmse", "r2", "pearson_r", "spearman_rho",
    "train_time_seconds", "predict_time_seconds", "status", "model_path",
    "prediction_path"
]


def read_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def stable_seed(*parts: object) -> int:
    text = "::".join(map(str, parts))
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(value))


def row_rt(row: dict) -> float:
    for key in ("rt", "RTadjust", "normalized_rt", "RTpeptide", "retention_time", "y"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise KeyError("No RT column found")


def row_seq(row: dict) -> str:
    for key in ("seq", "sequence", "peptide", "x"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    raise KeyError("No sequence column found")


def row_key(row: dict) -> str:
    for key in ("key", "peptide_key", "sequence_key"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return row_seq(row)


def canonical(row: dict, split_type: str, split_name: str) -> dict:
    return {
        "key": row_key(row),
        "seq": row_seq(row),
        "modifications": str(row.get("modifications") or ""),
        "rt": row_rt(row),
        "split_type": split_type,
        "split_name": split_name,
    }


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["key", "seq", "modifications", "rt", "split_type", "split_name"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def deduplicate(rows: list[dict]) -> list[dict]:
    result = {}
    for row in rows:
        result.setdefault(row_key(row), row)
    return list(result.values())


def balanced_order(groups: dict[str, list[dict]], seed: int) -> list[dict]:
    shuffled = {}
    for name, rows in sorted(groups.items()):
        items = list(rows)
        random.Random(stable_seed(seed, name)).shuffle(items)
        shuffled[name] = items
    output = []
    index = 0
    while True:
        added = False
        for name in sorted(shuffled):
            if index < len(shuffled[name]):
                output.append(shuffled[name][index])
                added = True
        if not added:
            return output
        index += 1


def make_nested_train(groups: dict[str, list[dict]], excluded_keys: set[str], seed: int, sizes: list[int]) -> dict[int, list[dict]]:
    filtered = {name: [row for row in rows if row_key(row) not in excluded_keys] for name, rows in groups.items()}
    ordered = deduplicate(balanced_order(filtered, seed))
    if len(ordered) < max(sizes):
        raise ValueError(f"Training pool has {len(ordered)} rows but requires {max(sizes)}")
    return {size: ordered[:size] for size in sizes}


def calculate_metrics(y_true: list[float], y_pred: list[float]) -> dict:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Prediction count does not match test count")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson_r": float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan"),
        "spearman_rho": float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan"),
    }


def append_metric(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in METRIC_COLUMNS})


def read_tasks(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tasks(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["task_id", "phase", "split_type", "split_name", "round_id", "seed", "train_size", "train_csv", "test_csv"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_grouped_pools(root: str | Path, split_type: str) -> dict[str, list[dict]]:
    root = Path(root) / split_type
    groups = {}
    for path in sorted(root.glob("*.parquet")):
        groups[path.stem] = deduplicate([canonical(row, split_type, path.stem) for row in pd.read_parquet(path).to_dict("records")])
    if not groups:
        raise FileNotFoundError(f"No parquet pools found in {root}")
    return groups


def dynamic_steps(total_steps: int, base_interval: int) -> list[int]:
    steps = {1, total_steps}
    step = 1
    while step < total_steps:
        steps.add(step)
        step += max(base_interval, step // 10)
    return sorted(step for step in steps if step <= total_steps)


def progressively_smoothed(frame: pd.DataFrame, start_step: int, base_window: int, step_fraction: float) -> pd.DataFrame:
    data = frame.sort_values("step").copy()
    values = data["mae"].to_numpy(dtype=float)
    smoothed = []
    steps = data["step"].to_numpy(dtype=int)
    for index, step in enumerate(steps):
        if step < start_step:
            smoothed.append(float(values[index]))
        else:
            width = max(base_window, int(round(step * step_fraction)))
            smoothed.append(float(values[np.abs(steps - step) <= width].mean()))
    data["smoothed_mae"] = smoothed
    return data
