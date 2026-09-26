from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import time
from pathlib import Path

from rt_utils import append_metric, calculate_metrics, read_config, read_rows, read_tasks


def write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x", "y"], delimiter="\t")
        writer.writeheader()
        writer.writerows({"x": row["seq"], "y": row["rt"]} for row in rows)


def predictions(path: Path) -> tuple[list[float], Path]:
    candidates = [path / "test_1.csv", *sorted(path.glob("test*.csv"))]
    file = next(item for item in candidates if item.exists() and item.stat().st_size > 0)
    with file.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = []
    for row in rows:
        value = row.get("y_pred") or row.get("predicted_rt") or row.get("predicted") or row.get("prediction")
        if value is None:
            value = list(row.values())[-1]
        values.append(float(value))
    return values, file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()
    config = read_config(args.config)
    root = Path(config["paths"]["benchmark_root"])
    repo = Path(config["paths"]["autort_repo"])
    python = config["paths"]["autort_python"]
    tasks = [task for index, task in enumerate(read_tasks(root / "benchmark_tasks.tsv")) if index % args.n_shards == args.shard_id]
    metric_path = root / "autort" / f"autort_metrics_shard_{args.shard_id}.csv"
    for task in tasks:
        train_size = int(task["train_size"])
        train_rows = read_rows(task["train_csv"])
        test_rows = read_rows(task["test_csv"])
        work = root / "autort" / "runs" / task["task_id"]
        model = work / "model.json"
        train_tsv = work / "train.tsv"
        test_tsv = work / "test.tsv"
        work.mkdir(parents=True, exist_ok=True)
        write_tsv(train_rows, train_tsv)
        write_tsv(test_rows, test_tsv)
        epochs = math.ceil(config["benchmark"]["autort_steps"][str(train_size)] / max(1, math.ceil(train_size / 512)))
        environment = os.environ.copy()
        environment["AUTORT_STRICT_FROM_SCRATCH"] = "1"
        environment["AUTORT_TOTAL_STEPS"] = str(config["benchmark"]["autort_steps"][str(train_size)])
        environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        started = time.time()
        if not model.exists():
            command = [python, "autort.py", "train", "-i", str(train_tsv), "-o", str(work), "-m", str(repo / "models" / "base_model" / "model.json"), "-e", str(epochs), "-b", "512", "-n", "999999", "-rlr"]
            subprocess.check_call(command, cwd=repo, env=environment)
        train_seconds = time.time() - started
        started = time.time()
        output = work / "predict"
        subprocess.check_call([python, "autort.py", "predict", "-t", str(test_tsv), "-s", str(model), "-o", str(output), "-p", "test"], cwd=repo, env=environment)
        y_pred, prediction_path = predictions(output)
        y_true = [float(row["rt"]) for row in test_rows]
        result = calculate_metrics(y_true, y_pred)
        append_metric(metric_path, {**task, **result, "software": "AutoRT", "phase": "benchmark", "total_steps": config["benchmark"]["autort_steps"][str(train_size)], "step": config["benchmark"]["autort_steps"][str(train_size)], "model_index": "ensemble", "train_count": len(train_rows), "test_count": len(test_rows), "train_time_seconds": train_seconds, "predict_time_seconds": time.time() - started, "status": "ok", "model_path": str(model), "prediction_path": str(prediction_path)})


if __name__ == "__main__":
    main()
