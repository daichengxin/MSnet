from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import time
from pathlib import Path

from rt_utils import append_metric, calculate_metrics, read_config, read_rows, read_tasks, stable_seed


def write_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"K.{row['seq']}.K\t{row['rt']}\n")


def parse_output(text: str) -> list[float]:
    values = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 5:
            try:
                values.append(float(fields[2]))
            except ValueError:
                pass
    return values


def amino_acids(rows: list[dict]) -> set[str]:
    return {amino_acid for row in rows for amino_acid in row["seq"]}


def largest_train_path(path: Path) -> Path:
    candidates = []
    for item in path.parent.glob("train_*.csv"):
        try:
            candidates.append((int(item.stem.split("_", 1)[1]), item))
        except ValueError:
            continue
    if not candidates:
        raise FileNotFoundError(f"No nested training CSV files found beside {path}")
    return max(candidates)[1]


def repair_amino_acid_coverage(train_rows: list[dict], test_rows: list[dict], train_path: Path, task: dict) -> list[dict]:
    required = amino_acids(test_rows)
    if required <= amino_acids(train_rows):
        return train_rows
    candidates = {row["key"]: row for row in read_rows(largest_train_path(train_path))}
    anchors = []
    anchor_keys = set()
    generator = random.Random(stable_seed("gptime_aa_repair", task["split_type"], task["split_name"], task["round_id"], task["train_size"], len(train_rows)))
    while True:
        result = anchors + [row for row in train_rows if row["key"] not in anchor_keys]
        result = result[:len(train_rows)]
        missing = required - amino_acids(result)
        if not missing:
            return result
        choices = [row for key, row in candidates.items() if key not in anchor_keys and missing.intersection(row["seq"])]
        if not choices:
            raise ValueError(f"Unable to cover amino acids {''.join(sorted(missing))}")
        generator.shuffle(choices)
        anchor = choices[0]
        anchors.append(anchor)
        anchor_keys.add(anchor["key"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()
    config = read_config(args.config)
    root = Path(config["paths"]["benchmark_root"])
    repo = Path(config["paths"]["gptime_repo"])
    python = config["paths"]["gptime_python"]
    maximum = config["benchmark"]["gptime_max_training_size"]
    tasks = [task for index, task in enumerate(read_tasks(root / "benchmark_tasks.tsv")) if index % args.n_shards == args.shard_id]
    metric_path = root / "gptime" / f"gptime_metrics_shard_{args.shard_id}.csv"
    for task in tasks:
        if int(task["train_size"]) > maximum:
            continue
        train_rows = read_rows(task["train_csv"])
        test_rows = read_rows(task["test_csv"])
        work = root / "gptime" / "runs" / task["task_id"]
        model = work / "model.pkl"
        train_rows = repair_amino_acid_coverage(train_rows, test_rows, Path(task["train_csv"]), task)
        write_tsv(train_rows, work / "train.tsv")
        write_tsv(test_rows, work / "test.tsv")
        environment = os.environ.copy()
        environment["GPTIME_OPTIMIZE_RESTARTS"] = str(config["benchmark"]["gptime_optimize_restarts"])
        started = time.time()
        if not model.exists():
            subprocess.check_call([python, "train.py", "--peptides", str(work / "train.tsv"), "--model", str(model), "--ntrain", str(len(train_rows))], cwd=repo, env=environment)
        train_seconds = time.time() - started
        started = time.time()
        process = subprocess.run([python, "test.py", "--peptides", str(work / "test.tsv"), "--model", str(model)], cwd=repo, env=environment, text=True, capture_output=True, check=True)
        y_pred = parse_output(process.stdout)
        y_true = [float(row["rt"]) for row in test_rows]
        result = calculate_metrics(y_true, y_pred)
        prediction_path = work / "predictions.txt"
        prediction_path.write_text(process.stdout, encoding="utf-8")
        append_metric(metric_path, {**task, **result, "software": "GPTime", "phase": "benchmark", "total_steps": "", "step": "", "model_index": "single", "train_count": len(train_rows), "test_count": len(test_rows), "train_time_seconds": train_seconds, "predict_time_seconds": time.time() - started, "status": "ok", "model_path": str(model), "prediction_path": str(prediction_path)})


if __name__ == "__main__":
    main()
