from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import time
from pathlib import Path

from tensorflow.keras.callbacks import Callback

from rt_utils import append_metric, calculate_metrics, read_config, read_rows, read_tasks


def write_csv(rows: list[dict], path: Path, target: bool) -> None:
    columns = ["seq", "modifications"] + (["tr"] if target else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            item = {"seq": row["seq"], "modifications": row.get("modifications", "")}
            if target:
                item["tr"] = row["rt"]
            writer.writerow(item)


def read_prediction(path: Path) -> list[float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    column = next(name for name in rows[0] if "pred" in name.lower())
    return [float(row[column]) for row in rows]


class StopAndSaveCallback(Callback):
    def __init__(self, total_steps: int, model_path: Path):
        super().__init__()
        self.total_steps = int(total_steps)
        self.model_path = Path(model_path)
        self.steps = 0
        self.saved = False

    def save(self) -> None:
        if not self.saved:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.model_path)
            self.saved = True

    def on_train_batch_end(self, batch, logs=None) -> None:
        self.steps += 1
        if self.steps >= self.total_steps:
            self.save()
            self.model.stop_training = True

    def on_train_end(self, logs=None) -> None:
        self.save()


def write_train_entry(path: Path, train_csv: Path, output: Path, total_steps: int) -> None:
    callback_name = "deeplc_stop_" + hashlib.md5(str(output).encode("utf-8")).hexdigest()[:12]
    callback_path = output / f"{callback_name}.py"
    callback_path.write_text(f'''import sys
sys.path.insert(0, r"{Path(__file__).resolve().parent}")
from run_deeplc_benchmark import StopAndSaveCallback
def make_callback():
    return StopAndSaveCallback({int(total_steps)}, r"{output / 'final_model.keras'}")
''', encoding="utf-8")
    text = f'''import os
import pandas as pd
from deeplcretrainer import deeplcretrainer
from psm_utils import PSM, PSMList
data = pd.read_csv(r"{train_csv}")
psms = PSMList(psm_list=[PSM(peptidoform=str(row.seq), spectrum_id=str(index), run="msnet", retention_time=float(row.tr)) for index, row in data.iterrows()])
os.environ["DEEPLC_STEP_EVAL_CALLBACK"] = "{callback_name}:make_callback"
os.environ["PYTHONPATH"] = r"{output}" + ":" + r"{Path(__file__).resolve().parent}" + ":" + os.environ.get("PYTHONPATH", "")
deeplcretrainer.retrain({{"msnet": psms}}, outpath=r"{output}", mods_transfer_learning=[], freeze_layers=False, n_epochs=999999, freeze_after_concat=0, verbose=True)
'''
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()
    config = read_config(args.config)
    root = Path(config["paths"]["benchmark_root"])
    repo = Path(config["paths"]["deeplc_repo"])
    python = config["paths"]["deeplc_python"]
    tasks = [task for index, task in enumerate(read_tasks(root / "benchmark_tasks.tsv")) if index % args.n_shards == args.shard_id]
    metric_path = root / "deeplc" / f"deeplc_metrics_shard_{args.shard_id}.csv"
    for task in tasks:
        train_size = int(task["train_size"])
        train_rows = read_rows(task["train_csv"])
        test_rows = read_rows(task["test_csv"])
        work = root / "deeplc" / "runs" / task["task_id"]
        work.mkdir(parents=True, exist_ok=True)
        train_csv = work / "train.csv"
        test_csv = work / "test.csv"
        write_csv(train_rows, train_csv, True)
        write_csv(test_rows, test_csv, False)
        models = sorted(work.glob("full_hc_*.keras"))
        started = time.time()
        if len(models) != 3:
            entry = work / "train_deeplc.py"
            write_train_entry(entry, train_csv, work, config["benchmark"]["deeplc_steps"][str(train_size)])
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
            subprocess.check_call([python, str(entry)], cwd=repo, env=environment)
            models = sorted(work.glob("full_hc_*.keras"))
        if len(models) != 3:
            raise RuntimeError(f"Expected three DeepLC models in {work}")
        train_seconds = time.time() - started
        model_predictions = []
        started = time.time()
        for index, model in enumerate(models):
            output = work / f"prediction_model_{index}.csv"
            subprocess.check_call([python, "-m", "deeplc", "--file_pred", str(test_csv), "--file_pred_out", str(output), "--file_model", str(model), "--log_level", "info"], cwd=repo)
            model_predictions.append(read_prediction(output))
        y_pred = [sum(values) / len(values) for values in zip(*model_predictions)]
        y_true = [float(row["rt"]) for row in test_rows]
        result = calculate_metrics(y_true, y_pred)
        prediction_path = work / "prediction_ensemble.csv"
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["seq", "observed_rt", "predicted_rt"])
            writer.writeheader()
            writer.writerows({"seq": row["seq"], "observed_rt": target, "predicted_rt": predicted} for row, target, predicted in zip(test_rows, y_true, y_pred))
        append_metric(metric_path, {**task, **result, "software": "DeepLC", "phase": "benchmark", "total_steps": config["benchmark"]["deeplc_steps"][str(train_size)], "step": config["benchmark"]["deeplc_steps"][str(train_size)], "model_index": "ensemble", "train_count": len(train_rows), "test_count": len(test_rows), "train_time_seconds": train_seconds, "predict_time_seconds": time.time() - started, "status": "ok", "model_path": ";".join(map(str, models)), "prediction_path": str(prediction_path)})


if __name__ == "__main__":
    main()
