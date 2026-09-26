#!/usr/bin/env python3
"""Phase 2 compute scaling with exact optimizer-step checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(
    os.environ.get("CCS_SCALING_ROOT", Path.cwd() / "ccs_scaling_runs")
).resolve()
PHASE0 = EXPERIMENT_ROOT / "phase0d" / "artifacts"
PHASE1 = EXPERIMENT_ROOT / "phase1_dataset_scaling"
OUT = EXPERIMENT_ROOT / "phase2_compute_scaling"
TRAIN_PATH = PHASE0 / "training_pool.parquet"
VAL_PATH = PHASE0 / "validation_set.parquet"
TEST_PATH = PHASE0 / "proteometools_test.parquet"
INDEX_MANIFEST_PATH = PHASE1 / "nested_indices" / "nested_indices_manifest.json"

N_VALUES = [500, 2500, 10000, 50000, 100000, 280000, 465356]
SEEDS = [42, 123, 2026]
CHECKPOINT_STEPS = [10, 30, 50, 100, 300, 1000, 3000, 10000, 30000, 50000]
MAX_STEPS = 50000
EXPECTED_ROWS = {TRAIN_PATH: 465356, VAL_PATH: 51706, TEST_PATH: 168570}
EXPECTED_SHA256 = {
    TRAIN_PATH: "2618f499d23429a42c282e1dae5ce17e22dbe95a8efa7ec2ade2bab62f2cb73a",
    VAL_PATH: "bc997242ab689c002fac31b2378ba826cb9880069ce87b04c49e24c209b9ff0b",
    TEST_PATH: "8e771af8c1a196a2cce9676e7c172795264f0d1dfe4aef560ce26aba0f7b497f",
}
INDEX_SHA256 = {
    42: "6a53b1b290febe3237dd91a4d4f077b7f261bdde9d1934c170073210a17aa7ed",
    123: "5036d553fa9c0b1bc99c9fe14cadd9cc983d56b82a9aa47928e4cfb3f41c3035",
    2026: "9c938dc3531b45dbe8ef9dd83d164418c8718a69dc6417cb5b7ed668f886617d",
}
CONFIG = {
    "phase": "PHASE 2 — COMPUTE SCALING",
    "model_class": "peptdeep.model.ccs.Model_CCS_LSTM",
    "hidden_dim": 256,
    "trainable_params": 712428,
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "scheduler": "none (constant learning rate; independent of N, epoch, validation, and test)",
    "loss": "L1Loss",
    "batch_size": 1024,
    "gradient_clip_norm": 1.0,
    "early_stopping": False,
    "N_values": N_VALUES,
    "seeds": SEEDS,
    "checkpoint_optimizer_steps": CHECKPOINT_STEPS,
    "max_optimizer_steps": MAX_STEPS,
    "compute_definition": "global_optimizer_step increments exactly once, immediately after each executed optimizer.step()",
    "checkpoint_semantics": "model state is saved immediately after the predetermined optimizer step, including intra-epoch",
    "evaluation_semantics": "all training completes first; only then each predetermined checkpoint is loaded and evaluated on train, validation, and fixed ProteomeTools test",
    "test_role": "post-training reporting only; never used for training, scheduling, checkpoint selection, or hyperparameter tuning",
    "pretrained": False,
    "warm_start": False,
}


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_torch_save(path: Path, obj: Any) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate_frozen_inputs(read_tables: bool = True) -> dict[str, Any]:
    observed = {}
    for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        digest = sha256_file(path)
        if digest != EXPECTED_SHA256[path]:
            raise RuntimeError(f"Frozen input SHA256 mismatch: {path}: {digest}")
        item = {"path": str(path), "sha256": digest, "expected_rows": EXPECTED_ROWS[path]}
        if read_tables:
            df = pd.read_parquet(path)
            if len(df) != EXPECTED_ROWS[path]:
                raise RuntimeError(f"Frozen input row mismatch: {path}: {len(df)}")
            required = {"sequence", "mods", "mod_sites", "charge", "canonical_precursor_key", "CCS", "nAA"}
            if required.difference(df.columns):
                raise RuntimeError(f"Missing model columns: {path}")
            if df.canonical_precursor_key.duplicated().any():
                raise RuntimeError(f"Duplicate precursor: {path}")
            for col in df.select_dtypes(include=["object", "string"]).columns:
                if df[col].astype(str).str.contains("PXD017703|diaPASEF", case=False, regex=True, na=False).any():
                    raise RuntimeError(f"PXD017703 contamination: {path}:{col}")
            item["rows"] = len(df)
        observed[path.name] = item
    index_manifest = json.loads(INDEX_MANIFEST_PATH.read_text())
    if index_manifest["training_pool_rows"] != 465356 or index_manifest["seeds"] != SEEDS:
        raise RuntimeError("Phase 1 nested-index manifest mismatch")
    for seed in SEEDS:
        p = PHASE1 / "nested_indices" / f"seed_{seed}_permutation.npy"
        if sha256_file(p) != INDEX_SHA256[seed] or index_manifest["files"][str(seed)]["sha256"] != INDEX_SHA256[seed]:
            raise RuntimeError(f"Phase 1 nested index SHA256 mismatch: seed={seed}")
    return observed


def init_experiment() -> None:
    observed = validate_frozen_inputs(read_tables=True)
    for d in ["checkpoints", "runs", "logs", "scripts", "figures"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    config = dict(CONFIG)
    config.update({
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "frozen_inputs": observed,
        "phase1_nested_indices_manifest": str(INDEX_MANIFEST_PATH),
        "phase1_nested_indices_manifest_sha256": sha256_file(INDEX_MANIFEST_PATH),
        "nested_index_sha256": {str(k): v for k, v in INDEX_SHA256.items()},
        "expected_runs": len(N_VALUES) * len(SEEDS),
        "expected_checkpoint_evaluations": len(N_VALUES) * len(SEEDS) * len(CHECKPOINT_STEPS),
        "software": {"python": sys.version, "platform": platform.platform()},
    })
    atomic_json(OUT / "compute_scaling_config.json", config)
    shutil.copy2(Path(__file__), OUT / "scripts" / "phase2_compute_scaling.py")
    print(json.dumps({"status": "initialized", "output": str(OUT), "config": config}, indent=2))


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(ccs_model: Any, df: pd.DataFrame) -> dict[str, float]:
    pred_df = ccs_model.predict(df.copy(), batch_size=CONFIG["batch_size"], verbose=False)
    pred = pred_df[ccs_model.target_column_to_predict].to_numpy(np.float64)
    target = df["CCS"].to_numpy(np.float64)
    err = pred - target
    ae = np.abs(err)
    sst = float(np.square(target - target.mean()).sum())
    return {
        "L1": float(ae.mean()),
        "median_relative_error": float(np.median(ae / np.abs(target))),
        "Pearson_r": float(np.corrcoef(pred, target)[0, 1]),
        "R2": float(1.0 - np.square(err).sum() / sst),
    }


def run_one(n: int, seed: int) -> None:
    if n not in N_VALUES or seed not in SEEDS:
        raise ValueError(f"Not a frozen run: N={n}, seed={seed}")
    if not (OUT / "compute_scaling_config.json").is_file():
        raise RuntimeError("Run init first")
    final_path = OUT / "runs" / f"N{n}_seed{seed}.json"
    if final_path.exists():
        print(f"Already complete: {final_path}")
        return
    wall_start = time.perf_counter()
    validate_frozen_inputs(read_tables=False)
    index_path = PHASE1 / "nested_indices" / f"seed_{seed}_permutation.npy"
    perm = np.load(index_path)
    train_pool = pd.read_parquet(TRAIN_PATH)
    train_df = train_pool.iloc[perm[:n]].copy().reset_index(drop=True)
    del train_pool
    if len(train_df) != n or train_df.canonical_precursor_key.duplicated().any():
        raise RuntimeError("Invalid Phase 1 nested subset")
    train_key_sha = hashlib.sha256("\n".join(train_df.canonical_precursor_key).encode()).hexdigest()

    run_dir = OUT / "checkpoints" / f"N{n}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    expected_paths = {step: run_dir / f"step_{step:06d}.pt" for step in CHECKPOINT_STEPS}
    existing = [step for step, path in expected_paths.items() if path.exists()]
    training_already_complete = expected_paths[MAX_STEPS].exists()
    if existing and not training_already_complete:
        raise RuntimeError(f"Partial training checkpoints exist; archive and restart this run fresh: {existing}")

    set_seeds(seed)
    import torch
    from peptdeep.model.ccs import AlphaCCSModel, Model_CCS_LSTM

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each run must see exactly one CUDA GPU")
    ccs_model = AlphaCCSModel(model_class=Model_CCS_LSTM, dropout=0.1, device="gpu")
    ccs_model.target_column_to_train = "CCS"
    params = int(sum(p.numel() for p in ccs_model.model.parameters() if p.requires_grad))
    if params != CONFIG["trainable_params"] or ccs_model.model.__class__.__name__ != "Model_CCS_LSTM":
        raise RuntimeError(f"Architecture mismatch: {params}")
    ccs_model._prepare_training(train_df, CONFIG["learning_rate"])
    optimizer = ccs_model.optimizer
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_start.record()
    global_step = 0
    examples_seen = 0
    epoch = 0
    interval_abs_sum = 0.0
    interval_examples = 0
    checkpoint_metadata = {}
    rng = np.random.RandomState(seed + 104729)

    if not training_already_complete:
        while global_step < MAX_STEPS:
            epoch += 1
            ccs_model.model.train()
            shuffled = train_df.sample(frac=1.0, random_state=int(rng.randint(0, 2**31 - 1)))
            groups = list(shuffled.groupby("nAA", sort=True))
            for group_i in rng.permutation(len(groups)):
                group = groups[int(group_i)][1]
                for start in range(0, len(group), CONFIG["batch_size"]):
                    if global_step >= MAX_STEPS:
                        break
                    batch = group.iloc[start : start + CONFIG["batch_size"]]
                    targets = ccs_model._get_targets_from_batch_df(batch)
                    features = ccs_model._get_features_from_batch_df(batch)
                    optimizer.zero_grad(set_to_none=True)
                    predictions = ccs_model.model(*features)
                    loss = ccs_model.loss_func(predictions, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ccs_model.model.parameters(), CONFIG["gradient_clip_norm"])
                    optimizer.step()
                    global_step += 1  # exact: increment only after the real optimizer.step above
                    bn = len(batch)
                    examples_seen += bn
                    interval_examples += bn
                    interval_abs_sum += float(loss.item()) * bn
                    if global_step in expected_paths:
                        meta = {
                            "N": n, "seed": seed, "optimizer_step": global_step,
                            "epoch": epoch, "examples_seen": examples_seen,
                            "online_train_L1_since_previous_checkpoint": interval_abs_sum / interval_examples,
                            "trainable_params": params, "model_class": "Model_CCS_LSTM",
                            "nested_index_path": str(index_path),
                            "nested_index_sha256": INDEX_SHA256[seed],
                            "train_key_sha256": train_key_sha,
                            "selection": "predetermined optimizer step; no metric used",
                            "model_state_dict": ccs_model.model.state_dict(),
                        }
                        atomic_torch_save(expected_paths[global_step], meta)
                        checkpoint_metadata[global_step] = {k: v for k, v in meta.items() if k != "model_state_dict"}
                        print(json.dumps(checkpoint_metadata[global_step]), flush=True)
                        interval_abs_sum = 0.0
                        interval_examples = 0
                if global_step >= MAX_STEPS:
                    break
        if global_step != MAX_STEPS or sorted(checkpoint_metadata) != CHECKPOINT_STEPS:
            raise RuntimeError(f"Exact-step training audit failed: step={global_step}, checkpoints={sorted(checkpoint_metadata)}")
    else:
        # Training finished in an earlier invocation; only post-hoc predetermined evaluation remains.
        final_meta = torch.load(expected_paths[MAX_STEPS], map_location="cpu")
        global_step = int(final_meta["optimizer_step"])
        examples_seen = int(final_meta["examples_seen"])
        epoch = int(final_meta["epoch"])

    # Test and validation are first opened after all 50,000 optimizer updates are complete.
    val_df = pd.read_parquet(VAL_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    if set(train_df.canonical_precursor_key).intersection(val_df.canonical_precursor_key):
        raise RuntimeError("Train/validation overlap")
    if set(train_df.canonical_precursor_key).intersection(test_df.canonical_precursor_key):
        raise RuntimeError("Train/test overlap")
    if set(val_df.canonical_precursor_key).intersection(test_df.canonical_precursor_key):
        raise RuntimeError("Validation/test overlap")

    evaluations = []
    for step in CHECKPOINT_STEPS:
        cp = expected_paths[step]
        state = torch.load(cp, map_location=ccs_model.device)
        if int(state["optimizer_step"]) != step or state["N"] != n or state["seed"] != seed:
            raise RuntimeError(f"Checkpoint metadata mismatch: {cp}")
        ccs_model.model.load_state_dict(state["model_state_dict"], strict=True)
        train_metrics = evaluate(ccs_model, train_df)
        val_metrics = evaluate(ccs_model, val_df)
        test_metrics = evaluate(ccs_model, test_df)
        row = {
            "N": n, "seed": seed, "optimizer_step": step,
            "train_L1": train_metrics["L1"], "validation_L1": val_metrics["L1"],
            "test_L1": test_metrics["L1"],
            "median_relative_error": test_metrics["median_relative_error"],
            "Pearson_r": test_metrics["Pearson_r"], "R2": test_metrics["R2"],
            "checkpoint": str(cp), "checkpoint_sha256": sha256_file(cp),
            "checkpoint_epoch": int(state["epoch"]),
            "examples_seen": int(state["examples_seen"]),
            "online_train_L1_since_previous_checkpoint": float(state["online_train_L1_since_previous_checkpoint"]),
            "test_evaluations": 1,
        }
        evaluations.append(row)
        atomic_json(OUT / "runs" / f"N{n}_seed{seed}_evaluation_progress.json", {
            "status": "post_training_evaluation", "completed_checkpoints": len(evaluations),
            "evaluations": evaluations,
        })
        print(json.dumps({k: v for k, v in row.items() if "checkpoint" not in k}), flush=True)
    gpu_end.record()
    torch.cuda.synchronize()
    result = {
        "status": "complete", "N": n, "seed": seed,
        "trainable_params": params, "final_optimizer_step": global_step,
        "final_epoch_partial_or_complete": epoch, "final_examples_seen": examples_seen,
        "wall_time_seconds": float(time.perf_counter() - wall_start),
        "GPU_time_seconds": float(gpu_start.elapsed_time(gpu_end) / 1000.0),
        "nested_index_path": str(index_path), "nested_index_sha256": INDEX_SHA256[seed],
        "train_key_sha256": train_key_sha,
        "validation_sha256": EXPECTED_SHA256[VAL_PATH], "test_sha256": EXPECTED_SHA256[TEST_PATH],
        "checkpoint_steps": CHECKPOINT_STEPS,
        "test_used_during_training": False,
        "validation_used_for_training_decisions": False,
        "evaluations": evaluations,
    }
    atomic_json(final_path, result)
    progress = OUT / "runs" / f"N{n}_seed{seed}_evaluation_progress.json"
    if progress.exists():
        progress.unlink()
    print(json.dumps({k: v for k, v in result.items() if k != "evaluations"}, indent=2), flush=True)


def aggregate() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = []
    run_summaries = []
    for n in N_VALUES:
        for seed in SEEDS:
            p = OUT / "runs" / f"N{n}_seed{seed}.json"
            if not p.is_file():
                raise RuntimeError(f"Missing run: {p}")
            obj = json.loads(p.read_text())
            if obj["status"] != "complete" or obj["final_optimizer_step"] != MAX_STEPS:
                raise RuntimeError(f"Incomplete run: {p}")
            if obj["checkpoint_steps"] != CHECKPOINT_STEPS or obj["test_used_during_training"]:
                raise RuntimeError(f"Protocol violation: {p}")
            run_summaries.append({k: v for k, v in obj.items() if k != "evaluations"})
            raw.extend(obj["evaluations"])
    runs = pd.DataFrame(raw).sort_values(["N", "seed", "optimizer_step"])
    if len(runs) != 210 or runs[["N", "seed", "optimizer_step"]].drop_duplicates().shape[0] != 210:
        raise RuntimeError("Expected exactly 210 raw checkpoint evaluations")
    aggregates = []
    for (n, step), g in runs.groupby(["N", "optimizer_step"], sort=True):
        aggregates.append({
            "N": int(n), "optimizer_step": int(step), "mean_test_L1": float(g.test_L1.mean()),
            "SD_test_L1": float(g.test_L1.std(ddof=1)), "SEM_test_L1": float(g.test_L1.sem(ddof=1)),
            "n_seeds": int(len(g)), "mean_train_L1": float(g.train_L1.mean()),
            "mean_validation_L1": float(g.validation_L1.mean()),
            "mean_median_relative_error": float(g.median_relative_error.mean()),
            "mean_Pearson_r": float(g.Pearson_r.mean()), "mean_R2": float(g.R2.mean()),
        })
    agg = pd.DataFrame(aggregates).sort_values(["N", "optimizer_step"])
    runs.to_csv(OUT / "compute_scaling_runs.csv", index=False)
    agg.to_csv(OUT / "compute_scaling_aggregate.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(N_VALUES)))
    for color, n in zip(colors, N_VALUES):
        g = agg[agg.N == n]
        ax.errorbar(g.optimizer_step, g.mean_test_L1, yerr=g.SD_test_L1,
                    color=color, marker="o", markersize=3.5, linewidth=1.2,
                    capsize=2, elinewidth=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Test Loss vs. Compute (Train steps)")
    ax.set_xlabel("Compute"); ax.set_ylabel("Test L1 Loss")
    ax.grid(True, which="both", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"compute_scaling_preview.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)

    raw_min, raw_max = float(runs.test_L1.min()), float(runs.test_L1.max())
    lines = [
        "# PHASE 2 — COMPUTE SCALING", "", "**Status: PASS**", "",
        f"- Completed runs: {len(run_summaries)} / {len(N_VALUES)*len(SEEDS)}",
        f"- N curves: {N_VALUES}", f"- Seeds: {SEEDS}", f"- Checkpoints: {CHECKPOINT_STEPS}",
        f"- Raw Test-L1 range: {raw_min:.6f} to {raw_max:.6f}", "",
        "## Audit", "", "- true optimizer step: PASS", "- all checkpoints present: PASS",
        "- 3 seeds complete: PASS", "- nested indices reused: PASS", "- PXD017703 excluded: PASS",
        "- test not used for training decisions: PASS", "",
        "The fixed checkpoints were saved immediately after their exact optimizer update. Train, validation, and test metrics were computed post-training by loading each predetermined checkpoint. No early stopping or checkpoint selection was used.", "",
        "Phase 2 complete. Parameter Scaling was not started.", "",
    ]
    (OUT / "compute_scaling_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"completed_runs": len(run_summaries), "raw_rows": len(runs), "aggregate_rows": len(agg), "test_loss_range": [raw_min, raw_max]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("init")
    r = subs.add_parser("run"); r.add_argument("--n", type=int, required=True); r.add_argument("--seed", type=int, required=True)
    subs.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "init": init_experiment()
    elif args.command == "run": run_one(args.n, args.seed)
    else: aggregate()


if __name__ == "__main__":
    main()
