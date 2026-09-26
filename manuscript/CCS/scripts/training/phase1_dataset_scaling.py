#!/usr/bin/env python3
"""Clean-room Phase 1 dataset scaling for AlphaPeptDeep CCS.

Commands:
  init       validate frozen inputs and create nested index permutations
  run        train exactly one fresh N x seed run
  aggregate combine completed runs, fit scaling laws, and render figures/report

The ProteomeTools test table is opened only inside the final evaluation block of
`run`, after early stopping and reloading the best-validation checkpoint.
"""

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_ROOT = Path(
    os.environ.get("CCS_SCALING_ROOT", Path.cwd() / "ccs_scaling_runs")
).resolve()
ALPHAPEPTDEEP_ROOT = Path(
    os.environ.get("ALPHAPEPTDEEP_ROOT", Path.cwd())
).resolve()
PHASE0D = EXPERIMENT_ROOT / "phase0d"
ARTIFACTS = PHASE0D / "artifacts"
TRAIN_PATH = ARTIFACTS / "training_pool.parquet"
VAL_PATH = ARTIFACTS / "validation_set.parquet"
TEST_PATH = ARTIFACTS / "proteometools_test.parquet"
SOURCE_MANIFEST = ARTIFACTS / "split_manifest.json"
OUT = EXPERIMENT_ROOT / "phase1_dataset_scaling"

N_VALUES = [500, 1000, 2500, 5600, 10000, 25000, 50000, 100000, 200000, 280000, 400000, 465356]
SEEDS = [42, 123, 2026]
EXPECTED_ROWS = {TRAIN_PATH: 465356, VAL_PATH: 51706, TEST_PATH: 168570}
EXPECTED_SHA256 = {
    TRAIN_PATH: "2618f499d23429a42c282e1dae5ce17e22dbe95a8efa7ec2ade2bab62f2cb73a",
    VAL_PATH: "bc997242ab689c002fac31b2378ba826cb9880069ce87b04c49e24c209b9ff0b",
    TEST_PATH: "8e771af8c1a196a2cce9676e7c172795264f0d1dfe4aef560ce26aba0f7b497f",
}


@dataclass(frozen=True)
class Protocol:
    model_class: str = "peptdeep.model.ccs.Model_CCS_LSTM"
    hidden_dim: int = 256
    trainable_params: int = 712428
    optimizer: str = "Adam"
    learning_rate: float = 1e-3
    loss: str = "L1Loss"
    batch_size: int = 1024
    scheduler: str = "ReduceLROnPlateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    min_lr: float = 1e-6
    early_stopping_patience: int = 10
    min_delta: float = 1e-3
    # Safety ceiling only. The initial 100-epoch dry protocol was superseded
    # because multiple runs were still improving at epochs 95-100.
    max_epochs: int = 200
    grad_clip_norm: float = 1.0
    intra_epoch_checkpoint_steps: int = 500


PROTOCOL = Protocol()


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
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


def git_revision(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def validate_frozen_inputs(read_tables: bool = True) -> dict[str, Any]:
    source = json.loads(SOURCE_MANIFEST.read_text())
    observed = {}
    for path, expected_hash in EXPECTED_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != expected_hash:
            raise RuntimeError(f"Frozen input hash mismatch: {path}: {digest} != {expected_hash}")
        observed[str(path)] = {"sha256": digest, "expected_rows": EXPECTED_ROWS[path]}
        if read_tables:
            df = pd.read_parquet(path)
            if len(df) != EXPECTED_ROWS[path]:
                raise RuntimeError(f"Frozen input row mismatch: {path}: {len(df)}")
            required = {"sequence", "mods", "mod_sites", "charge", "canonical_precursor_key", "CCS", "nAA"}
            missing = required.difference(df.columns)
            if missing:
                raise RuntimeError(f"Missing columns in {path}: {sorted(missing)}")
            if df.canonical_precursor_key.duplicated().any():
                raise RuntimeError(f"Duplicate canonical precursor in {path}")
            for col in df.select_dtypes(include=["object", "string"]).columns:
                if df[col].astype(str).str.contains("PXD017703|diaPASEF", case=False, regex=True, na=False).any():
                    raise RuntimeError(f"PXD017703 contamination in {path}:{col}")
            observed[str(path)]["rows"] = len(df)
    if source["training_pool"] != 465356 or source["validation"] != 51706:
        raise RuntimeError("Phase 0D split manifest count mismatch")
    return observed


def init_experiment() -> None:
    observed = validate_frozen_inputs(read_tables=True)
    for d in ["nested_indices", "checkpoints", "recovery", "runs", "logs", "scripts", "figures"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    key_hash = hashlib.sha256()
    keys = pd.read_parquet(TRAIN_PATH, columns=["canonical_precursor_key"])["canonical_precursor_key"]
    for key in keys:
        key_hash.update(str(key).encode())
        key_hash.update(b"\0")
    index_manifest: dict[str, Any] = {
        "training_pool_rows": len(keys),
        "training_pool_ordered_key_sha256": key_hash.hexdigest(),
        "N_values": N_VALUES,
        "seeds": SEEDS,
        "definition": "For each seed, D_N is the first N integer row indices of one fixed RandomState(seed) permutation.",
        "files": {},
    }
    for seed in SEEDS:
        p = OUT / "nested_indices" / f"seed_{seed}_permutation.npy"
        if p.exists():
            perm = np.load(p)
        else:
            perm = np.random.RandomState(seed).permutation(len(keys)).astype(np.int64)
            np.save(p, perm, allow_pickle=False)
        if perm.shape != (len(keys),) or len(np.unique(perm)) != len(keys):
            raise RuntimeError(f"Invalid permutation for seed {seed}")
        for left, right in zip(N_VALUES, N_VALUES[1:]):
            if not np.array_equal(perm[:left], perm[:right][:left]):
                raise RuntimeError(f"Nested-prefix assertion failed for seed {seed}: {left}, {right}")
        index_manifest["files"][str(seed)] = {
            "path": str(p), "sha256": sha256_file(p), "dtype": str(perm.dtype), "count": len(perm)
        }
    atomic_json(OUT / "nested_indices" / "nested_indices_manifest.json", index_manifest)
    manifest = {
        "experiment": "PHASE 1 — DATASET SCALING",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "phase0d_root": str(PHASE0D),
        "frozen_inputs": observed,
        "protocol": asdict(PROTOCOL),
        "N_values": N_VALUES,
        "seeds": SEEDS,
        "expected_runs": len(N_VALUES) * len(SEEDS),
        "prohibitions": ["pretrained weights", "fine-tuning", "warm start", "PXD017703", "test-based selection"],
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "alphapeptdeep_git": git_revision(str(ALPHAPEPTDEEP_ROOT)),
        },
    }
    atomic_json(OUT / "experiment_manifest.json", manifest)
    shutil.copy2(Path(__file__), OUT / "scripts" / Path(__file__).name)
    print(json.dumps({"status": "initialized", "output": str(OUT), "index_manifest": index_manifest}, indent=2))


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate_df(ccs_model: Any, df: pd.DataFrame, batch_size: int) -> tuple[dict[str, float], np.ndarray]:
    pred_df = ccs_model.predict(df.copy(), batch_size=batch_size, verbose=False)
    pred = pred_df[ccs_model.target_column_to_predict].to_numpy(dtype=np.float64)
    target = df["CCS"].to_numpy(dtype=np.float64)
    residual = pred - target
    abs_err = np.abs(residual)
    sst = float(np.square(target - target.mean()).sum())
    metrics = {
        "L1": float(abs_err.mean()),
        "median_relative_error": float(np.median(abs_err / np.abs(target))),
        "Pearson_r": float(np.corrcoef(pred, target)[0, 1]),
        "R2": float(1.0 - np.square(residual).sum() / sst),
    }
    return metrics, pred


def run_one(n: int, seed: int) -> None:
    if n not in N_VALUES or seed not in SEEDS:
        raise ValueError(f"N and seed must be frozen values; got N={n}, seed={seed}")
    if not (OUT / "experiment_manifest.json").is_file():
        raise RuntimeError("Run init first")
    final_path = OUT / "runs" / f"N{n}_seed{seed}.json"
    if final_path.exists():
        print(f"Already complete: {final_path}")
        return
    overall_wall_start = time.perf_counter()
    validate_frozen_inputs(read_tables=False)
    perm_path = OUT / "nested_indices" / f"seed_{seed}_permutation.npy"
    index_manifest = json.loads((OUT / "nested_indices" / "nested_indices_manifest.json").read_text())
    if sha256_file(perm_path) != index_manifest["files"][str(seed)]["sha256"]:
        raise RuntimeError(f"Nested index hash mismatch for seed {seed}")

    # TEST_PATH is deliberately not opened here. It is opened only after best-checkpoint reload below.
    train_pool = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    perm = np.load(perm_path)
    train_df = train_pool.iloc[perm[:n]].copy().reset_index(drop=True)
    if len(train_df) != n or train_df.canonical_precursor_key.duplicated().any():
        raise RuntimeError("Invalid nested training subset")
    if set(train_df.canonical_precursor_key).intersection(val_df.canonical_precursor_key):
        raise RuntimeError("Train/validation overlap")
    del train_pool

    set_all_seeds(seed)
    import torch
    from peptdeep.model.ccs import AlphaCCSModel, Model_CCS_LSTM

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each run must see exactly one CUDA GPU; set CUDA_VISIBLE_DEVICES to one device")
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_start.record()

    # Fresh constructor => fresh random model. No load occurs before training.
    ccs_model = AlphaCCSModel(model_class=Model_CCS_LSTM, dropout=0.1, device="gpu")
    ccs_model.target_column_to_train = "CCS"
    trainable_params = int(sum(p.numel() for p in ccs_model.model.parameters() if p.requires_grad))
    if trainable_params != PROTOCOL.trainable_params or ccs_model.model.__class__.__name__ != "Model_CCS_LSTM":
        raise RuntimeError(f"Architecture mismatch: {ccs_model.model.__class__.__name__}, {trainable_params}")
    ccs_model._prepare_training(train_df, PROTOCOL.learning_rate)
    optimizer = ccs_model.optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PROTOCOL.scheduler_factor,
        patience=PROTOCOL.scheduler_patience, threshold=PROTOCOL.min_delta,
        threshold_mode="abs", min_lr=PROTOCOL.min_lr,
    )

    checkpoint_path = OUT / "checkpoints" / f"N{n}_seed{seed}_best.pt"
    recovery_path = OUT / "recovery" / f"N{n}_seed{seed}_latest.pt"
    curves = []
    global_optimizer_step = 0
    examples_seen = 0
    best_val = float("inf")
    best_epoch = 0
    stopping_reference = float("inf")
    stale_epochs = 0
    rng = np.random.RandomState(seed + 104729)

    for epoch in range(1, PROTOCOL.max_epochs + 1):
        ccs_model.model.train()
        shuffled = train_df.sample(frac=1.0, random_state=int(rng.randint(0, 2**31 - 1)))
        groups = list(shuffled.groupby("nAA", sort=True))
        group_order = rng.permutation(len(groups))
        epoch_abs_sum = 0.0
        epoch_examples = 0
        for group_i in group_order:
            _, group = groups[int(group_i)]
            for start in range(0, len(group), PROTOCOL.batch_size):
                batch = group.iloc[start : start + PROTOCOL.batch_size]
                targets = ccs_model._get_targets_from_batch_df(batch)
                features = ccs_model._get_features_from_batch_df(batch)
                optimizer.zero_grad(set_to_none=True)
                predictions = ccs_model.model(*features)
                loss = ccs_model.loss_func(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ccs_model.model.parameters(), PROTOCOL.grad_clip_norm)
                optimizer.step()
                # The counter advances only here, immediately after a real optimizer.step().
                global_optimizer_step += 1
                batch_n = len(batch)
                examples_seen += batch_n
                epoch_examples += batch_n
                epoch_abs_sum += float(loss.item()) * batch_n
                if global_optimizer_step % PROTOCOL.intra_epoch_checkpoint_steps == 0:
                    torch.save({
                        "purpose": "same-run recovery only; never used for model selection",
                        "N": n, "seed": seed, "epoch": epoch,
                        "global_optimizer_step": global_optimizer_step,
                        "examples_seen": examples_seen,
                        "model_state_dict": ccs_model.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, recovery_path)
        if epoch_examples != n:
            raise RuntimeError(f"Epoch coverage mismatch: {epoch_examples} != {n}")
        train_l1 = epoch_abs_sum / epoch_examples
        val_metrics, _ = evaluate_df(ccs_model, val_df, PROTOCOL.batch_size)
        val_l1 = val_metrics["L1"]
        lr_before = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_l1)
        lr_after = float(optimizer.param_groups[0]["lr"])

        # Save every strict validation improvement; min_delta controls patience only.
        if val_l1 < best_val:
            best_val = val_l1
            best_epoch = epoch
            torch.save({
                "selection": "minimum validation L1 only",
                "N": n, "seed": seed, "epoch": epoch,
                "validation_L1": val_l1,
                "global_optimizer_step": global_optimizer_step,
                "examples_seen": examples_seen,
                "trainable_params": trainable_params,
                "model_class": ccs_model.model.__class__.__name__,
                "model_state_dict": ccs_model.model.state_dict(),
            }, checkpoint_path)
        if val_l1 < stopping_reference - PROTOCOL.min_delta:
            stopping_reference = val_l1
            stale_epochs = 0
        else:
            stale_epochs += 1
        curve = {
            "N": n, "seed": seed, "epoch": epoch, "train_L1": train_l1,
            "validation_L1": val_l1, "learning_rate_before": lr_before,
            "learning_rate_after": lr_after, "optimizer_steps": global_optimizer_step,
            "examples_seen": examples_seen, "stale_epochs": stale_epochs,
        }
        curves.append(curve)
        atomic_json(OUT / "runs" / f"N{n}_seed{seed}_progress.json", {
            "status": "training", "latest": curve, "best_epoch": best_epoch,
            "best_validation_L1": best_val, "curves": curves,
        })
        print(json.dumps(curve), flush=True)
        if stale_epochs >= PROTOCOL.early_stopping_patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=ccs_model.device)
    ccs_model.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if checkpoint["epoch"] != best_epoch or not math.isclose(checkpoint["validation_L1"], best_val):
        raise RuntimeError("Best checkpoint metadata mismatch")

    # This is the sole test-table read and sole test evaluation in a successful run.
    test_df = pd.read_parquet(TEST_PATH)
    if set(test_df.canonical_precursor_key).intersection(train_df.canonical_precursor_key):
        raise RuntimeError("Train/test overlap")
    if set(test_df.canonical_precursor_key).intersection(val_df.canonical_precursor_key):
        raise RuntimeError("Validation/test overlap")
    test_metrics, _ = evaluate_df(ccs_model, test_df, PROTOCOL.batch_size)
    gpu_end.record()
    torch.cuda.synchronize()
    gpu_seconds = float(gpu_start.elapsed_time(gpu_end) / 1000.0)
    wall_seconds = float(time.perf_counter() - overall_wall_start)
    result = {
        "status": "complete", "N": n, "seed": seed,
        "trainable_params": trainable_params, "best_epoch": best_epoch,
        "best_validation_L1": best_val,
        "true_optimizer_steps": global_optimizer_step,
        "optimizer_steps_at_best": int(checkpoint["global_optimizer_step"]),
        "examples_seen": examples_seen,
        "examples_seen_at_best": int(checkpoint["examples_seen"]),
        "test_L1": test_metrics["L1"],
        "median_relative_error": test_metrics["median_relative_error"],
        "Pearson_r": test_metrics["Pearson_r"], "R2": test_metrics["R2"],
        "wall_time_seconds": wall_seconds, "GPU_time_seconds": gpu_seconds,
        "epochs_trained": len(curves), "stopped_early": len(curves) < PROTOCOL.max_epochs,
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "nested_permutation": str(perm_path), "nested_permutation_sha256": sha256_file(perm_path),
        "train_key_sha256": hashlib.sha256("\n".join(train_df.canonical_precursor_key).encode()).hexdigest(),
        "validation_sha256": EXPECTED_SHA256[VAL_PATH], "test_sha256": EXPECTED_SHA256[TEST_PATH],
        "test_evaluations": 1, "curves": curves,
    }
    atomic_json(final_path, result)
    progress = OUT / "runs" / f"N{n}_seed{seed}_progress.json"
    if progress.exists():
        progress.unlink()
    if recovery_path.exists():
        recovery_path.unlink()
    print(json.dumps({k: v for k, v in result.items() if k != "curves"}, indent=2), flush=True)


def fit_models(agg: pd.DataFrame, runs: pd.DataFrame) -> dict[str, Any]:
    from scipy.optimize import least_squares
    from scipy.stats import t

    x = agg["N"].to_numpy(float)
    y = agg["test_L1_mean"].to_numpy(float)

    def solve(floor: bool, yy: np.ndarray = y) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        if floor:
            fun = lambda p: (p[0] + p[1] * x ** (-p[2])) - yy
            p0 = np.array([max(0.0, yy[-1] * 0.8), max(1.0, (yy[0] - yy[-1]) * x[0] ** 0.2), 0.2])
            bounds = ([0.0, 0.0, 1e-8], [float(yy.min()), np.inf, 5.0])
        else:
            fun = lambda p: (p[0] * x ** (-p[1])) - yy
            slope, intercept = np.polyfit(np.log(x), np.log(yy), 1)
            p0 = np.array([math.exp(intercept), max(1e-4, -slope)])
            bounds = ([0.0, 1e-8], [np.inf, 5.0])
        res = least_squares(fun, p0, bounds=bounds, max_nfev=100000)
        pred = yy + res.fun
        rss = float(np.square(res.fun).sum())
        return res.x, pred, rss, res.jac

    def stats(name: str, floor: bool) -> dict[str, Any]:
        p, pred, rss, jac = solve(floor)
        k = len(p); m = len(y)
        tss = float(np.square(y - y.mean()).sum())
        r2 = 1.0 - rss / tss
        aic = m * math.log(max(rss / m, np.finfo(float).tiny)) + 2 * k
        aicc = aic + 2 * k * (k + 1) / (m - k - 1)
        try:
            covariance = np.linalg.inv(jac.T @ jac) * rss / (m - k)
            se = np.sqrt(np.diag(covariance))
            crit = float(t.ppf(0.975, m - k))
            ci = np.column_stack((p - crit * se, p + crit * se))
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan); ci = np.full((k, 2), np.nan)
        rng = np.random.RandomState(20260904)
        seed_matrix = runs.pivot(index="N", columns="seed", values="test_L1").loc[N_VALUES].to_numpy()
        boot = []
        for _ in range(2000):
            sampled = seed_matrix[:, rng.randint(0, seed_matrix.shape[1], seed_matrix.shape[1])].mean(axis=1)
            try:
                boot.append(solve(floor, sampled)[0])
            except Exception:
                continue
        boot = np.asarray(boot)
        names = ["L_inf", "A", "alpha"] if floor else ["A", "alpha"]
        return {
            "model": name, "equation": "L_inf + A*N^(-alpha)" if floor else "A*N^(-alpha)",
            "parameters": {n: float(v) for n, v in zip(names, p)},
            "parametric_95CI": {n: [float(a), float(b)] for n, (a, b) in zip(names, ci)},
            "seed_bootstrap_95CI": {n: [float(a), float(b)] for n, (a, b) in zip(names, np.percentile(boot, [2.5, 97.5], axis=0).T)},
            "R2": float(r2), "RSS": rss, "AIC": float(aic), "AICc": float(aicc),
            "predictions": [float(v) for v in pred],
            "residuals": [{"N": int(nn), "observed_mean": float(yy), "predicted": float(pp), "residual": float(yy-pp)} for nn, yy, pp in zip(x, y, pred)],
        }

    return {"fit_basis": "nonlinear least squares on 12 three-seed mean Test-L1 values in original loss space", "pure_power_law": stats("pure power law", False), "floor_power_law": stats("floor power law", True)}


def aggregate() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = []; curves = []
    for n in N_VALUES:
        for seed in SEEDS:
            p = OUT / "runs" / f"N{n}_seed{seed}.json"
            if not p.is_file():
                raise RuntimeError(f"Missing completed run: {p}")
            result = json.loads(p.read_text())
            if result.get("status") != "complete" or result.get("test_evaluations") != 1:
                raise RuntimeError(f"Invalid completed run: {p}")
            row = {k: v for k, v in result.items() if k != "curves"}
            records.append(row)
            curves.extend(result["curves"])
    runs = pd.DataFrame(records).sort_values(["N", "seed"])
    curve_df = pd.DataFrame(curves).sort_values(["N", "seed", "epoch"])
    metric_cols = ["test_L1", "best_validation_L1", "median_relative_error", "Pearson_r", "R2", "true_optimizer_steps", "examples_seen", "wall_time_seconds", "GPU_time_seconds"]
    grouped = runs.groupby("N")
    agg_rows = []
    for n, g in grouped:
        row = {"N": int(n), "seeds": len(g)}
        for col in metric_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_SD"] = float(g[col].std(ddof=1))
            row[f"{col}_SEM"] = float(g[col].sem(ddof=1))
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows).sort_values("N")
    runs.to_csv(OUT / "dataset_scaling_runs.csv", index=False)
    agg.to_csv(OUT / "dataset_scaling_aggregate.csv", index=False)
    curve_df.to_csv(OUT / "dataset_scaling_learning_curves.csv", index=False)
    fits = fit_models(agg, runs)

    y = agg.test_L1_mean.to_numpy(); sd = agg.test_L1_SD.to_numpy(); x = agg.N.to_numpy()
    pure = fits["pure_power_law"]; floor = fits["floor_power_law"]
    alpha_ci = pure["seed_bootstrap_95CI"]["alpha"]
    endpoint_improvement = float(y[0] - y[-1])
    median_sd = float(np.median(sd))
    overall_down = bool(y[-1] < y[0])
    pairwise_down_fraction = float(np.mean(np.diff(y) < 0))
    improvement_exceeds_variation = bool(endpoint_improvement > median_sd)
    stable_positive_alpha = bool(pure["parameters"]["alpha"] > 0 and alpha_ci[0] > 0)
    go = bool(overall_down and improvement_exceeds_variation and stable_positive_alpha and pairwise_down_fraction >= 0.6)
    decision = {
        "decision": "GO" if go else "NO-GO",
        "criteria": {
            "overall_endpoint_decline": overall_down,
            "pairwise_decline_fraction": pairwise_down_fraction,
            "endpoint_improvement": endpoint_improvement,
            "median_seed_SD": median_sd,
            "improvement_exceeds_seed_variation": improvement_exceeds_variation,
            "pure_alpha_positive_with_bootstrap_CI": stable_positive_alpha,
            "floor_supported_by_lower_AICc": bool(floor["AICc"] < pure["AICc"]),
        },
        "rule": "GO iff endpoint declines, endpoint improvement exceeds median seed SD, pure alpha and its seed-bootstrap lower 95% bound are positive, and >=60% adjacent mean-loss changes decline.",
    }
    fits["GO_NO_GO"] = decision
    fits["N_values"] = N_VALUES
    fits["seed_SD_and_SEM"] = agg[["N", "test_L1_SD", "test_L1_SEM"]].to_dict("records")
    atomic_json(OUT / "dataset_scaling_fit.json", fits)

    def style_ax(ax):
        ax.grid(True, which="both", color="#d9d9d9", linewidth=0.6, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.plot(x, y, color="#8a8a8a", linewidth=1.25, zorder=1)
    ax.errorbar(x, y, yerr=sd, fmt="o", color="#2878b5", ecolor="#2878b5", capsize=3, markersize=5, zorder=3)
    dense = np.geomspace(x.min(), x.max(), 400)
    A = pure["parameters"]["A"]; alpha = pure["parameters"]["alpha"]
    ax.plot(dense, A*dense**(-alpha), "--", color="#f28e2b", linewidth=2, label=fr"$L={A:.3g}N^{{-{alpha:.3f}}}$" + "\n" + fr"$R^2={pure['R2']:.4f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Test Loss vs. Dataset Size"); ax.set_xlabel("Dataset Size"); ax.set_ylabel("Test L1 Loss")
    ax.legend(frameon=False); style_ax(ax)
    for ext in ["png", "svg"]:
        fig.savefig(OUT / "figures" / f"dataset_scaling.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.plot(x, y, color="#8a8a8a", linewidth=1.25, zorder=1)
    ax.errorbar(x, y, yerr=sd, fmt="o", color="#2878b5", ecolor="#2878b5", capsize=3, markersize=5, label="3-seed mean ± SD")
    linf = floor["parameters"]["L_inf"]; Af = floor["parameters"]["A"]; alphaf = floor["parameters"]["alpha"]
    ax.plot(dense, linf+Af*dense**(-alphaf), "--", color="#f28e2b", linewidth=2, label=fr"$L={linf:.3g}+{Af:.3g}N^{{-{alphaf:.3f}}}$" + "\n" + fr"$R^2={floor['R2']:.4f}$")
    ax.set_xscale("log")
    ax.set_title("Floor-fit Diagnostic"); ax.set_xlabel("Dataset Size"); ax.set_ylabel("Test L1 Loss")
    ax.legend(frameon=False); style_ax(ax)
    for ext in ["png", "svg"]:
        fig.savefig(OUT / "figures" / f"dataset_scaling_floor_diagnostic.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)

    lines = [
        "# PHASE 1 — DATASET SCALING", "", f"**Decision: {decision['decision']}**", "",
        f"- Completed runs: {len(runs)} / {len(N_VALUES)*len(SEEDS)}", f"- N: {N_VALUES}", f"- Seeds: {SEEDS}",
        f"- Model: `Model_CCS_LSTM`, hidden_dim=256, trainable parameters={PROTOCOL.trainable_params:,}",
        f"- Frozen train/validation/test SHA256 verified: PASS", f"- PXD017703 excluded: PASS", "",
        "## Test L1 (mean ± SD)", "",
    ]
    lines += [f"- N={int(r.N):,}: {r.test_L1_mean:.6f} ± {r.test_L1_SD:.6f}" for r in agg.itertuples()]
    lines += ["", "## Scaling fits", "",
        f"- Pure: A={pure['parameters']['A']:.8g}, alpha={pure['parameters']['alpha']:.6f}, R²={pure['R2']:.6f}, AIC={pure['AIC']:.3f}, AICc={pure['AICc']:.3f}",
        f"- Floor: L_inf={floor['parameters']['L_inf']:.8g}, A={floor['parameters']['A']:.8g}, alpha={floor['parameters']['alpha']:.6f}, R²={floor['R2']:.6f}, AIC={floor['AIC']:.3f}, AICc={floor['AICc']:.3f}",
        "- Confidence intervals, residuals, and seed SD/SEM: `dataset_scaling_fit.json`", "", "## GO / NO-GO evidence", "",
    ]
    lines += [f"- {k}: {v}" for k, v in decision["criteria"].items()]
    lines += ["", f"Rule: {decision['rule']}", "", "Phase 1 complete. Compute Scaling and Parameter Scaling were not started.", ""]
    (OUT / "dataset_scaling_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"completed_runs": len(runs), "decision": decision, "fits": fits}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    run = sub.add_parser("run")
    run.add_argument("--n", type=int, required=True)
    run.add_argument("--seed", type=int, required=True)
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "init": init_experiment()
    elif args.command == "run": run_one(args.n, args.seed)
    elif args.command == "aggregate": aggregate()


if __name__ == "__main__":
    main()
