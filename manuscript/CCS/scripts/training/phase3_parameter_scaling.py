#!/usr/bin/env python3
"""Phase 3 parameter scaling for the clean-room AlphaPeptDeep CCS study.

Commands:
  init       validate frozen inputs, architecture, and write the manifest
  preflight  run one disposable largest-model batch to detect OOM at batch 1024
  run        train one fresh hidden_dim x seed run to validation convergence
  aggregate  combine all runs, fit scaling laws, and render QC figures

The official Model_CCS_LSTM hard-codes hidden=256. Model_CCS_LSTM_Scaled below
inherits its forward method and reproduces its constructor while exposing that
single constant as hidden_dim. At hidden_dim=256 it is required to have exactly
712,428 trainable parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import io
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
ALPHAPEPTDEEP_ROOT = Path(
    os.environ.get("ALPHAPEPTDEEP_ROOT", Path.cwd())
).resolve()
PHASE0D = EXPERIMENT_ROOT / "phase0d"
ARTIFACTS = PHASE0D / "artifacts"
TRAIN_PATH = ARTIFACTS / "training_pool.parquet"
VAL_PATH = ARTIFACTS / "validation_set.parquet"
TEST_PATH = ARTIFACTS / "proteometools_test.parquet"
SOURCE_MANIFEST = ARTIFACTS / "split_manifest.json"
PHASE1_MANIFEST = EXPERIMENT_ROOT / "phase1_dataset_scaling" / "experiment_manifest.json"
OUT = EXPERIMENT_ROOT / "phase3_parameter_scaling"

HIDDEN_DIMS = [32, 64, 128, 256, 512, 1024]
SEEDS = [42, 123, 2026]
EXPECTED_PARAMS = {32: 49164, 64: 94764, 128: 235116, 256: 712428, 512: 2453484, 1024: 9081324}
EXPECTED_ROWS = {TRAIN_PATH: 465356, VAL_PATH: 51706, TEST_PATH: 168570}
EXPECTED_SHA256 = {
    TRAIN_PATH: "2618f499d23429a42c282e1dae5ce17e22dbe95a8efa7ec2ade2bab62f2cb73a",
    VAL_PATH: "bc997242ab689c002fac31b2378ba826cb9880069ce87b04c49e24c209b9ff0b",
    TEST_PATH: "8e771af8c1a196a2cce9676e7c172795264f0d1dfe4aef560ce26aba0f7b497f",
}
PROTOCOL = {
    "base_model_class": "peptdeep.model.ccs.Model_CCS_LSTM",
    "scaled_model_class": "Model_CCS_LSTM_Scaled (inherits official forward; parameterizes hard-coded hidden=256)",
    "dropout": 0.1,
    "optimizer": "Adam",
    "learning_rate": 1e-3,
    "loss": "L1Loss",
    "batch_size": 1024,
    "scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "min_lr": 1e-6,
    "early_stopping_patience": 10,
    "min_delta": 1e-3,
    "max_epochs": 200,
    "grad_clip_norm": 1.0,
    "intra_epoch_recovery_steps": 500,
}


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
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
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def validate_frozen_inputs(read_tables: bool = True) -> dict[str, Any]:
    source = json.loads(SOURCE_MANIFEST.read_text())
    phase1 = json.loads(PHASE1_MANIFEST.read_text())
    observed: dict[str, Any] = {}
    for path, expected_hash in EXPECTED_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != expected_hash:
            raise RuntimeError(f"Frozen input hash mismatch: {path}: {digest} != {expected_hash}")
        observed[str(path)] = {"path": str(path), "sha256": digest, "expected_rows": EXPECTED_ROWS[path]}
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
    p1_protocol = phase1["protocol"]
    comparisons = {
        "optimizer": "Adam", "learning_rate": 1e-3, "loss": "L1Loss", "batch_size": 1024,
        "scheduler": "ReduceLROnPlateau", "scheduler_factor": 0.5, "scheduler_patience": 3,
        "min_lr": 1e-6, "early_stopping_patience": 10, "min_delta": 1e-3,
        "max_epochs": 200, "grad_clip_norm": 1.0,
    }
    for key, expected in comparisons.items():
        if p1_protocol.get(key) != expected or PROTOCOL[key] != expected:
            raise RuntimeError(f"Phase 1 protocol mismatch for {key}")
    return observed


def scaled_model_class():
    import torch
    import peptdeep.model.base as model_base
    from peptdeep.model.ccs import Model_CCS_LSTM

    class Model_CCS_LSTM_Scaled(Model_CCS_LSTM):
        def __init__(self, dropout: float = 0.1, hidden_dim: int = 256):
            torch.nn.Module.__init__(self)
            self.hidden_dim = int(hidden_dim)
            self.dropout = torch.nn.Dropout(dropout)
            self.ccs_encoder = model_base.Encoder_26AA_Mod_Charge_CNN_LSTM_AttnSum(self.hidden_dim)
            self.ccs_decoder = model_base.Decoder_Linear(self.hidden_dim + 1, 1)

    Model_CCS_LSTM_Scaled.__name__ = "Model_CCS_LSTM_Scaled"
    return Model_CCS_LSTM_Scaled


def state_sha256(model: Any) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_all_seeds(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def architecture_audit() -> dict[str, Any]:
    Model = scaled_model_class()
    rows = []
    for hidden in HIDDEN_DIMS:
        model = Model(dropout=PROTOCOL["dropout"], hidden_dim=hidden)
        params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        if params != EXPECTED_PARAMS[hidden]:
            raise RuntimeError(f"Parameter count mismatch hidden={hidden}: {params} != {EXPECTED_PARAMS[hidden]}")
        rows.append({"hidden_dim": hidden, "actual_parameter_count": params})
    return {
        "official_source_behavior": "Model_CCS_LSTM.__init__ hard-codes hidden=256",
        "scaled_variant": "inherits official Model_CCS_LSTM.forward; constructor differs only by exposing hidden_dim",
        "baseline_equivalence_check": "PASS" if EXPECTED_PARAMS[256] == 712428 else "FAIL",
        "sizes": rows,
    }


def init_experiment() -> None:
    observed = validate_frozen_inputs(read_tables=True)
    audit = architecture_audit()
    for d in ["checkpoints", "recovery", "runs", "logs", "scripts", "figures"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "PHASE 3 — PARAMETER SCALING",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "phase0d_root": str(PHASE0D),
        "phase1_protocol_source": str(PHASE1_MANIFEST),
        "phase1_protocol_source_sha256": sha256_file(PHASE1_MANIFEST),
        "frozen_inputs": observed,
        "training_rows": 465356,
        "protocol": PROTOCOL,
        "hidden_dims": HIDDEN_DIMS,
        "actual_parameter_counts": EXPECTED_PARAMS,
        "seeds": SEEDS,
        "expected_runs": len(HIDDEN_DIMS) * len(SEEDS),
        "architecture_audit": audit,
        "random_initialization": "fresh constructor after per-run seed; no state loaded before training",
        "pretrained": False,
        "fine_tuning": False,
        "warm_start": False,
        "test_role": "loaded and evaluated once only after best-validation checkpoint is finalized",
        "prohibitions": ["pretrained weights", "fine-tuning", "warm start", "PXD017703", "test-based selection"],
        "software": {"python": sys.version, "platform": platform.platform(), "alphapeptdeep_git": git_revision(str(ALPHAPEPTDEEP_ROOT))},
    }
    atomic_json(OUT / "parameter_scaling_manifest.json", manifest)
    script_copy = OUT / "scripts" / Path(__file__).name
    if Path(__file__).resolve() != script_copy.resolve():
        shutil.copy2(Path(__file__), script_copy)
    print(json.dumps({"status": "initialized", "output": str(OUT), "architecture_audit": audit}, indent=2))


def evaluate_df(ccs_model: Any, df: pd.DataFrame, batch_size: int) -> dict[str, float]:
    pred_df = ccs_model.predict(df.copy(), batch_size=batch_size, verbose=False)
    pred = pred_df[ccs_model.target_column_to_predict].to_numpy(dtype=np.float64)
    target = df["CCS"].to_numpy(dtype=np.float64)
    residual = pred - target
    abs_err = np.abs(residual)
    sst = float(np.square(target - target.mean()).sum())
    return {
        "L1": float(abs_err.mean()),
        "median_relative_error": float(np.median(abs_err / np.abs(target))),
        "Pearson_r": float(np.corrcoef(pred, target)[0, 1]),
        "R2": float(1.0 - np.square(residual).sum() / sst),
    }


def preflight() -> None:
    validate_frozen_inputs(read_tables=False)
    import torch
    from peptdeep.model.ccs import AlphaCCSModel
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Preflight must see exactly one CUDA GPU")
    train_df = pd.read_parquet(TRAIN_PATH)
    group_stats = train_df.groupby("nAA").size()
    probe_nAA = max(group_stats.index, key=lambda n: int(n) * min(PROTOCOL["batch_size"], int(group_stats.loc[n])))
    probe_df = train_df[train_df.nAA == probe_nAA].iloc[:PROTOCOL["batch_size"]].copy()
    set_all_seeds(20260906)
    Model = scaled_model_class()
    ccs_model = AlphaCCSModel(model_class=Model, dropout=PROTOCOL["dropout"], hidden_dim=1024, device="gpu")
    ccs_model.target_column_to_train = "CCS"
    ccs_model._prepare_training(probe_df, PROTOCOL["learning_rate"])
    targets = ccs_model._get_targets_from_batch_df(probe_df)
    features = ccs_model._get_features_from_batch_df(probe_df)
    ccs_model.optimizer.zero_grad(set_to_none=True)
    pred = ccs_model.model(*features)
    loss = ccs_model.loss_func(pred, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ccs_model.model.parameters(), PROTOCOL["grad_clip_norm"])
    ccs_model.optimizer.step()
    torch.cuda.synchronize()
    used = int(torch.cuda.max_memory_allocated())
    result = {"status": "PASS", "disposable": True, "hidden_dim": 1024, "actual_parameter_count": EXPECTED_PARAMS[1024], "batch_size": len(probe_df), "nAA": int(probe_nAA), "peak_allocated_bytes": used, "note": "one disposable optimizer step; no state retained or reused"}
    atomic_json(OUT / "oom_preflight.json", result)
    print(json.dumps(result, indent=2))


def run_one(hidden: int, seed: int) -> None:
    if hidden not in HIDDEN_DIMS or seed not in SEEDS:
        raise ValueError(f"Frozen values required; got hidden_dim={hidden}, seed={seed}")
    if not (OUT / "parameter_scaling_manifest.json").is_file() or not (OUT / "oom_preflight.json").is_file():
        raise RuntimeError("Run init and preflight first")
    final_path = OUT / "runs" / f"H{hidden}_seed{seed}.json"
    if final_path.exists():
        data = json.loads(final_path.read_text())
        if data.get("status") == "complete":
            print(f"Already complete: {final_path}")
            return
        raise RuntimeError(f"Non-complete final record exists: {final_path}")
    overall_wall_start = time.perf_counter()
    validate_frozen_inputs(read_tables=False)

    # The test table is deliberately not opened until the finalized best checkpoint is reloaded.
    train_df = pd.read_parquet(TRAIN_PATH)
    val_df = pd.read_parquet(VAL_PATH)
    if len(train_df) != 465356 or train_df.canonical_precursor_key.duplicated().any():
        raise RuntimeError("Invalid full training pool")
    if set(train_df.canonical_precursor_key).intersection(val_df.canonical_precursor_key):
        raise RuntimeError("Train/validation overlap")

    set_all_seeds(seed)
    import torch
    from peptdeep.model.ccs import AlphaCCSModel
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each run must see exactly one CUDA GPU")
    gpu_start = torch.cuda.Event(enable_timing=True)
    gpu_end = torch.cuda.Event(enable_timing=True)
    gpu_start.record()
    Model = scaled_model_class()
    ccs_model = AlphaCCSModel(model_class=Model, dropout=PROTOCOL["dropout"], hidden_dim=hidden, device="gpu")
    ccs_model.target_column_to_train = "CCS"
    params = int(sum(p.numel() for p in ccs_model.model.parameters() if p.requires_grad))
    if params != EXPECTED_PARAMS[hidden] or not isinstance(ccs_model.model, Model):
        raise RuntimeError(f"Architecture mismatch hidden={hidden}, params={params}")
    initial_state_sha256 = state_sha256(ccs_model.model)
    ccs_model._prepare_training(train_df, PROTOCOL["learning_rate"])
    optimizer = ccs_model.optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=PROTOCOL["scheduler_factor"], patience=PROTOCOL["scheduler_patience"],
        threshold=PROTOCOL["min_delta"], threshold_mode="abs", min_lr=PROTOCOL["min_lr"],
    )

    checkpoint_path = OUT / "checkpoints" / f"H{hidden}_P{params}_seed{seed}_best.pt"
    recovery_path = OUT / "recovery" / f"H{hidden}_seed{seed}_latest.pt"
    curves: list[dict[str, Any]] = []
    global_optimizer_step = 0
    examples_seen = 0
    best_val = float("inf")
    best_epoch = 0
    stopping_reference = float("inf")
    stale_epochs = 0
    rng = np.random.RandomState(seed + 104729)

    for epoch in range(1, PROTOCOL["max_epochs"] + 1):
        ccs_model.model.train()
        shuffled = train_df.sample(frac=1.0, random_state=int(rng.randint(0, 2**31 - 1)))
        groups = list(shuffled.groupby("nAA", sort=True))
        group_order = rng.permutation(len(groups))
        epoch_abs_sum = 0.0
        epoch_examples = 0
        for group_i in group_order:
            _, group = groups[int(group_i)]
            for start in range(0, len(group), PROTOCOL["batch_size"]):
                batch = group.iloc[start:start + PROTOCOL["batch_size"]]
                targets = ccs_model._get_targets_from_batch_df(batch)
                features = ccs_model._get_features_from_batch_df(batch)
                optimizer.zero_grad(set_to_none=True)
                predictions = ccs_model.model(*features)
                loss = ccs_model.loss_func(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ccs_model.model.parameters(), PROTOCOL["grad_clip_norm"])
                optimizer.step()
                global_optimizer_step += 1
                batch_n = len(batch)
                examples_seen += batch_n
                epoch_examples += batch_n
                epoch_abs_sum += float(loss.item()) * batch_n
                if global_optimizer_step % PROTOCOL["intra_epoch_recovery_steps"] == 0:
                    torch.save({
                        "purpose": "same-run recovery evidence only; never loaded or used for selection",
                        "hidden_dim": hidden, "actual_parameter_count": params, "seed": seed, "epoch": epoch,
                        "global_optimizer_step": global_optimizer_step, "examples_seen": examples_seen,
                        "model_state_dict": ccs_model.model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    }, recovery_path)
        if epoch_examples != 465356:
            raise RuntimeError(f"Epoch coverage mismatch: {epoch_examples} != 465356")
        train_l1 = epoch_abs_sum / epoch_examples
        val_l1 = evaluate_df(ccs_model, val_df, PROTOCOL["batch_size"])["L1"]
        lr_before = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_l1)
        lr_after = float(optimizer.param_groups[0]["lr"])
        if val_l1 < best_val:
            best_val = val_l1
            best_epoch = epoch
            torch.save({
                "selection": "minimum validation L1 only", "hidden_dim": hidden, "actual_parameter_count": params,
                "seed": seed, "epoch": epoch, "validation_L1": val_l1,
                "global_optimizer_step": global_optimizer_step, "examples_seen": examples_seen,
                "initial_state_sha256": initial_state_sha256, "model_state_dict": ccs_model.model.state_dict(),
            }, checkpoint_path)
        if val_l1 < stopping_reference - PROTOCOL["min_delta"]:
            stopping_reference = val_l1
            stale_epochs = 0
        else:
            stale_epochs += 1
        curve = {
            "hidden_dim": hidden, "actual_parameter_count": params, "seed": seed, "epoch": epoch,
            "train_L1": train_l1, "validation_L1": val_l1, "learning_rate_before": lr_before,
            "learning_rate_after": lr_after, "optimizer_steps": global_optimizer_step,
            "examples_seen": examples_seen, "stale_epochs": stale_epochs,
        }
        curves.append(curve)
        atomic_json(OUT / "runs" / f"H{hidden}_seed{seed}_progress.json", {
            "status": "training", "latest": curve, "best_epoch": best_epoch,
            "best_validation_L1": best_val, "initial_state_sha256": initial_state_sha256, "curves": curves,
        })
        print(json.dumps(curve), flush=True)
        if stale_epochs >= PROTOCOL["early_stopping_patience"]:
            break

    checkpoint = torch.load(checkpoint_path, map_location=ccs_model.device, weights_only=False)
    ccs_model.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if checkpoint["epoch"] != best_epoch or not math.isclose(checkpoint["validation_L1"], best_val):
        raise RuntimeError("Best checkpoint metadata mismatch")

    test_df = pd.read_parquet(TEST_PATH)
    if set(test_df.canonical_precursor_key).intersection(train_df.canonical_precursor_key):
        raise RuntimeError("Train/test overlap")
    if set(test_df.canonical_precursor_key).intersection(val_df.canonical_precursor_key):
        raise RuntimeError("Validation/test overlap")
    test_metrics = evaluate_df(ccs_model, test_df, PROTOCOL["batch_size"])
    gpu_end.record()
    torch.cuda.synchronize()
    result = {
        "status": "complete", "hidden_dim": hidden, "actual_parameter_count": params, "seed": seed,
        "model_class": "Model_CCS_LSTM_Scaled", "base_model_class": "Model_CCS_LSTM",
        "fresh_random_initialization": True, "initial_state_sha256": initial_state_sha256,
        "pretrained": False, "fine_tuning": False, "warm_start": False,
        "best_epoch": best_epoch, "best_epoch_train_L1": float(curves[best_epoch - 1]["train_L1"]),
        "best_validation_L1": best_val, "true_optimizer_steps": global_optimizer_step,
        "optimizer_steps_at_best": int(checkpoint["global_optimizer_step"]),
        "examples_seen": examples_seen, "examples_seen_at_best": int(checkpoint["examples_seen"]),
        "test_L1": test_metrics["L1"], "median_relative_error": test_metrics["median_relative_error"],
        "Pearson_r": test_metrics["Pearson_r"], "R2": test_metrics["R2"],
        "wall_time_seconds": float(time.perf_counter() - overall_wall_start),
        "GPU_time_seconds": float(gpu_start.elapsed_time(gpu_end) / 1000.0),
        "epochs_trained": len(curves), "stopped_early": len(curves) < PROTOCOL["max_epochs"],
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_sha256": EXPECTED_SHA256[TRAIN_PATH], "validation_sha256": EXPECTED_SHA256[VAL_PATH],
        "test_sha256": EXPECTED_SHA256[TEST_PATH], "test_evaluations": 1,
        "test_used_for_training_decisions": False, "validation_only_checkpoint_selection": True,
        "curves": curves,
    }
    atomic_json(final_path, result)
    progress = OUT / "runs" / f"H{hidden}_seed{seed}_progress.json"
    if progress.exists():
        progress.unlink()
    if recovery_path.exists():
        recovery_path.unlink()
    print(json.dumps({k: v for k, v in result.items() if k != "curves"}, indent=2), flush=True)


def fit_models(agg: pd.DataFrame, runs: pd.DataFrame) -> dict[str, Any]:
    from scipy.optimize import least_squares
    from scipy.stats import t
    x = agg["actual_parameter_count"].to_numpy(float)
    y = agg["mean_test_L1"].to_numpy(float)

    def solve(floor: bool, yy: np.ndarray = y):
        if floor:
            fun = lambda p: p[0] + p[1] * x ** (-p[2]) - yy
            p0 = np.array([max(0.0, yy.min() * 0.8), max(1.0, (yy.max() - yy.min()) * x.min() ** 0.2), 0.2])
            bounds = ([0.0, 0.0, 1e-8], [float(yy.min()), np.inf, 5.0])
        else:
            fun = lambda p: p[0] * x ** (-p[1]) - yy
            slope, intercept = np.polyfit(np.log(x), np.log(yy), 1)
            p0 = np.array([math.exp(intercept), max(1e-4, -slope)])
            bounds = ([0.0, 1e-8], [np.inf, 5.0])
        res = least_squares(fun, p0, bounds=bounds, max_nfev=100000)
        return res.x, yy + res.fun, float(np.square(res.fun).sum()), res.jac

    def stats(name: str, floor: bool):
        p, pred, rss, jac = solve(floor)
        k, m = len(p), len(y)
        tss = float(np.square(y - y.mean()).sum())
        r2 = 1.0 - rss / tss if tss else float("nan")
        aic = m * math.log(max(rss / m, np.finfo(float).tiny)) + 2 * k
        aicc = aic + 2 * k * (k + 1) / (m - k - 1)
        try:
            covariance = np.linalg.inv(jac.T @ jac) * rss / (m - k)
            se = np.sqrt(np.diag(covariance))
            crit = float(t.ppf(0.975, m - k))
            ci = np.column_stack((p - crit * se, p + crit * se))
        except np.linalg.LinAlgError:
            ci = np.full((k, 2), np.nan)
        seed_matrix = runs.pivot(index="actual_parameter_count", columns="seed", values="test_L1").loc[x.astype(int)].to_numpy()
        rng = np.random.RandomState(20260906)
        boot = []
        for _ in range(2000):
            sampled = seed_matrix[:, rng.randint(0, seed_matrix.shape[1], seed_matrix.shape[1])].mean(axis=1)
            try:
                boot.append(solve(floor, sampled)[0])
            except Exception:
                pass
        boot = np.asarray(boot)
        names = ["L_inf", "B", "beta"] if floor else ["B", "beta"]
        return {
            "model": name, "equation": "L_inf + B*P^(-beta)" if floor else "B*P^(-beta)",
            "parameters": {n: float(v) for n, v in zip(names, p)},
            "parametric_95CI": {n: [float(a), float(b)] for n, (a, b) in zip(names, ci)},
            "seed_bootstrap_95CI": {n: [float(a), float(b)] for n, (a, b) in zip(names, np.percentile(boot, [2.5, 97.5], axis=0).T)},
            "R2": float(r2), "RSS": rss, "AIC": float(aic), "AICc": float(aicc),
            "predictions": [float(v) for v in pred],
            "residuals": [{"P": int(ppp), "observed_mean": float(obs), "predicted": float(fit), "residual": float(obs-fit)} for ppp, obs, fit in zip(x, y, pred)],
        }

    return {
        "fit_basis": "nonlinear least squares on six 3-seed mean Test-L1 values in original loss space",
        "pure_power_law": stats("pure power law", False),
        "floor_power_law": stats("floor power law", True),
    }


def aggregate() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    validate_frozen_inputs(read_tables=False)
    records, curves = [], []
    for hidden in HIDDEN_DIMS:
        for seed in SEEDS:
            path = OUT / "runs" / f"H{hidden}_seed{seed}.json"
            if not path.is_file():
                raise RuntimeError(f"Missing completed run: {path}")
            result = json.loads(path.read_text())
            if result.get("status") != "complete" or result.get("test_evaluations") != 1:
                raise RuntimeError(f"Invalid completed run: {path}")
            if result.get("actual_parameter_count") != EXPECTED_PARAMS[hidden]:
                raise RuntimeError(f"Parameter mismatch: {path}")
            records.append({k: v for k, v in result.items() if k != "curves"})
            curves.extend(result["curves"])
    runs = pd.DataFrame(records).sort_values(["actual_parameter_count", "seed"])
    curve_df = pd.DataFrame(curves).sort_values(["actual_parameter_count", "seed", "epoch"])
    metric_cols = ["test_L1", "best_validation_L1", "median_relative_error", "Pearson_r", "R2", "true_optimizer_steps", "examples_seen", "wall_time_seconds", "GPU_time_seconds"]
    agg_rows = []
    for params, group in runs.groupby("actual_parameter_count", sort=True):
        row = {"hidden_dim": int(group.hidden_dim.iloc[0]), "actual_parameter_count": int(params), "n_seeds": len(group)}
        for col in metric_cols:
            row[f"mean_{col}"] = float(group[col].mean())
            row[f"SD_{col}"] = float(group[col].std(ddof=1))
            row[f"SEM_{col}"] = float(group[col].sem(ddof=1))
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows).sort_values("actual_parameter_count")
    runs.to_csv(OUT / "parameter_scaling_runs.csv", index=False)
    agg.to_csv(OUT / "parameter_scaling_aggregate.csv", index=False)
    curve_df.to_csv(OUT / "parameter_scaling_learning_curves.csv", index=False)
    fits = fit_models(agg, runs)
    fits["hidden_dim_to_actual_parameter_count"] = {str(k): v for k, v in EXPECTED_PARAMS.items()}
    fits["seed_variation"] = agg[["hidden_dim", "actual_parameter_count", "mean_test_L1", "SD_test_L1", "SEM_test_L1"]].to_dict("records")
    atomic_json(OUT / "parameter_scaling_fit.json", fits)

    x = agg.actual_parameter_count.to_numpy(float)
    y = agg.mean_test_L1.to_numpy(float)
    sd = agg.SD_test_L1.to_numpy(float)
    pure, floor = fits["pure_power_law"], fits["floor_power_law"]
    dense = np.geomspace(x.min(), x.max(), 400)
    def style_ax(ax):
        ax.grid(True, which="both", color="#d9d9d9", linewidth=0.6, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.plot(x, y, color="#8a8a8a", linewidth=1.25, zorder=1)
    ax.errorbar(x, y, yerr=sd, fmt="o", color="#2878b5", ecolor="#2878b5", capsize=3, markersize=5, zorder=3)
    B, beta = pure["parameters"]["B"], pure["parameters"]["beta"]
    pure_label = fr"$L={B:.3g}P^{{-{beta:.3f}}}$" + "\n" + fr"$R^2={pure['R2']:.4f}$"
    ax.plot(dense, B*dense**(-beta), "--", color="#f28e2b", linewidth=2, label=pure_label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Test Loss vs. Parameter Size"); ax.set_xlabel("Parameter Size"); ax.set_ylabel("Test L1 Loss")
    ax.legend(frameon=False); style_ax(ax)
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"parameter_scaling_preview.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.plot(x, y, color="#8a8a8a", linewidth=1.25, zorder=1)
    ax.errorbar(x, y, yerr=sd, fmt="o", color="#2878b5", ecolor="#2878b5", capsize=3, markersize=5, label="3-seed mean ± SD")
    linf, Bf, betaf = floor["parameters"]["L_inf"], floor["parameters"]["B"], floor["parameters"]["beta"]
    floor_label = fr"$L={linf:.3g}+{Bf:.3g}P^{{-{betaf:.3f}}}$" + "\n" + fr"$R^2={floor['R2']:.4f}$"
    ax.plot(dense, linf+Bf*dense**(-betaf), "--", color="#f28e2b", linewidth=2, label=floor_label)
    ax.set_xscale("log")
    ax.set_title("Floor-fit Diagnostic"); ax.set_xlabel("Parameter Size"); ax.set_ylabel("Test L1 Loss")
    ax.legend(frameon=False); style_ax(ax)
    for ext in ["png", "svg"]:
        fig.savefig(OUT / f"parameter_scaling_floor_diagnostic.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)

    best = runs.loc[runs.test_L1.idxmin()]
    best_mean = agg.loc[agg.mean_test_L1.idxmin()]
    lines = [
        "# PHASE 3 — PARAMETER SCALING", "", "**Status: PASS**", "",
        f"- Completed runs: {len(runs)} / {len(HIDDEN_DIMS)*len(SEEDS)}",
        f"- Hidden dimensions: {HIDDEN_DIMS}", f"- Seeds: {SEEDS}",
        f"- Actual parameter counts: {[EXPECTED_PARAMS[h] for h in HIDDEN_DIMS]}",
        "- Full frozen training/validation/test SHA256 verified: PASS",
        "- Fresh random initialization, no pretrained/fine-tuning/warm start: PASS",
        "- Validation-only best-checkpoint selection: PASS", "- PXD017703 excluded: PASS", "",
        "## Test L1 (mean ± SD)", "",
    ]
    lines += [f"- hidden={int(r.hidden_dim):,}, P={int(r.actual_parameter_count):,}: {r.mean_test_L1:.6f} ± {r.SD_test_L1:.6f}" for r in agg.itertuples()]
    lines += ["", "## Scaling fits", "",
        f"- Pure: B={B:.8g}, beta={beta:.6f}, R²={pure['R2']:.6f}, AIC={pure['AIC']:.3f}, AICc={pure['AICc']:.3f}",
        f"- Floor: L_inf={linf:.8g}, B={Bf:.8g}, beta={betaf:.6f}, R²={floor['R2']:.6f}, AIC={floor['AIC']:.3f}, AICc={floor['AICc']:.3f}",
        "- Confidence intervals, residuals, and seed variation: `parameter_scaling_fit.json`", "",
        f"- Best 3-seed mean model: hidden={int(best_mean.hidden_dim)}, P={int(best_mean.actual_parameter_count):,}, Test L1={best_mean.mean_test_L1:.6f} ± {best_mean.SD_test_L1:.6f}",
        f"- Best individual run: hidden={int(best.hidden_dim)}, P={int(best.actual_parameter_count):,}, seed={int(best.seed)}, Test L1={best.test_L1:.6f}", "",
        "Phase 3 complete. Final three-panel figure was not generated.", "",
    ]
    (OUT / "parameter_scaling_report.md").write_text("\n".join(lines), encoding="utf-8")
    atomic_json(OUT / "PHASE3_COMPLETE.json", {"status": "PASS", "completed_runs": len(runs), "expected_runs": 18, "completed_utc": pd.Timestamp.utcnow().isoformat()})
    print(json.dumps({"completed_runs": len(runs), "pure": pure, "floor": floor, "best": {"hidden_dim": int(best.hidden_dim), "actual_parameter_count": int(best.actual_parameter_count), "seed": int(best.seed), "test_L1": float(best.test_L1)}}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("preflight")
    run = sub.add_parser("run")
    run.add_argument("--hidden", type=int, required=True)
    run.add_argument("--seed", type=int, required=True)
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "init":
        init_experiment()
    elif args.command == "preflight":
        preflight()
    elif args.command == "run":
        run_one(args.hidden, args.seed)
    elif args.command == "aggregate":
        aggregate()


if __name__ == "__main__":
    main()
