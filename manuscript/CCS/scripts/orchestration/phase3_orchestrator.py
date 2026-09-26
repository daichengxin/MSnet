#!/usr/bin/env python3
"""Single-GPU, resource-aware, fail-closed orchestrator for Phase 3."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENT_ROOT = Path(os.environ.get("CCS_SCALING_ROOT", Path.cwd() / "ccs_scaling_runs")).resolve()
RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUT = EXPERIMENT_ROOT / "phase3_parameter_scaling"
SCRIPT = RELEASE_ROOT / "scripts" / "training" / "phase3_parameter_scaling.py"
PYTHON = os.environ.get("CCS_PYTHON", sys.executable)
ALPHAPEPTDEEP_ROOT = os.environ.get("ALPHAPEPTDEEP_ROOT", "")
HIDDEN_DIMS = [32, 64, 128, 256, 512, 1024]
SEEDS = [42, 123, 2026]
GPU = 1
MIN_FREE_MIB = 12000
MAX_START_UTIL = 15


def gpu_state() -> tuple[int, int]:
    text = subprocess.check_output([
        "nvidia-smi", f"--id={GPU}", "--query-gpu=memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    free, util = (int(v.strip()) for v in text.split(","))
    return free, util


def is_complete(hidden: int, seed: int) -> bool:
    path = OUT / "runs" / f"H{hidden}_seed{seed}.json"
    if not path.is_file():
        return False
    try:
        obj = json.loads(path.read_text())
        return (
            obj.get("status") == "complete"
            and obj.get("test_evaluations") == 1
            and obj.get("fresh_random_initialization") is True
            and obj.get("test_used_for_training_decisions") is False
        )
    except Exception:
        return False


def write_status(queue, running, state) -> None:
    payload = {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completed": sum(is_complete(h, s) for h in HIDDEN_DIMS for s in SEEDS),
        "remaining_queue": len(queue),
        "running": None if running is None else {
            "pid": running[0].pid, "hidden_dim": running[2], "seed": running[3], "gpu": GPU,
        },
        "gpu_state": {"index": GPU, "free_MiB": state[0], "utilization_percent": state[1]},
    }
    (OUT / "orchestrator_status.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    lock = OUT / "orchestrator.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise SystemExit(f"Refusing duplicate orchestrator: {lock}")
    (lock / "pid").write_text(str(os.getpid()))
    queue = [(h, s) for h in HIDDEN_DIMS for s in SEEDS if not is_complete(h, s)]
    running = None
    try:
        while queue or running is not None:
            if running is not None:
                process, log_handle, hidden, seed = running
                code = process.poll()
                if code is not None:
                    log_handle.close()
                    running = None
                    if code != 0 or not is_complete(hidden, seed):
                        failure = {
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "hidden_dim": hidden, "seed": seed, "gpu": GPU, "exit_code": code,
                            "log": str(OUT / "logs" / f"H{hidden}_seed{seed}.log"),
                            "action": "STOP; batch size and protocol were not changed",
                        }
                        (OUT / "ORCHESTRATOR_FAILED.json").write_text(json.dumps(failure, indent=2))
                        raise RuntimeError(f"Run failed closed: hidden={hidden}, seed={seed}, exit={code}")
            state = gpu_state()
            if running is None and queue and state[0] >= MIN_FREE_MIB and state[1] <= MAX_START_UTIL:
                hidden, seed = queue.pop(0)
                log_path = OUT / "logs" / f"H{hidden}_seed{seed}.log"
                log_handle = log_path.open("a", buffering=1)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(GPU)
                if ALPHAPEPTDEEP_ROOT:
                    env["PYTHONPATH"] = ALPHAPEPTDEEP_ROOT
                env["PYTHONHASHSEED"] = str(seed)
                process = subprocess.Popen(
                    [PYTHON, str(SCRIPT), "run", "--hidden", str(hidden), "--seed", str(seed)],
                    stdout=log_handle, stderr=subprocess.STDOUT, env=env,
                )
                running = (process, log_handle, hidden, seed)
                print(f"START gpu={GPU} hidden={hidden} seed={seed} pid={process.pid} free={state[0]} util={state[1]}", flush=True)
            write_status(queue, running, state)
            time.sleep(20)

        aggregate_log = (OUT / "logs" / "aggregate.log").open("w")
        code = subprocess.call(
            [PYTHON, str(SCRIPT), "aggregate"], stdout=aggregate_log, stderr=subprocess.STDOUT,
            env={**os.environ, **({"PYTHONPATH": ALPHAPEPTDEEP_ROOT} if ALPHAPEPTDEEP_ROOT else {})},
        )
        aggregate_log.close()
        if code != 0:
            failure = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stage": "aggregate", "exit_code": code, "log": str(OUT / "logs" / "aggregate.log")}
            (OUT / "ORCHESTRATOR_FAILED.json").write_text(json.dumps(failure, indent=2))
            raise RuntimeError(f"Aggregation failed: exit={code}")
        write_status([], None, gpu_state())
    finally:
        try:
            (lock / "pid").unlink(missing_ok=True)
            lock.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
