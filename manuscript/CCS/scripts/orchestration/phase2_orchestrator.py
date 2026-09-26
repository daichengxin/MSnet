#!/usr/bin/env python3
"""Fail-closed single-GPU orchestrator for Phase 2."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT_ROOT = Path(os.environ.get("CCS_SCALING_ROOT", Path.cwd() / "ccs_scaling_runs")).resolve()
RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUT = EXPERIMENT_ROOT / "phase2_compute_scaling"
SCRIPT = RELEASE_ROOT / "scripts" / "training" / "phase2_compute_scaling.py"
PYTHON = os.environ.get("CCS_PYTHON", sys.executable)
ALPHAPEPTDEEP_ROOT = os.environ.get("ALPHAPEPTDEEP_ROOT", "")
N_VALUES = [500, 2500, 10000, 50000, 100000, 280000, 465356]
SEEDS = [42, 123, 2026]
GPU = 1
MIN_FREE_MIB = 12000
MAX_START_UTIL = 15


def gpu_state():
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    return [tuple(int(v.strip()) for v in line.split(",")) for line in text.strip().splitlines()]


def complete(n, seed):
    p = OUT / "runs" / f"N{n}_seed{seed}.json"
    try:
        o = json.loads(p.read_text())
        return o.get("status") == "complete" and o.get("final_optimizer_step") == 50000 and len(o.get("evaluations", [])) == 10
    except Exception:
        return False


def main():
    lock = OUT / "orchestrator.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise SystemExit(f"Refusing duplicate orchestrator: {lock}")
    (lock / "pid").write_text(str(os.getpid()))
    queue = [(n, s) for n in N_VALUES for s in SEEDS if not complete(n, s)]
    process = None
    log_handle = None
    current = None
    try:
        while queue or process is not None:
            if process is not None:
                code = process.poll()
                if code is not None:
                    log_handle.close()
                    n, seed = current
                    if code != 0 or not complete(n, seed):
                        failure = {"N": n, "seed": seed, "gpu": GPU, "exit_code": code,
                                   "log": str(OUT / "logs" / f"N{n}_seed{seed}.log")}
                        (OUT / "ORCHESTRATOR_FAILED.json").write_text(json.dumps(failure, indent=2))
                        raise RuntimeError(f"Run failed closed: {failure}")
                    process = None; log_handle = None; current = None
            states = gpu_state()
            if process is None and queue:
                _, free, util = next(x for x in states if x[0] == GPU)
                if free >= MIN_FREE_MIB and util <= MAX_START_UTIL:
                    n, seed = queue.pop(0)
                    path = OUT / "logs" / f"N{n}_seed{seed}.log"
                    log_handle = path.open("a", buffering=1)
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(GPU),
                           "PYTHONHASHSEED": str(seed)}
                    if ALPHAPEPTDEEP_ROOT:
                        env["PYTHONPATH"] = ALPHAPEPTDEEP_ROOT
                    process = subprocess.Popen(
                        [PYTHON, str(SCRIPT), "run", "--n", str(n), "--seed", str(seed)],
                        stdout=log_handle, stderr=subprocess.STDOUT, env=env,
                    )
                    current = (n, seed)
                    print(f"START gpu={GPU} N={n} seed={seed} pid={process.pid}", flush=True)
            status = {
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "completed": sum(complete(n, s) for n in N_VALUES for s in SEEDS),
                "remaining_queue": len(queue),
                "running": None if process is None else {"pid": process.pid, "N": current[0], "seed": current[1], "gpu": GPU},
                "gpu_state": states,
            }
            (OUT / "orchestrator_status.json").write_text(json.dumps(status, indent=2))
            time.sleep(20)
        with (OUT / "logs" / "aggregate.log").open("w") as f:
            code = subprocess.call([PYTHON, str(SCRIPT), "aggregate"], stdout=f, stderr=subprocess.STDOUT,
                                   env={**os.environ, **({"PYTHONPATH": ALPHAPEPTDEEP_ROOT} if ALPHAPEPTDEEP_ROOT else {})})
        if code != 0:
            raise RuntimeError(f"Aggregation failed: exit={code}")
        (OUT / "PHASE2_COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    finally:
        if log_handle is not None:
            log_handle.close()
        try:
            (lock / "pid").unlink(missing_ok=True)
            lock.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
