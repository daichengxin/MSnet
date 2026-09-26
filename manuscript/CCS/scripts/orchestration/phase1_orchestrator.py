#!/usr/bin/env python3
"""Resource-aware, fail-closed orchestrator for the 36 frozen Phase-1 runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENT_ROOT = Path(os.environ.get("CCS_SCALING_ROOT", Path.cwd() / "ccs_scaling_runs")).resolve()
RELEASE_ROOT = Path(__file__).resolve().parents[2]
OUT = EXPERIMENT_ROOT / "phase1_dataset_scaling"
SCRIPT = RELEASE_ROOT / "scripts" / "training" / "phase1_dataset_scaling.py"
PYTHON = os.environ.get("CCS_PYTHON", sys.executable)
ALPHAPEPTDEEP_ROOT = os.environ.get("ALPHAPEPTDEEP_ROOT", "")
N_VALUES = [500, 1000, 2500, 5600, 10000, 25000, 50000, 100000, 200000, 280000, 400000, 465356]
SEEDS = [42, 123, 2026]
MIN_FREE_MIB = 12000
MAX_START_UTIL = 15


def gpu_state():
    text = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    state = []
    for line in text.strip().splitlines():
        idx, free, util = (int(v.strip()) for v in line.split(","))
        state.append((idx, free, util))
    return state


def is_complete(n, seed):
    p = OUT / "runs" / f"N{n}_seed{seed}.json"
    if not p.is_file():
        return False
    try:
        obj = json.loads(p.read_text())
        return obj.get("status") == "complete" and obj.get("test_evaluations") == 1
    except Exception:
        return False


def main():
    lock = OUT / "orchestrator.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise SystemExit(f"Refusing duplicate orchestrator: {lock}")
    (lock / "pid").write_text(str(os.getpid()))
    queue = [(n, seed) for n in N_VALUES for seed in SEEDS if not is_complete(n, seed)]
    running = {}
    try:
        while queue or running:
            for gpu, item in list(running.items()):
                process, log_handle, n, seed = item
                code = process.poll()
                if code is None:
                    continue
                log_handle.close()
                del running[gpu]
                if code != 0 or not is_complete(n, seed):
                    (OUT / "ORCHESTRATOR_FAILED.json").write_text(json.dumps({
                        "N": n, "seed": seed, "gpu": gpu, "exit_code": code,
                        "log": str(OUT / "logs" / f"N{n}_seed{seed}.log"),
                    }, indent=2))
                    raise RuntimeError(f"Run failed closed: N={n}, seed={seed}, GPU={gpu}, exit={code}")
            states = gpu_state()
            for gpu, free, util in states:
                if not queue or gpu in running:
                    continue
                if free < MIN_FREE_MIB or util > MAX_START_UTIL:
                    continue
                n, seed = queue.pop(0)
                log_path = OUT / "logs" / f"N{n}_seed{seed}.log"
                log_handle = log_path.open("a", buffering=1)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                if ALPHAPEPTDEEP_ROOT:
                    env["PYTHONPATH"] = ALPHAPEPTDEEP_ROOT
                env["PYTHONHASHSEED"] = str(seed)
                process = subprocess.Popen(
                    [PYTHON, str(SCRIPT), "run", "--n", str(n), "--seed", str(seed)],
                    stdout=log_handle, stderr=subprocess.STDOUT, env=env,
                )
                running[gpu] = (process, log_handle, n, seed)
                print(f"START gpu={gpu} N={n} seed={seed} pid={process.pid} free={free} util={util}", flush=True)
            status = {
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "remaining": len(queue),
                "running": {str(g): {"pid": p.pid, "N": n, "seed": s} for g, (p, _, n, s) in running.items()},
                "completed": sum(is_complete(n, s) for n in N_VALUES for s in SEEDS),
                "gpu_state": states,
            }
            (OUT / "orchestrator_status.json").write_text(json.dumps(status, indent=2))
            time.sleep(20)
        aggregate_log = (OUT / "logs" / "aggregate.log").open("w")
        code = subprocess.call(
            [PYTHON, str(SCRIPT), "aggregate"], stdout=aggregate_log, stderr=subprocess.STDOUT,
            env={**os.environ, **({"PYTHONPATH": ALPHAPEPTDEEP_ROOT} if ALPHAPEPTDEEP_ROOT else {})},
        )
        aggregate_log.close()
        if code != 0:
            raise RuntimeError(f"Aggregation failed: exit={code}")
        (OUT / "PHASE1_COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    finally:
        try:
            (lock / "pid").unlink(missing_ok=True)
            lock.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
