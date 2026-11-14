"""Parallel independent training controls; no validation references are opened."""
from pathlib import Path
import concurrent.futures
import argparse
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--runs", type=int, default=160)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
root = a.root
sys.path.insert(0, str(root / "scripts"))
from train import CANDIDATES

out = root / ("fits%d" % a.runs)
out.mkdir(exist_ok=True)
logs = out / "logs"
logs.mkdir(exist_ok=True)
tasks = [dict(charge=charge, subset=subset, candidate=None)
         for subset in ("full", "even", "odd") for charge in ("wp", "wm")]
tasks += [dict(charge=charge, subset="full", candidate=name)
          for name in CANDIDATES for charge in ("wp", "wm")]
manifest = out / "TASKS.json"
if manifest.exists():
    assert json.loads(manifest.read_text()) == tasks
else:
    manifest.write_text(json.dumps(tasks, indent=2) + "\n")


def run(task):
    name = task["charge"] + "_" + task["subset"] + "_" + (task["candidate"] or "sequence")
    complete = logs / (name + ".complete.json")
    if complete.exists():
        old = json.loads(complete.read_text())
        assert old["returncode"] == 0
        return old
    claim = logs / (name + ".claimed")
    claim.mkdir()
    cmd = [sys.executable, str(root / "scripts/train.py"), "--training",
           str(root / ("pooled%d" % a.runs) / ("T_" + task["charge"].upper() + ".npz")),
           "--prior", str(root / "priors" / ("w_prior_v6_nomec_" + task["charge"] + ".npz")),
           "--charge", task["charge"], "--output", str(out), "--subset", task["subset"]]
    if task["candidate"]:
        cmd += ["--candidate", task["candidate"]]
    env = dict(os.environ, WPRED_ROOT=str(root), OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    started = time.monotonic()
    with (logs / (name + ".log")).open("w") as stream:
        process = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(task, returncode=process.returncode, elapsed_s=time.monotonic() - started,
                  command=cmd, job=os.environ.get("SLURM_JOB_ID"))
    complete.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return record


with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
    results = list(pool.map(run, tasks))
assert all(r["returncode"] == 0 for r in results), "Inspect failed training controls; retain every outcome"
print("ALL_TRAINING_TASKS_COMPLETE", flush=True)
