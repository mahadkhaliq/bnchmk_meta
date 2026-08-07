"""Launch the matched 1x1 w0 mapping ablation across CUDA devices."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .run_constraint_ablation import read_json, sha256_file, utc_now, write_json
from .train_1x1 import DEFAULT_DATASET


ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "reference": {
        "w0_mapping": "bounded",
        "w0_margin": 0.15,
        "constraints": "all",
        "description": "Current bounded sigmoid with margin outside the data band.",
    },
    "data_band": {
        "w0_mapping": "bounded",
        "w0_margin": 0.0,
        "constraints": "all",
        "description": "Bound resonances to the measured 12-26 GHz data band.",
    },
    "wide_band": {
        "w0_mapping": "bounded",
        "w0_margin": 0.30,
        "constraints": "all",
        "description": "Double the Reference margin around the data band.",
    },
    "lower_bounded": {
        "w0_mapping": "lower_bounded",
        "w0_margin": 0.15,
        "constraints": "all",
        "description": "Keep the positive lower bound but remove the upper bound.",
    },
    "raw": {
        "w0_mapping": "bounded",
        "w0_margin": 0.15,
        "constraints": "wp,gamma,epsilon_inf,mu_inf",
        "description": "Feed raw w0 to the decoder; all other mappings stay enabled.",
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", choices=("silu", "relu"), default="silu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--thickness-mm", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--profiles", nargs="+", choices=tuple(PROFILES), default=tuple(PROFILES)
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args, profile_name, seed, run_dir):
    profile = PROFILES[profile_name]
    return [
        str(args.python),
        "-u",
        "-m",
        "lorentz.train_1x1",
        "--dataset",
        str(args.dataset.resolve()),
        "--checkpoint",
        str((run_dir / "model.pt").resolve()),
        "--history",
        str((run_dir / "history.csv").resolve()),
        "--metrics",
        str((run_dir / "metrics.json").resolve()),
        "--constraints",
        profile["constraints"],
        "--w0-mapping",
        profile["w0_mapping"],
        "--w0-margin",
        str(profile["w0_margin"]),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--hidden",
        str(args.hidden),
        "--depth",
        str(args.depth),
        "--activation",
        args.activation,
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--thickness-mm",
        str(args.thickness_mm),
        "--seed",
        str(seed),
        "--split-seed",
        str(args.split_seed),
        "--max-samples",
        str(args.max_samples),
        "--device",
        "cuda",
    ]


def write_matrix(args):
    path = args.output_root / "experiment_matrix.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "profile",
                "w0_mapping",
                "w0_margin",
                "constraints",
                "description",
                "model_seeds",
                "split_seed",
                "epochs",
            ),
        )
        writer.writeheader()
        for name in args.profiles:
            writer.writerow(
                {
                    "profile": name,
                    **PROFILES[name],
                    "model_seeds": ",".join(str(seed) for seed in args.seeds),
                    "split_seed": args.split_seed,
                    "epochs": args.epochs,
                }
            )
    return path


def build_tasks(args):
    tasks = []
    for seed in args.seeds:
        for profile in args.profiles:
            run_dir = args.output_root / "runs" / profile / f"seed_{seed}"
            status_path = run_dir / "status.json"
            status = read_json(status_path)
            if not args.rerun and status and status.get("state") in {
                "completed",
                "failed",
            }:
                print(
                    f"skip terminal {profile} seed={seed} "
                    f"state={status.get('state')}",
                    flush=True,
                )
                continue
            tasks.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "run_dir": run_dir,
                    "status_path": status_path,
                    "command": build_command(args, profile, seed, run_dir),
                }
            )
    return tasks


def launch(task, gpu):
    task["run_dir"].mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONUNBUFFERED"] = "1"
    config = {
        "profile": task["profile"],
        "seed": task["seed"],
        "settings": PROFILES[task["profile"]],
        "physical_gpu": gpu,
        "command": task["command"],
        "started_at": utc_now(),
        "host": socket.gethostname(),
    }
    write_json(task["run_dir"] / "config.json", config)
    log_handle = (task["run_dir"] / "train.log").open("w")
    log_handle.write(json.dumps(config, indent=2) + "\n\n")
    log_handle.flush()
    process = subprocess.Popen(
        task["command"],
        cwd=ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    status = {**config, "state": "running", "pid": process.pid}
    write_json(task["status_path"], status)
    print(
        f"started {task['profile']} seed={task['seed']} "
        f"gpu={gpu} pid={process.pid}",
        flush=True,
    )
    return {
        **task,
        "gpu": gpu,
        "process": process,
        "log_handle": log_handle,
        "started_monotonic": time.monotonic(),
        "status": status,
    }


def finish(active):
    return_code = active["process"].returncode
    active["log_handle"].close()
    metrics = read_json(active["run_dir"] / "metrics.json") or {}
    state = (
        "completed"
        if return_code == 0 and metrics.get("status") == "completed"
        else "failed"
    )
    status = {
        **active["status"],
        "state": state,
        "return_code": return_code,
        "finished_at": utc_now(),
        "seconds": time.monotonic() - active["started_monotonic"],
        "metrics_status": metrics.get("status"),
        "failure": metrics.get("failure"),
    }
    write_json(active["status_path"], status)
    print(
        f"{state} {active['profile']} seed={active['seed']} "
        f"gpu={active['gpu']} return_code={return_code}",
        flush=True,
    )


def main():
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.dataset = args.dataset.resolve()
    args.python = args.python.resolve()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one CUDA device.")
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    if not args.python.exists():
        raise FileNotFoundError(args.python)

    matrix_path = write_matrix(args)
    manifest = {
        "experiment": "1x1 Lorentz w0 mapping ablation",
        "created_at": utc_now(),
        "host": socket.gethostname(),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "output_root": str(args.output_root),
        "python": str(args.python),
        "profiles": {name: PROFILES[name] for name in args.profiles},
        "model_seeds": args.seeds,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "depth": args.depth,
        "activation": args.activation,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "thickness_mm": args.thickness_mm,
        "max_samples": args.max_samples,
        "gpus": gpus,
        "matrix": str(matrix_path),
        "source_sha256": {
            path.name: sha256_file(path)
            for path in (
                ROOT / "lorentz" / "f1.py",
                ROOT / "lorentz" / "lorentz.py",
                ROOT / "lorentz" / "losses.py",
                ROOT / "lorentz" / "train_1x1.py",
                ROOT / "lorentz" / "run_w0_mapping_ablation.py",
            )
        },
    }
    write_json(args.output_root / "manifest.json", manifest)
    tasks = build_tasks(args)
    print(
        f"campaign {args.output_root} | tasks={len(tasks)} | gpus={gpus}",
        flush=True,
    )
    if args.dry_run:
        for task in tasks:
            print(" ".join(task["command"]))
        return

    pending = list(tasks)
    active = []
    available = list(gpus)
    while pending or active:
        while pending and available:
            active.append(launch(pending.pop(0), available.pop(0)))
        time.sleep(1.0)
        for run in list(active):
            if run["process"].poll() is None:
                continue
            finish(run)
            available.append(run["gpu"])
            active.remove(run)

    print("campaign complete", flush=True)


if __name__ == "__main__":
    main()
