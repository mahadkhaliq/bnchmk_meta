"""Launch the 1x1 Lorentz mapping-constant ablation across CUDA devices."""

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
    "baseline": {
        "wp_scale": 0.5,
        "wp_floor": 1e-5,
        "gamma_scale": 0.1,
        "gamma_floor": 1e-4,
        "epsilon_inf_offset": 1.0,
        "mu_inf_offset": 1.0,
    },
    "no_wp_scale": {
        "wp_scale": 1.0,
        "wp_floor": 1e-5,
        "gamma_scale": 0.1,
        "gamma_floor": 1e-4,
        "epsilon_inf_offset": 1.0,
        "mu_inf_offset": 1.0,
    },
    "no_wp_floor": {
        "wp_scale": 0.5,
        "wp_floor": 0.0,
        "gamma_scale": 0.1,
        "gamma_floor": 1e-4,
        "epsilon_inf_offset": 1.0,
        "mu_inf_offset": 1.0,
    },
    "wp_softplus_only": {
        "wp_scale": 1.0,
        "wp_floor": 0.0,
        "gamma_scale": 0.1,
        "gamma_floor": 1e-4,
        "epsilon_inf_offset": 1.0,
        "mu_inf_offset": 1.0,
    },
    "no_gamma_scale": {
        "wp_scale": 0.5,
        "wp_floor": 1e-5,
        "gamma_scale": 1.0,
        "gamma_floor": 1e-4,
        "epsilon_inf_offset": 1.0,
        "mu_inf_offset": 1.0,
    },
    "no_gamma_floor": {
        "wp_scale": 0.5,
        "wp_floor": 1e-5,
        "gamma_scale": 0.1,
        "gamma_floor": 0.0,
        "epsilon_inf_offset": 1.0,
        "mu_inf_offset": 1.0,
    },
    "gamma_softplus_only": {
        "wp_scale": 0.5,
        "wp_floor": 1e-5,
        "gamma_scale": 1.0,
        "gamma_floor": 0.0,
        "epsilon_inf_offset": 1.0,
        "mu_inf_offset": 1.0,
    },
    "no_epsilon_offset": {
        "wp_scale": 0.5,
        "wp_floor": 1e-5,
        "gamma_scale": 0.1,
        "gamma_floor": 1e-4,
        "epsilon_inf_offset": 0.0,
        "mu_inf_offset": 1.0,
    },
    "no_mu_offset": {
        "wp_scale": 0.5,
        "wp_floor": 1e-5,
        "gamma_scale": 0.1,
        "gamma_floor": 1e-4,
        "epsilon_inf_offset": 1.0,
        "mu_inf_offset": 0.0,
    },
    "no_background_offsets": {
        "wp_scale": 0.5,
        "wp_floor": 1e-5,
        "gamma_scale": 0.1,
        "gamma_floor": 1e-4,
        "epsilon_inf_offset": 0.0,
        "mu_inf_offset": 0.0,
    },
}
PARAMETER_NAMES = tuple(next(iter(PROFILES.values())))


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
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILES),
        default=tuple(PROFILES),
    )
    parser.add_argument("--gpus", type=str, default="0,1,2,3")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args, profile, seed, run_dir):
    constants = PROFILES[profile]
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
        "all",
        "--wp-scale",
        str(constants["wp_scale"]),
        "--wp-floor",
        str(constants["wp_floor"]),
        "--gamma-scale",
        str(constants["gamma_scale"]),
        "--gamma-floor",
        str(constants["gamma_floor"]),
        "--epsilon-inf-offset",
        str(constants["epsilon_inf_offset"]),
        "--mu-inf-offset",
        str(constants["mu_inf_offset"]),
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
            handle, fieldnames=("profile", *PARAMETER_NAMES, "seeds", "epochs")
        )
        writer.writeheader()
        for profile in args.profiles:
            writer.writerow(
                {
                    "profile": profile,
                    **PROFILES[profile],
                    "seeds": ",".join(str(seed) for seed in args.seeds),
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
    run_dir = task["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONUNBUFFERED"] = "1"
    config = {
        "profile": task["profile"],
        "seed": task["seed"],
        "constants": PROFILES[task["profile"]],
        "all_mappings_enabled": True,
        "physical_gpu": gpu,
        "command": task["command"],
        "started_at": utc_now(),
        "host": socket.gethostname(),
    }
    write_json(run_dir / "config.json", config)
    log_handle = (run_dir / "train.log").open("w")
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
    return status


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
        "experiment": "1x1 Lorentz mapping-constant ablation",
        "created_at": utc_now(),
        "host": socket.gethostname(),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "output_root": str(args.output_root),
        "python": str(args.python),
        "profiles": {profile: PROFILES[profile] for profile in args.profiles},
        "seeds": args.seeds,
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
                ROOT / "lorentz" / "run_mapping_constant_ablation.py",
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
    available_gpus = list(gpus)
    active = []
    statuses = []
    while pending or active:
        while pending and available_gpus:
            active.append(launch(pending.pop(0), available_gpus.pop(0)))
        time.sleep(2.0)
        still_active = []
        for run in active:
            if run["process"].poll() is None:
                still_active.append(run)
                continue
            statuses.append(finish(run))
            available_gpus.append(run["gpu"])
        active = still_active

    completed = sum(status["state"] == "completed" for status in statuses)
    failed = len(statuses) - completed
    campaign_status = {
        **manifest,
        "finished_at": utc_now(),
        "state": "completed" if failed == 0 else "completed_with_failures",
        "completed_runs": completed,
        "failed_runs": failed,
    }
    write_json(args.output_root / "campaign_status.json", campaign_status)
    print(json.dumps(campaign_status, indent=2), flush=True)


if __name__ == "__main__":
    main()
