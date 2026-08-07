"""Launch matched synthetic unary recovery seeds across available CUDA devices."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, sha256_file, write_json
from .train_synthetic_1x1 import DEFAULT_DATASET


DEFAULT_OUTPUT = (
    ROOT / "lorentz" / "experiments" / "unary_validation_1x1_20260806"
    / "synthetic_recovery"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", choices=("silu", "relu"), default="silu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--parameter-loss-weight", type=float, default=0.0)
    parser.add_argument("--thickness-mm", type=float, default=0.2)
    parser.add_argument("--gpus", type=str, default="0,1,2")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def build_command(args, seed, run_dir):
    return [
        str(args.python),
        "-u",
        "-m",
        "lorentz.validation.train_synthetic_1x1",
        "--dataset",
        str(args.dataset),
        "--output-dir",
        str(run_dir),
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
        "--parameter-loss-weight",
        str(args.parameter_loss_weight),
        "--thickness-mm",
        str(args.thickness_mm),
        "--seed",
        str(seed),
        "--device",
        "cuda",
    ]


def main():
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.output_root = args.output_root.resolve()
    args.python = args.python.resolve()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    if not args.python.exists():
        raise FileNotFoundError(args.python)
    if not gpus:
        raise ValueError("--gpus must contain at least one device.")

    manifest = {
        "experiment": "multi-seed self-consistent synthetic unary recovery",
        "created_at": utc_now(),
        "host": socket.gethostname(),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "output_root": str(args.output_root),
        "python": str(args.python),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden": args.hidden,
        "depth": args.depth,
        "activation": args.activation,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "parameter_loss_weight": args.parameter_loss_weight,
        "thickness_mm": args.thickness_mm,
        "gpus": gpus,
        "source_sha256": {
            path.name: sha256_file(path)
            for path in (
                ROOT / "lorentz" / "f1.py",
                ROOT / "lorentz" / "lorentz.py",
                ROOT / "lorentz" / "losses.py",
                ROOT / "lorentz" / "validation" / "common.py",
                ROOT / "lorentz" / "validation" / "train_synthetic_1x1.py",
                ROOT / "lorentz" / "validation" / "run_synthetic_recovery.py",
            )
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "manifest.json", manifest)

    pending = []
    for seed in args.seeds:
        run_dir = args.output_root / f"seed_{seed}"
        status_path = run_dir / "status.json"
        status = read_json(status_path)
        if not args.rerun and status and status.get("state") in {"completed", "failed"}:
            print(f"skip seed={seed} state={status['state']}", flush=True)
            continue
        pending.append(
            {
                "seed": seed,
                "run_dir": run_dir,
                "status_path": status_path,
                "command": build_command(args, seed, run_dir),
            }
        )
    print(
        f"campaign {args.output_root} | tasks={len(pending)} | gpus={gpus}",
        flush=True,
    )
    if args.dry_run:
        for task in pending:
            print(" ".join(task["command"]))
        return

    available = list(gpus)
    active = []
    statuses = []
    while pending or active:
        while pending and available:
            task = pending.pop(0)
            gpu = available.pop(0)
            task["run_dir"].mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONUNBUFFERED"] = "1"
            config = {
                "seed": task["seed"],
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
            active.append(
                {
                    **task,
                    "gpu": gpu,
                    "process": process,
                    "log_handle": log_handle,
                    "status": status,
                    "started_monotonic": time.monotonic(),
                }
            )
            print(
                f"started seed={task['seed']} gpu={gpu} pid={process.pid}",
                flush=True,
            )

        time.sleep(2.0)
        running = []
        for task in active:
            if task["process"].poll() is None:
                running.append(task)
                continue
            task["log_handle"].close()
            metrics = read_json(task["run_dir"] / "metrics.json") or {}
            state = (
                "completed"
                if task["process"].returncode == 0
                and metrics.get("status") == "completed"
                else "failed"
            )
            status = {
                **task["status"],
                "state": state,
                "return_code": task["process"].returncode,
                "finished_at": utc_now(),
                "seconds": time.monotonic() - task["started_monotonic"],
                "best_epoch": metrics.get("best_epoch"),
                "test_mse": (metrics.get("test") or {}).get("mse"),
            }
            write_json(task["status_path"], status)
            statuses.append(status)
            available.append(task["gpu"])
            print(
                f"{state} seed={task['seed']} gpu={task['gpu']} "
                f"return_code={task['process'].returncode}",
                flush=True,
            )
        active = running

    completed = sum(status["state"] == "completed" for status in statuses)
    campaign_status = {
        **manifest,
        "finished_at": utc_now(),
        "state": "completed" if completed == len(statuses) else "completed_with_failures",
        "completed_runs": completed,
        "failed_runs": len(statuses) - completed,
    }
    write_json(args.output_root / "campaign_status.json", campaign_status)
    print(json.dumps(campaign_status, indent=2))


if __name__ == "__main__":
    main()
