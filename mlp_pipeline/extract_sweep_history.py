"""Recover per-config loss curves from a sweep TEXT log.

The first capacity sweep was run with an older harness that printed every
epoch to logs/sweep_<GRID>.log but did NOT save structured per-run history
CSVs. All the per-epoch numbers are still in that text log, so this script
parses them back out into:

    logs/history/sweep_<GRID>_baseline_<tag>.csv   (epoch, train, val)
    logs/history/sweep_<GRID>_neigh_<tag>.csv
    plots/sweep_<GRID>_losscurves.png              (val loss vs epoch per config)

It's a one-shot backfill; future sweeps write these directly from
sweep_powertx.py.  Run:

    python extract_sweep_history.py
"""
import os
import re
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config_powertx as C

LOG_PATH = f"logs/sweep_{C.GRID}.log"
os.makedirs("logs/history", exist_ok=True)
os.makedirs("plots", exist_ok=True)

EPOCH_RE = re.compile(r"Epoch\s+(\d+) \| train (\S+) \| val (\S+)")
CONFIG_RE = re.compile(r"=+ config (\S+) =+")
# variant markers inside a config block
BASE_RE = re.compile(r"-+ baseline ")
NEIGH_RE = re.compile(r"-+ neighbourhood ")


def parse_log(path):
    """-> {tag: {'baseline': (e,tr,va), 'neigh': (e,tr,va)}} from the text log."""
    if not os.path.exists(path):
        raise SystemExit(f"no log at {path}")
    out = {}
    cur_tag = None
    cur_variant = None
    with open(path) as f:
        for line in f:
            mc = CONFIG_RE.search(line)
            if mc:
                cur_tag = mc.group(1)
                out.setdefault(cur_tag, {})
                cur_variant = None
                continue
            if BASE_RE.search(line):
                cur_variant = "baseline"
                out[cur_tag][cur_variant] = ([], [], [])
                continue
            if NEIGH_RE.search(line):
                cur_variant = "neigh"
                out[cur_tag][cur_variant] = ([], [], [])
                continue
            me = EPOCH_RE.search(line)
            if me and cur_tag and cur_variant:
                e, tr, va = out[cur_tag][cur_variant]
                e.append(int(me.group(1)))
                tr.append(float(me.group(2)))
                va.append(float(me.group(3)))
    return out


def save_history(curve, path):
    e, tr, va = curve
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train", "val"])
        for row in zip(e, tr, va):
            w.writerow(row)


def main():
    data = parse_log(LOG_PATH)
    if not data:
        raise SystemExit(f"parsed nothing from {LOG_PATH}")

    tags = list(data.keys())
    print(f"parsed configs: {tags}")

    # write per-run history CSVs
    for tag, variants in data.items():
        for variant, curve in variants.items():
            if not curve[0]:
                continue
            name = "baseline" if variant == "baseline" else "neigh"
            path = f"logs/history/sweep_{C.GRID}_{name}_{tag}.csv"
            save_history(curve, path)
            print("saved", path, f"({len(curve[0])} epochs)")

    # loss-curve figure: val loss vs epoch, one line per config
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, variant, title in [(axes[0], "baseline", "baseline"),
                               (axes[1], "neigh", "neighbourhood")]:
        for tag in tags:
            curve = data[tag].get(variant)
            if curve and curve[0]:
                e, _, va = curve
                ax.plot(e, va, label=tag)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(f"{title} val loss")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(title="HIDDENxN")
    axes[0].set_ylabel("val MSE (log)")
    fig.suptitle(f"Power-transmission {C.GRID}: validation loss per capacity (from log)")
    plt.tight_layout()
    out = f"plots/sweep_{C.GRID}_losscurves.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
