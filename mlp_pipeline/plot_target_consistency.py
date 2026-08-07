"""Visualise the v3 target consistency issue (stored T vs |S21|^2) as SEPARATE
standalone figures, with legends placed OUTSIDE the data.

For 1x1 T and |S21|^2 are identical; for 2x2 / 3x3 they agree almost everywhere
but a sparse set of points (only at sharp resonance edges) deviate up to ~0.52 /
~0.25.

    python plot_target_consistency.py   ->  plots/target_*.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)
INK = "#22303f"
T_CLR = "#22303f"       # stored target T
S_CLR = "#e76f51"       # |S21|^2 recomputed
HL = "#c1121f"          # deviating points
THR = 1e-2


def load(tag):
    d = np.load(f"../power_tx_data/version_3/preprocessed_{tag}.npz", allow_pickle=True)
    T = d["T"].astype(np.float64)
    s21 = (np.abs(d["S21"]) ** 2).astype(np.float64)
    return d["freq_GHz"], T, s21


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e5ebf0", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10, colors=INK)


def save(fig, name):
    out = f"plots/{name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out)


def spectrum_figure(tag):
    freq, T, s21 = load(tag)
    ad = np.abs(T - s21)
    smp = int(ad.max(axis=1).argmax())            # sample with the single worst point
    dev = ad[smp] > THR
    j = int(ad[smp].argmax())

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(freq, T[smp], color=T_CLR, lw=2.0, label="stored T (target)", zorder=3)
    ax.plot(freq, s21[smp], color=S_CLR, lw=1.4, ls="--", label="|S21|² (recomputed)", zorder=4)
    # hollow markers so they never hide the curves
    ax.scatter(freq[dev], T[smp][dev], s=70, facecolor="none", edgecolor=HL, lw=1.8,
               zorder=6, label=f"deviating points (>{THR:g})")
    # annotation parked in empty lower-left, arrow to the worst point
    ax.annotate(f"max Δ = {ad[smp][j]:.3f}\n{int(dev.sum())} points, at resonance edges",
                xy=(freq[j], (T[smp][j] + s21[smp][j]) / 2), xycoords="data",
                xytext=(0.03, 0.30), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=HL, lw=1.6),
                fontsize=10, color=HL, weight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=HL, lw=1.2))
    ax.set_title(f"v3 {tag}: stored T vs |S21|²  (sample {smp}) — identical except sparse points",
                 fontsize=12, weight="bold", color=INK)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("transmission")
    ax.set_ylim(-0.03, 1.08)
    style(ax)
    # legend OUTSIDE, below the axes
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
              frameon=False, fontsize=9.5)
    save(fig, f"target_{tag}_spectrum")


def hist_figure():
    fig, ax = plt.subplots(figsize=(9, 5))
    for tag, c in [("2x2", "#457b9d"), ("3x3", "#2a9d8f")]:
        _, T, s21 = load(tag)
        ad = np.abs(T - s21).ravel()
        pct = (ad > THR).mean() * 100
        ax.hist(np.clip(ad, 1e-9, None), bins=np.logspace(-9, 0, 60),
                histtype="step", lw=2, color=c, label=f"{tag}  ({pct:.2f}% exceed {THR:g})")
    ax.axvline(THR, color=HL, ls=":", lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("How rare the deviations are  (every element, both scales)",
                 fontsize=12, weight="bold", color=INK)
    ax.set_xlabel("|T − |S21|²|   (log)")
    ax.set_ylabel("count (log)")
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
              frameon=False, fontsize=9.5)
    save(fig, "target_diff_histogram")


def frequency_figure():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for tag, c in [("2x2", "#457b9d"), ("3x3", "#2a9d8f")]:
        freq, T, s21 = load(tag)
        frac = (np.abs(T - s21) > THR).mean(axis=0) * 100
        ax.plot(freq, frac, color=c, lw=1.6, label=tag)
    ax.set_title("Where the deviations occur in frequency  (concentrated at the resonances)",
                 fontsize=12, weight="bold", color=INK)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel(f"% of samples deviating (>{THR:g})")
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, fontsize=9.5)
    save(fig, "target_diff_by_frequency")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    spectrum_figure("2x2")
    spectrum_figure("3x3")
    hist_figure()
    frequency_figure()


if __name__ == "__main__":
    main()
