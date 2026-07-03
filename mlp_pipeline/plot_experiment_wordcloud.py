"""Create a word cloud for the power-transmission metasurface experiments."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud


OUT = "plots/experiment_wordcloud.png"
os.makedirs("plots", exist_ok=True)

FREQUENCIES = {
    "Power Transmission": 120,
    "Metasurface": 112,
    "T = |S21|^2": 108,
    "Frequency Spectrum": 96,
    "2x2 Integrated": 94,
    "21,625 samples": 92,
    "SiLU": 90,
    "beta2 loss": 88,
    "K=3 Neighborhood": 84,
    "Full ReLU": 80,
    "MSE": 78,
    "Sigmoid Output": 76,
    "CST Simulation": 72,
    "Low-T Dips": 72,
    "Weighted MSE": 70,
    "Plain Test MSE": 68,
    "Geometry": 66,
    "d g l w": 64,
    "T_clean": 64,
    "S21": 62,
    "Scale-Invariant": 62,
    "Shared MLP": 60,
    "512 x 4": 60,
    "2000 x 10": 58,
    "1.82M params": 58,
    "40.1M params": 56,
    "channels = 4": 56,
    "Periodic Padding": 54,
    "Wrap-Around": 52,
    "KxK Window": 52,
    "Average Cells": 50,
    "Supercell": 50,
    "1-30 GHz": 50,
    "500 epochs": 50,
    "706 samples": 48,
    "2x2": 48,
    "1x1": 44,
    "3x3": 44,
    "3,006 samples": 42,
    "484 samples": 40,
    "2001 outputs": 42,
    "2003 outputs": 36,
    "BatchNorm": 40,
    "No BatchNorm": 40,
    "ReLU": 38,
    "MLPSiLU": 38,
    "ScaleInvariantSiLU": 36,
    "ScaleInvariantMetasurface": 36,
    "Adam": 34,
    "CosineAnnealingLR": 34,
    "Weight Decay": 30,
    "Train Val Test": 34,
    "14704 train": 30,
    "3677 val": 28,
    "3244 test": 28,
    "480 train": 26,
    "120 val": 24,
    "106 test": 24,
    "Held-Out Test": 32,
    "Absolute Error": 32,
    "Test Spectra": 34,
    "freq_GHz": 30,
    "S11": 28,
    "w = 1 + 2(1-T)^2": 36,
    "w in [1,3]": 30,
    "High T": 26,
    "Low T": 30,
    "dataset_1x1.npz": 24,
    "dataset_2x2.npz": 26,
    "dataset_3x3.npz": 22,
    "dataset_2x2_integrated.npz": 28,
    "No Sweep": 26,
    "Hellbender": 24,
    "CUDA": 22,
    "MPS": 20,
}


def main():
    wc = WordCloud(
        width=2400,
        height=1350,
        background_color="#fbfcfd",
        colormap="viridis",
        prefer_horizontal=0.88,
        max_words=95,
        random_state=7,
        collocations=False,
        normalize_plurals=False,
        contour_width=0,
        margin=8,
        font_path=None,
    ).generate_from_frequencies(FREQUENCIES)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "Power-Transmission Metasurface Experiments",
        fontsize=22,
        fontweight="bold",
        color="#233142",
        pad=18,
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT, bbox_inches="tight", facecolor="#fbfcfd")
    print("saved", OUT)


if __name__ == "__main__":
    main()
