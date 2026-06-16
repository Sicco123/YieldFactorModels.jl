#!/usr/bin/env python3
"""Compare moving-window fitted params between thread_id__6 and thread_id__6_submitted."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/skooiker/TVNets/YieldFactorModels.jl/results"
F_A = f"{BASE}/thread_id__6/2SSD-NNS/2SSD-NNS__thread_id__6__moving_window_fitted_params.csv"
F_B = f"{BASE}/thread_id__6_submitted/2SSD-NNS/2SSD-NNS__thread_id__6__moving_window_fitted_params.csv"
OUTDIR = f"{BASE}/param_comparison_plots"
os.makedirs(OUTDIR, exist_ok=True)

a = pd.read_csv(F_A, header=None)
b = pd.read_csv(F_B, header=None)

# Column 0 is the moving-window time index; rest are parameters.
xa, xb = a.iloc[:, 0], b.iloc[:, 0]
param_cols = list(a.columns[1:])
labels = [f"param_{i}" for i in param_cols]  # param_1 .. param_42

# --- Combined grid overview ---
n = len(param_cols)
ncols = 6
nrows = int(np.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), squeeze=False)
for idx, col in enumerate(param_cols):
    ax = axes[idx // ncols][idx % ncols]
    ax.plot(xa, a.iloc[:, col], lw=1.0, label="thread_id__6", color="C0")
    ax.plot(xb, b.iloc[:, col], lw=1.0, label="submitted", color="C1")
    ax.set_title(labels[idx], fontsize=9)
    ax.tick_params(labelsize=7)
# hide unused axes
for idx in range(n, nrows * ncols):
    axes[idx // ncols][idx % ncols].axis("off")
handles, leg = axes[0][0].get_legend_handles_labels()
fig.legend(handles, leg, loc="upper center", ncol=2, fontsize=10)
fig.suptitle("Moving-window fitted params: thread_id__6 vs submitted", y=0.995, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
overview = f"{OUTDIR}/_overview_all_params.png"
fig.savefig(overview, dpi=130)
plt.close(fig)

# --- One plot per parameter ---
for idx, col in enumerate(param_cols):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xa, a.iloc[:, col], lw=1.3, label="thread_id__6", color="C0")
    ax.plot(xb, b.iloc[:, col], lw=1.3, label="submitted", color="C1")
    ax.set_title(f"{labels[idx]} (column {col})")
    ax.set_xlabel("moving-window index")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/{labels[idx]}.png", dpi=120)
    plt.close(fig)

print(f"Wrote {n} per-param plots + overview to {OUTDIR}")
print(f"Overview: {overview}")
