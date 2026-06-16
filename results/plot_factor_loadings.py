#!/usr/bin/env python3
"""Factor loadings over time per maturity (column): thread_id__6 vs submitted."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/skooiker/TVNets/YieldFactorModels.jl/results"
MAT = "/home/skooiker/TVNets/YieldFactorModels.jl/data/thread_id__6__maturities.csv"
OUTDIR = f"{BASE}/param_comparison_plots"
os.makedirs(OUTDIR, exist_ok=True)

maturities = pd.read_csv(MAT, header=None).iloc[:, 0].tolist()


def path(run, factor):
    return (f"{BASE}/{run}/2SSD-NNS/"
            f"2SSD-NNS__thread_id__6__factor_loadings_{factor}_filtered_outofsample.csv")


for factor in (1, 2):
    a = pd.read_csv(path("thread_id__6", factor), header=None)
    b = pd.read_csv(path("thread_id__6_submitted", factor), header=None)
    ta, tb = np.arange(1, len(a) + 1), np.arange(1, len(b) + 1)
    ncol = a.shape[1]

    nc = 6
    nr = int(np.ceil(ncol / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 2.6 * nr), squeeze=False)
    for j in range(ncol):
        ax = axes[j // nc][j % nc]
        ax.plot(ta, a.iloc[:, j], lw=1.0, color="C0", label="thread_id__6")
        ax.plot(tb, b.iloc[:, j], lw=1.0, color="C1", label="submitted")
        mlabel = maturities[j] if j < len(maturities) else j + 1
        ax.set_title(f"maturity {mlabel}", fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(ncol, nr * nc):
        axes[j // nc][j % nc].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=10)
    fig.suptitle(f"Factor loading {factor} over time, per maturity "
                 f"(filtered out-of-sample): thread_id__6 vs submitted", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{OUTDIR}/factor_loadings_{factor}_over_time.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Wrote {out}  ({len(a)} time points x {ncol} maturities)")
