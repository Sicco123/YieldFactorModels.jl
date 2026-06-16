#!/usr/bin/env python3
"""One-step-ahead cumulative forecast SSE over selected maturities: thread_id__6 vs submitted."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/skooiker/TVNets/YieldFactorModels.jl/results"
DATA = "/home/skooiker/TVNets/YieldFactorModels.jl/data/thread_id__6__data.csv"
MAT = "/home/skooiker/TVNets/YieldFactorModels.jl/data/thread_id__6__maturities.csv"
FC_A = f"{BASE}/thread_id__6/2SSD-NNS/2SSD-NNS__thread_id__6__moving_window_forecasts.csv"
FC_B = f"{BASE}/thread_id__6_submitted/2SSD-NNS/2SSD-NNS__thread_id__6__moving_window_forecasts.csv"
OUTDIR = f"{BASE}/param_comparison_plots"
os.makedirs(OUTDIR, exist_ok=True)

WANT = [3, 6, 12, 24, 60, 120]

maturities = pd.read_csv(MAT, header=None).iloc[:, 0].tolist()
sel = [maturities.index(m) for m in WANT]  # 0-based column indices
print("maturities:", maturities)
print("selected maturities:", WANT, "-> column indices", sel)

data = pd.read_csv(DATA, header=None).values  # (24 maturities, 480 times)
n_mat, n_time = data.shape


def one_step_sse(fc_path):
    fc = pd.read_csv(fc_path, header=None).values
    origin, target = fc[:, 0], fc[:, 1]
    keep = np.isclose(target, origin + 1)
    fc = fc[keep]
    target = target[keep].astype(int)
    fc_yields = fc[:, 2:]  # (k, 24)
    rows = []
    for tgt, pred in zip(target, fc_yields):
        if tgt < 1 or tgt > n_time:
            continue
        actual = data[:, tgt - 1]
        sse = np.sum((pred[sel] - actual[sel]) ** 2)
        rows.append((tgt, sse))
    df = pd.DataFrame(rows, columns=["time", "sse"]).sort_values("time").set_index("time")
    df["cum_sse"] = df["sse"].cumsum()
    return df


a = one_step_sse(FC_A)
b = one_step_sse(FC_B)
print(f"thread_id__6 : total SSE (subset) = {a['sse'].sum():.4f}")
print(f"submitted    : total SSE (subset) = {b['sse'].sum():.4f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ttl = "maturities " + ", ".join(map(str, WANT))
ax1.plot(a.index, a["cum_sse"], lw=1.6, color="C0", label=f"thread_id__6 (total={a['sse'].sum():.3f})")
ax1.plot(b.index, b["cum_sse"], lw=1.6, color="C1", label=f"submitted (total={b['sse'].sum():.3f})")
ax1.set_ylabel("cumulative SSE")
ax1.set_title(f"One-step-ahead cumulative forecast SSE (summed over {ttl})")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(a.index, a["sse"], lw=1.0, color="C0", label="thread_id__6")
ax2.plot(b.index, b["sse"], lw=1.0, color="C1", label="submitted")
ax2.set_ylabel("per-period SSE"); ax2.set_xlabel("forecast target time")
ax2.set_title("Per-period one-step-ahead squared error")
ax2.legend(); ax2.grid(alpha=0.3)

fig.tight_layout()
out = f"{OUTDIR}/cumulative_forecast_loss_subset.png"
fig.savefig(out, dpi=130)
plt.close(fig)
print(f"Wrote {out}")
