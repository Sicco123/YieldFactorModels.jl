# Plan: diagnose the post-speed-up forecast regression (TVλ / 1C, thread 8, moving)

Symptom: after the Kalman speed-up (PLAN_kalman_speedup.md, landed ~15:25 on
2026-08-04), a full run of `test-thread_7.jl` (thread 8, moving window,
`reestimate=true`) produces forecasts that are much worse (seen in
`output/thread_id__8_moving/per_horizon/rel_rmse/...`).

## Facts already established (don't re-derive)

- Current `src/` is exactly the delivered state — no post-hoc edits. Both code
  states are saved at `/gpfs/home4/skooiker/TVNets/code_snapshots_20260804/`:
  `src_baseline/` = pre-change code, `src_optimized/` = delivered code
  (verified identical to current `src/`). These are the A/B pair for Phase B.
- The equivalence gate (`test_equivalence.jl`) passed at delivery: filtering,
  loss, and predictions are identical at 4 param points/model. The gate does NOT
  cover optimizer trajectories or the rolling-forecast DB pipeline. The regression
  therefore lives in estimation dynamics, the pipeline, or post-processing.
- Bad-run timeline (all after the code change): 1C out_params 15:40, 1C merged DB
  15:45, TVλ in-sample CSVs 15:48, TVλ merged DB + forecasts CSV 15:53, **two
  orphan TVλ shards (tasks 244, 247) written 15:59 — after the merge**. Orphan
  shards after a `delete_shards=true` merge means several processes raced:
  the merge can fire while other workers are mid-task, so the merged DB can be
  missing tasks or hold half a run. This race is pre-existing code, but the
  speed-up changed timing, which can surface it.
- TVλ warm-starts from `results/thread_id__8/1C/1C__thread_id__8__out_params.csv`,
  which was itself re-estimated with the new code at 15:40. If 1C estimation
  regressed, TVλ inherits a bad start on top of its own estimation.
- Julia: no `julia` on PATH; use
  `/home/skooiker/.julia/juliaup/julia-1.12.1+0.x64.linux.gnu/bin/julia`
  with `--project=/home/skooiker/TVNets/TVNets-1.11.6`.

## Hypotheses, ranked

- **H1 — PSD strictness / failure semantics** (the one semantic change flagged at
  delivery). Old code: `inv(F)` (LU) almost never fails; old TVλ on failure kept a
  *stale* `F_inv` and carried on with a finite garbage likelihood. New code: any
  non-PSD `F` → Cholesky fails → whole log-likelihood = -Inf → `compute_loss`
  returns the 1e10 penalty. Near-unit-root Phi + the non-Joseph P update can
  drift P slightly indefinite, so line searches that used to tiptoe through now
  hit a wall mid-search → LBFGS stops early at worse params, every window.
- **H2 — incomplete/raced merged DB.** Merged DB missing tasks (see orphan
  shards) → missing rows in `TVλ__..._moving_window_forecasts.csv` → downstream
  rel_rmse tables computed on partial/misaligned data.
- **H3 — inherited bad warm start.** 1C regressed (via H1) → its out_params are
  bad → TVλ starts every window from a bad point. Discriminator: is 1C's own
  rel_rmse also destroyed, or only TVλ's?
- **H4 — bit-level drift steering finite-difference LBFGS elsewhere.** Loss agrees
  to ~1e-10 at identical params, so this should cause small dispersion, not
  systematic destruction. Only pursue if H1–H3 all come back clean.

## Phase A — read the artifacts (no reruns, ~minutes)

1. Pin down "suck": which table/plot, which models, how much worse, and vs what
   baseline run (ask/locate the pre-change output for thread 8). If only TVλ is
   bad → H3 gains; if 1C too → common estimation cause (H1).
2. Completeness of the merged DB: count DISTINCT task_id in
   `results/thread_id__8/TVλ/db/forecasts_moving_merged.sqlite3` (expect 250:
   tasks 231..480) and rows in the exported forecasts CSV. Missing tasks → H2
   confirmed as at least a contributing cause.
3. Per-task stored loss: SELECT task_id, loss FROM forecasts — how many are NaN /
   -1e10-ish / wildly worse than neighbors? A run of ~-1e10 losses is H1's
   signature (windows where the optimizer never escaped the penalty).
4. Find the run logs (slurm `tvnets_par_*.log`, `run_*.log` wherever the job ran):
   grep for "Static parameters ... not found", "Error optimizing group",
   "estimation summary ... average seconds per task". Seconds-per-task an order
   of magnitude below pre-change logs = optimizer bailing early (H1 signature),
   not the ~2.5x the filter speed-up explains.

## Phase B — the decisive experiment (single window, deterministic, ~minutes)

5. Script (nothing to do with the DB pipeline): thread 8 data, TVλ with the same
   warm-start logic `run` uses, pick 3 windows (first/middle/last task), call
   `run_estimation!` directly per window. Run it twice: once with `src_baseline`
   swapped in, once with current src (swap `src/`, restore after; snapshots above).
   Compare final LL and params per window.
   - Old ≈ new → estimation is fine → it's H2/H3/pipeline; go back to Phase A.
   - New clearly worse → estimation regressed; go to Phase C.
6. While in there, instrument cheaply: a global counter for Cholesky failures per
   `get_loss` call (current code) vs "Error inverting F" prints (baseline). Old ≈ 0
   and new ≫ 0 during optimization = H1 confirmed directly.

## Phase C — isolate the mechanism (only if Phase B shows regression)

The diff has four independent pieces. Toggle ONE at a time on top of current code,
rerun the Phase B script (and the equivalence gate) after each:
7. Failure semantics (H1): on `!issuccess(C)`, fall back to the old LU path
   (`F_inv = inv(F)` semantics + stale-carry-on) instead of NaN-ing out. If
   results recover → culprit found. The *fix* is then a decision, not a reflex:
   old behavior was silently wrong (garbage likelihood), so prefer LU fallback
   only for the in-search evaluations, or a Joseph-form P update to keep P PSD —
   but that decision comes after confirmation.
8. Gain via `ldiv!` vs explicit `P Z' F⁻¹`: compare max|ΔK| along one real filter
   pass at estimated params (micro-test showed 0.0 on a random SPD case; check at
   the model's own near-singular F).
9. Shared-kernel head ordering (TVλ `predict_observation!` / jacobian): diff
   state paths step-by-step for one window, old vs new, at identical params.
10. GC.gc() removal: timing only, cannot change numbers — skip unless everything
    else is exonerated.

## Phase D — verdict and recovery

11. Name the culprit (or culprit mix — H1 + H2 can both be live). Apply the
    chosen fix, rerun the equivalence gate, then ONE full clean TVλ moving pass
    on thread 8 (fresh results dir, single process to dodge the merge race),
    regenerate the rel_rmse table, and compare against the pre-change output
    before declaring recovery.

## Open questions for the user

- Which exact output/table defines "destroyed", and is a pre-change copy of it
  still around for comparison?
- Was `results/thread_id__8/` wiped before the bad run, and how many parallel
  processes ran (`run_parallel.sh` with 16, or a single `julia test-thread_7.jl`)?
