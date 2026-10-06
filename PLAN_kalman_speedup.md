# Plan: speed up the 1C / TVλ / AFNS Kalman models without changing their behavior

Goal: the three Kalman models (`1C`=DNS, `TVλ`, `AFNS`) **remain three distinct
models** — each keeps its own type, parameters, and math (1C: fixed λ; TVλ: λ as a
4th state with EKF linearization; AFNS: yield-adjustment intercept + continuous-time
transition). Every model produces **the same outputs as today** (loglik, predictions,
estimates — within floating-point tolerance), just faster. No merging of model
behavior, no optimizer changes, no warm starts, no algorithm swaps that alter results.

Ordered checklist for the implementing agent. Each phase ends green on the Phase 0 gate.

---

## Phase 0 — regression gate (do this first, touch nothing else)

1. Write `test_equivalence.jl` (repo root):
   - Load `data/thread_id__7__data.csv` + maturities via `load_data`.
   - For each of `"1C"`, `"TVλ"`, `"AFNS"`: build via `create_model`, `set_params!`
     with column 1 of the matching `initializations/<name>/init_params_*.csv`
     (AFNS: after `reconcile_init_params`; AFNS also needs `data./100`, `maturities./12`
     as in `run`).
   - Record `get_loss`, `get_loss_array`, and all 5 fields of `predict` to
     `test_equivalence_ref.jls` (Serialization) on the **first** run; on later runs
     compare: `isapprox(...; rtol=1e-6, atol=1e-8)` elementwise, loglik `rtol=1e-8`.
   - Also run each with 2–3 random parameter draws (fixed seed) pushed through
     `transform_params`, so the gate covers more than one point in parameter space.
2. Run once on current code to create the reference. This file is the definition of
   "functionally the same" for every later phase.

## Phase 1 — one filter kernel instead of three (implementation dedup ONLY)

This phase changes **code layout, not model behavior**: the three models keep their
own types and their own observation-prediction math via dispatch; only the copied
update/predict tail is shared. If a minimal-risk diff is preferred, this phase may
be skipped entirely — then apply Phases 2–3 to each of the three `filter!` methods
separately (three times the edit sites, same result).

The three `filter!` bodies are copies. Everything from "Update F" down is
line-for-line identical (TVλ's `Z_dot` *is* `m.base.Z`, see
`src/models/kalman/filter.jl:76`); only the observation-prediction head differs:

- 1C (`filter.jl:165`): `y_pred = Z*beta`, Z fixed.
- TVλ (`filter.jl:52`): refresh loadings from `beta[4]`, `y_pred = Z[:,1:3]*beta[1:3]`,
  write EKF derivative into `Z[:,4]` (lines 78–86).
- AFNS (`afns.jl:231`): `y_pred = Z*beta + A`.

3. Add one hook, `predict_observation!(m)`, with three small methods (DNS default,
   TVλ, AFNS). It must fill `y_pred` and, for TVλ, columns 2–4 of `Z`.
4. Rewrite `filter!(m::AbstractKalmanModel, y, cache)` once:
   `predict_observation!(m)` → NaN branch (shared prediction-only tail) → shared
   measurement tail. Delete the TVλ and AFNS `filter!` methods (~120 lines).
   - Note: TVλ then computes `Z[:,4]` on NaN steps too. Harmless — nothing reads
     it on those steps (`predict` stores only cols 2–3) — but confirm via the gate.
   - Keep the TVλ-specific `initialize_filter` (`filter.jl:29`) as is.
5. Gate: run Phase 0 test. Must pass for all three models.

## Phase 2 — one factorization per step (the main flops win)

Today every accepted step factorizes the N×N `F` **twice** and allocates:
`inv(F)` in `filter!` (`filter.jl:190`) and `logdet(F)` in `get_loss`
(`filter.jl:235`), plus `(v' * F_inv * v)[1]` allocating.

6. First grep: confirm `F_inv` / `logdet` are used nowhere outside
   `src/models/kalman/filter.jl` (db/save/forecast code must not read them).
7. Add to `KalmanBaseModel` (constructor + both inner constructors):
   `F_work::Matrix{Fβ}` (N×N scratch) and `logdetF::Vector{Fβ}` (length 1).
8. In the unified `filter!`:
   - `copyto!(F_work, F); C = cholesky!(Symmetric(F_work), check=false)`.
   - `issuccess(C) || (F_inv .= Inf; logdetF[1] = NaN; return nothing)` — this
     replaces the try/catch around `inv`. (Indefinite F used to surface as a
     DomainError in `get_loss`'s logdet and yield `-Inf`; now it yields NaN →
     `-Inf` via the existing isnan check. Same observable outcome; one println is
     lost — acceptable.)
   - `logdetF[1] = logdet(C)`; `F_inv .= inv(C)` (Cholesky-based inverse, ~2x
     cheaper than LU; K-gain code unchanged).
9. In `get_loss` (`filter.jl:222`): replace `logdet(base.F)` with
   `base.logdetF[1]` and `(v'*F_inv*v)[1]` with `dot(base.v, base.F_inv, base.v)`.
   Keep the try/catch and the `t > 1` semantics exactly.
10. Gate: Phase 0 test. Cholesky vs LU inverse shifts last bits — the 1e-6/1e-8
    tolerances absorb this; if a model fails the gate, investigate, don't loosen.

## Phase 3 — allocation scrub in the hot loop

11. `any(isnan.(y))` → `any(isnan, y)` (one site after Phase 1).
12. `get_loss` (`filter.jl:231`): `@views filter!(model, data[:, t], cache)` —
    `get_loss_array` and `predict` already do this; `get_loss` doesn't.
13. TVλ head: `m.base.beta[4:4]` → `view(m.base.beta, 4:4)`;
    `dlambda_da4 = (m.lambda .- 1e-2)` → scalar `dλ = m.lambda[1] - 1e-2`.
14. NaN branch: `y_pred .= Z * beta` → `mul!(y_pred, Z, beta)` (comes free with
    the shared head from Phase 1).
15. Delete `GC.gc()` at `src/optimization.jl:339` and `src/forecasting.jl:200`.
    Full collections per optimizer group / per forecast task; zero effect on
    outputs, pure wall-clock waste with 16 processes per node.
16. Gate: Phase 0 test, plus `@allocated` on one `filter!` call ≈ 0 (the
    Cholesky wrapper may allocate a few hundred bytes; anything O(N²) per step
    is a regression).

## Phase 4 — measure

17. Small benchmark script (extend `bench_score.jl` or add `bench_kalman.jl`):
    median time of `get_loss(model, data[:, 1:231])` for each of the three
    models, before (git stash / previous commit) vs after. Report a 3×2 table.
    Target: ≥3x on `get_loss`; if less, profile before doing anything else.

## Phase 5 — optional, only if Phase 4 says it's still too slow

18. Woodbury: `Omega_obs` is diagonal for all three (σ²I for 1C/TVλ,
    per-maturity diag for AFNS), so `F = D + Z P Z'` with rank M=3 (4 for TVλ).
    `F⁻¹` and `logdet F` via the M×M capacitance matrix: O(N·M²) instead of
    O(N³). Mathematically identical likelihood, but larger float drift — keep the
    same gate tolerances; if they fail, compare against an extended-precision
    (BigFloat, T=20) run to decide which side is more accurate.

## Explicitly out of scope (they change results)

- Finite-difference → ForwardDiff gradients in `estimate_steps!` (biggest
  remaining lever; requires making `KalmanBaseModel` buffers Dual-friendly).
- Warm-starting rolling windows from the previous task's params.
- Optimizer/iteration/tolerance changes, central→forward differences.

Mention these in the final report but do not implement.
