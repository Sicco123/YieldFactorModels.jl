# Is the fast Kalman filter's log-likelihood right?
#
# `filter!` (src/models/kalman/filter.jl) takes one Cholesky of F per step and
# reads log|F| and F^-1 v off it, out of buffers that are recycled nine times
# inside the step. `get_loss` currently scores -SSE, which touches neither of
# those two quantities — so a bug in them would show up only when the objective
# is switched back to the likelihood. This file checks them.
#
#   ref_step! / get_loss_ref  the same recursion written the obvious way: fresh
#                             allocations, LU (not Cholesky) for both log|F| and
#                             F\v, explicit K = P Z' inv(F). It shares only the
#                             observation head with the fast path, so anything
#                             that disagrees is in the update.
#   lockstep                  both steps run from the SAME (beta, P) at every t,
#                             so per-step error is isolated from compounding.
#   jacobian_check            TVλ's EKF column Z[:,4] against a central finite
#                             difference of y_pred wrt beta[4]. That column is
#                             the one thing ref and fast share, and it enters
#                             the likelihood through F and log|F| far harder
#                             than it enters one-step MSE.
#
#   julia --project=<env> test_loglik_reference.jl [model ...]     # default: TVλ 1C
#
# 1C is the control: it runs the identical update with a fixed Z, so if 1C
# matches and TVλ does not, the λ column is the culprit and not the buffers.

using Pkg
Pkg.activate("/home/skooiker/TVNets/TVNets-1.11.6")

using YieldFactorModels
using LinearAlgebra
using DelimitedFiles
using Printf

const YFM = YieldFactorModels

ENV["OPENBLAS_NUM_THREADS"] = "1"
LinearAlgebra.BLAS.set_num_threads(1)

const ROOT   = @__DIR__
const THREAD = "8"
const IN_END = 231          # in-sample end used by thread_.jl
const TOL    = 1e-9         # relative, per step

# ---------------------------------------------------------------------------
# setup (same path as test_equivalence.jl / `run`)
# ---------------------------------------------------------------------------

function setup(model_type::String)
    data_raw, mat_raw = load_data(joinpath(ROOT, "data") * "/", THREAD)
    data       = convert(Matrix{Float64}, data_raw)
    maturities = convert(Vector{Float64}, mat_raw)
    if model_type == "AFNS"
        data ./= 100
        maturities ./= 12
    end

    N, M = length(maturities), 3
    model, canonical = YFM.create_model(model_type, maturities, N, M, Float64,
                                        joinpath(ROOT, "results_ref_scratch", model_type) * "/")

    # Both the init CSV and out_params.csv are stored in the model's own natural
    # quantities (Γ and Φ for TVλ); reconcile_init_params maps them to what
    # set_params! wants. Identity for models that need no map.
    read_params(path) = Float64.(vec(
        YFM.reconcile_init_params(model, convert(Matrix{Float64}, readdlm(path, ',')))[:, 1]))

    points = Pair{String,Vector{Float64}}[]
    init = joinpath(ROOT, "initializations", canonical, "init_params_$(canonical).csv")
    isfile(init) && push!(points, "init" => read_params(init))
    fitted = joinpath(ROOT, "results", "thread_id__$(THREAD)", canonical,
                      "$(canonical)__thread_id__$(THREAD)__out_params.csv")
    isfile(fitted) && push!(points, "fitted" => read_params(fitted))

    return model, data[:, 1:IN_END], points
end

# ---------------------------------------------------------------------------
# step 2 — the robust reference
# ---------------------------------------------------------------------------

"""
    ref_step!(m, y)

One Kalman step, written to be obviously correct rather than fast. Drives the
model's own `beta`/`P` (the observation head reads them) but does every piece of
arithmetic in fresh locals, so no buffer can be stale or aliased.

log|F| and F\\v come from an LU, not a Cholesky, on purpose: a bug in the
Cholesky path cannot cancel against itself here.
"""
function ref_step!(m, y)
    base = m.base
    YFM.predict_observation!(m)

    if any(isnan, y)
        base.beta .= base.delta .+ base.Phi * base.beta
        base.P    .= base.Phi * base.P * base.Phi' .+ base.Omega_state
        return (v = fill(NaN, base.N), logdetF = NaN, quad = NaN,
                sgn = NaN, mineig = NaN, condF = NaN)
    end

    v = y .- base.y_pred
    YFM.update_measurement_jacobian!(m)

    Z = Matrix(base.Z)
    P = Matrix(base.P)
    F = Z * P * Z' + base.Omega_obs

    logdetF, sgn = logabsdet(lu(F))   # (log|det|, sign) — sign < 0 means F is not PD
    Finv_v = F \ v
    K = P * Z' * inv(F)

    beta_upd = base.beta .+ K * v
    P_upd    = (I - K * Z) * P

    base.beta .= base.delta .+ base.Phi * beta_upd
    base.P    .= base.Phi * P_upd * base.Phi' .+ base.Omega_state

    return (v = v, logdetF = logdetF, quad = dot(v, Finv_v), sgn = sgn,
            mineig = eigmin(Symmetric(F)), condF = cond(F))
end

"""
    get_loss_ref(model, data)

`get_loss`'s likelihood, computed through `ref_step!`. Same t-range and same
t = 1 burn-in as `get_loss`, so the two numbers are directly comparable.
"""
function get_loss_ref(model, data)
    base = model.base
    YFM.initialize_filter(model)
    ll = 0.0
    c = base.N * log(2π)
    for t in 1:size(data, 2)-1
        r = ref_step!(model, view(data, :, t))
        t > 1 || continue
        ll -= 0.5 * (r.logdetF + r.quad + c)
    end
    return ll
end

"""
    fast_loglik(model, data)

What `get_loss` would return with the likelihood line at filter.jl:196 restored,
computed without editing it — same buffers, same `filter!`.
"""
function fast_loglik(model, data)
    base = model.base
    YFM.initialize_filter(model)
    ll = 0.0
    c = base.N * log(2π)
    for t in 1:size(data, 2)-1
        YFM.filter!(model, view(data, :, t), nothing)
        t > 1 || continue
        ll -= 0.5 * (base.logdetF[1] + dot(base.v, base.Finv_v) + c)
    end
    return ll
end

# ---------------------------------------------------------------------------
# step 3 — lockstep diff
# ---------------------------------------------------------------------------

reldiff(a, b) = abs(a - b) / max(1.0, abs(b))
maxreldiff(a, b) = isempty(a) ? 0.0 : maximum(reldiff.(a, b))

"""
    lockstep(model, data)

Run `filter!` and `ref_step!` from the *same* (beta, P) at every t and diff the
five things the likelihood depends on. The trajectory continues on the fast
path's own states, so what is measured at t is the error that step alone made.

Returns the worst relative deviation per quantity and the first t past `TOL`.
"""
function lockstep(model, data)
    base = model.base
    YFM.initialize_filter(model)

    worst = Dict(k => 0.0 for k in (:logdetF, :quad, :v, :beta, :P))
    first_bad = 0
    diag = NamedTuple[]

    for t in 1:size(data, 2)-1
        y = view(data, :, t)
        b0, P0 = copy(base.beta), copy(base.P)

        YFM.filter!(model, y, nothing)
        fast = (logdetF = base.logdetF[1], quad = dot(base.v, base.Finv_v),
                v = copy(base.v), beta = copy(base.beta), P = copy(base.P))

        base.beta .= b0
        base.P    .= P0
        r = ref_step!(model, y)
        ref = (logdetF = r.logdetF, quad = r.quad,
               v = r.v, beta = copy(base.beta), P = copy(base.P))

        base.beta .= fast.beta      # carry on along the fast path
        base.P    .= fast.P

        d = (logdetF = reldiff(fast.logdetF, ref.logdetF),
             quad    = reldiff(fast.quad, ref.quad),
             v       = maxreldiff(fast.v, ref.v),
             beta    = maxreldiff(fast.beta, ref.beta),
             P       = maxreldiff(vec(fast.P), vec(ref.P)))

        d = map(x -> isnan(x) ? Inf : x, d)   # a NaN on one side only is a failure
        for k in keys(worst)
            worst[k] = max(worst[k], d[k])
        end
        if first_bad == 0 && maximum(values(d)) > TOL
            first_bad = t
        end

        lam = model isa YFM.AbstractTVλDNSModel ? model.lambda[1] : NaN
        push!(diag, (t = t, lam = lam, logdetF = ref.logdetF, quad = ref.quad,
                     mineig = r.mineig, condF = r.condF, sgn = r.sgn,
                     maxdev = maximum(values(d))))
    end

    return worst, first_bad, diag
end

# ---------------------------------------------------------------------------
# step 4 — finite-difference the EKF column
# ---------------------------------------------------------------------------

"""
    jacobian_check(model, data, ts)

Z[:,4] is d y_pred / d beta[4] by hand (`update_measurement_jacobian!`). Compare
it against a central difference of `predict_observation!` at the filtered state
of each t in `ts`. Everything else in this file shares that column, so this is
the only check that covers it.
"""
function jacobian_check(model, data, ts)
    model isa YFM.AbstractTVλDNSModel || return NamedTuple[]
    base = model.base
    YFM.initialize_filter(model)
    out = NamedTuple[]

    for t in 1:maximum(ts)
        if t in ts
            b4 = base.beta[4]
            h  = 1e-6 * max(1.0, abs(b4))

            base.beta[4] = b4 + h
            YFM.predict_observation!(model)
            yp = copy(base.y_pred)

            base.beta[4] = b4 - h
            YFM.predict_observation!(model)
            ym = copy(base.y_pred)

            base.beta[4] = b4
            YFM.predict_observation!(model)
            YFM.update_measurement_jacobian!(model)

            fd = (yp .- ym) ./ (2h)
            an = copy(base.Z[:, 4])
            scale = max(maximum(abs, an), 1e-12)
            push!(out, (t = t, lam = model.lambda[1], b4 = b4,
                        maxabs = maximum(abs, fd .- an),
                        rel = maximum(abs, fd .- an) / scale))
        end
        YFM.filter!(model, view(data, :, t), nothing)
    end
    return out
end

# ---------------------------------------------------------------------------

function report(model_type::String)
    model, data, points = setup(model_type)
    isempty(points) && (println("  no parameter files found for $model_type"); return true)
    ok = true

    for (name, p) in points
        println("\n", "-"^72)
        @printf("%s @ %s (%d params)\n", model_type, name, length(p))
        println("-"^72)

        set_params!(model, p)
        ll_fast = fast_loglik(model, data)
        set_params!(model, p)
        ll_ref = get_loss_ref(model, data)
        set_params!(model, p)
        sse = get_loss(model, data)          # whatever get_loss currently scores

        @printf("  loglik  fast %.10f   ref %.10f   reldiff %.3e\n",
                ll_fast, ll_ref, reldiff(ll_fast, ll_ref))
        @printf("  get_loss (as it stands today) %.6f\n", sse)

        set_params!(model, p)
        worst, first_bad, diag = lockstep(model, data)
        println("  per-step worst relative deviation (fast vs reference):")
        for k in (:logdetF, :quad, :v, :beta, :P)
            @printf("    %-8s %.3e\n", k, worst[k])
        end
        if first_bad == 0
            println("  ✓ no step exceeds tol $(TOL)")
        else
            ok = false
            @printf("  ✗ first step past tol: t = %d\n", first_bad)
            println("     t     lambda     log|F|        v'F^-1v      eigmin(F)     cond(F)   sgn   maxdev")
            for d in diag[max(1, first_bad-2):min(end, first_bad+2)]
                @printf("  %4d  %9.5f  %12.4f  %12.4f  %12.3e  %10.3e  %4.1f  %.3e\n",
                        d.t, d.lam, d.logdetF, d.quad, d.mineig, d.condF, d.sgn, d.maxdev)
            end
        end

        # F conditioning over the whole pass — a degenerate F is a real problem
        # even when both implementations agree about it.
        @printf("  F over the pass: min eigmin %.3e   max cond %.3e   non-PD steps %d\n",
                minimum(d.mineig for d in diag), maximum(d.condF for d in diag),
                count(d -> d.mineig <= 0 || d.sgn <= 0, diag))
        # LAMBDA_START_MAX gates lambda_1 = link(mu4) only; the filtered path is
        # free to walk past it, and at large lambda*tau the loadings collapse to
        # level-only, which is a fit the likelihood can still like.
        if model isa YFM.AbstractTVλDNSModel
            hi = argmax(d -> d.lam, diag)
            @printf("  lambda over the pass: min %.5f  max %.5f (t=%d)  steps > %.2f: %d/%d\n",
                    minimum(d.lam for d in diag), hi.lam, hi.t,
                    YFM.LAMBDA_START_MAX, count(d -> d.lam > YFM.LAMBDA_START_MAX, diag),
                    length(diag))
        end

        set_params!(model, p)
        jac = jacobian_check(model, data, (2, 25, 60, 120, 180, 230))
        if !isempty(jac)
            println("  EKF column Z[:,4] vs central difference of y_pred wrt beta[4]:")
            for j in jac
                flag = j.rel > 1e-5 ? "  ✗" : ""
                @printf("    t=%3d  lambda %.5f  beta4 %8.4f  max|diff| %.3e  rel %.3e%s\n",
                        j.t, j.lam, j.b4, j.maxabs, j.rel, flag)
                j.rel > 1e-5 && (ok = false)
            end
        end
    end
    return ok
end

function main()
    models = isempty(ARGS) ? ["TVλ", "1C"] : ARGS
    ok = all([report(mt) for mt in models])
    println("\n", "="^72)
    println(ok ? "RESULT: fast filter agrees with the reference" :
                 "RESULT: DISAGREEMENT — see the ✗ lines above")
    return ok ? 0 : 1
end

exit(main())
