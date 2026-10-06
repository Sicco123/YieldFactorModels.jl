# Why does an RMSE fit forecast better out of sample than an ML fit?
#
# test_loglik_reference.jl established that the likelihood is computed correctly
# (fast filter vs an LU reference: 1e-14). So the gap is not in the objective's
# value, it is in which optimum the objective picks. This estimates the SAME
# model on the SAME window under both objectives — one flag apart, everything
# else identical — and puts the two parameter vectors side by side against the
# out-of-sample forecasts they produce.
#
#   julia --project=<env> test_objective_ab.jl
#
# Out-of-sample here holds the parameters fixed and rolls the forecast origin
# (the pipeline's `reestimate=false` mode). That is deliberately not the full
# 250-window protocol: fixing the parameters is what isolates the effect of the
# parameters, which is the question.

using Pkg
Pkg.activate("/home/skooiker/TVNets/TVNets-1.11.6")

using YieldFactorModels
using LinearAlgebra
using DelimitedFiles
using Printf
using Statistics

const YFM = YieldFactorModels

ENV["OPENBLAS_NUM_THREADS"] = "1"
LinearAlgebra.BLAS.set_num_threads(1)

const ROOT      = @__DIR__
const THREAD    = "8"
const IN_END    = 231       # in-sample end used by thread_.jl
const HORIZON   = 12
const MODEL     = isempty(ARGS) ? "TVλ" : ARGS[1]   # 1C is the control: no λ state

# ---------------------------------------------------------------------------

function setup()
    data_raw, mat_raw = load_data(joinpath(ROOT, "data") * "/", THREAD)
    data       = convert(Matrix{Float64}, data_raw)
    maturities = convert(Vector{Float64}, mat_raw)
    N, M = length(maturities), 3

    model, canonical = YFM.create_model(MODEL, maturities, N, M, Float64,
                                        joinpath(ROOT, "results_ab_scratch") * "/")
    raw = readdlm(joinpath(ROOT, "initializations", canonical,
                           "init_params_$(canonical).csv"), ',')
    init = convert(Matrix{Float64}, YFM.reconcile_init_params(model, raw))[:, 1:1]
    return model, data, init
end

"""
    lambda_path(model, data)

The filtered λ over the in-sample window, at the model's current parameters.
"""
function lambda_path(model, data)
    YFM.initialize_filter(model)
    path = Float64[]
    for t in 1:size(data, 2)-1
        YFM.filter!(model, view(data, :, t), nothing)
        push!(path, model.lambda[1])
    end
    return path
end

"""
    oos_rmse(model, data, origins)

h = 1..HORIZON forecast RMSE, pooled over maturities and forecast origins, at
the model's current parameters. Same construction as `run_forecast_window_database`
with `reestimate=false`: filter through everything up to the origin, then run
the transition forward over HORIZON NaN observations.

Returns (model RMSE by h, random-walk RMSE by h).
"""
function oos_rmse(model, data, origins)
    se  = zeros(HORIZON)
    srw = zeros(HORIZON)
    n   = 0
    for origin in origins
        temp = hcat(data[:, 1:origin], fill(NaN, size(data, 1), HORIZON - 1))
        preds = predict(model, temp).preds          # preds[:, k] = E[y_{k+1} | F_k]
        for h in 1:HORIZON
            actual = @view data[:, origin+h]
            se[h]  += sum(abs2, preds[:, origin+h-1] .- actual)
            srw[h] += sum(abs2, data[:, origin] .- actual)   # RW: last observed curve
        end
        n += 1
    end
    nobs = n * size(data, 1)
    return sqrt.(se ./ nobs), sqrt.(srw ./ nobs)
end

"""
    fit(model, data, init, objective)

Estimate on data[:, 1:IN_END] under `objective`. Everything except
`YFM.OBJECTIVE[]` is identical between the two calls.
"""
function fit(model, data, init, objective::Symbol; free_sigma::Bool = false)
    YFM.OBJECTIVE[] = objective
    groups = YFM.get_param_groups(model, String[])
    # Slot 1 is the observation variance, pinned at the 1C value by
    # get_param_groups. It is the only parameter that can absorb the likelihood's
    # log|F| term directly; with it frozen, ML's only lever on log|F| is the state
    # covariance — which is what the forecasts ride on. Free it to test that.
    free_sigma && model isa YFM.AbstractTVλDNSModel && (groups[1] = "2")
    t = @elapsed begin
        _, loss, params, _ = YFM.run_estimation!(model, data, IN_END, copy(init),
                                                 groups, 10, 1e-8; printing = false)
    end
    @printf("  %-8s done in %.1f s, objective at optimum = %.4f\n", objective, t, loss)
    return params
end

function describe(model, data, params, label, origins)
    base = model.base
    insample = data[:, 1:IN_END]

    set_params!(model, params)
    YFM.OBJECTIVE[] = :loglik
    ll = get_loss(model, insample)
    set_params!(model, params)
    YFM.OBJECTIVE[] = :sse
    sse = -get_loss(model, insample)
    set_params!(model, params)
    rr = YFM.rrmse_vs_rw(model, insample)

    set_params!(model, params)
    lam = lambda_path(model, insample)

    set_params!(model, params)
    sigma2 = base.Omega_obs[1, 1]
    rho    = YFM.spectral_radius(base.Phi)
    tv     = model isa YFM.AbstractTVλDNSModel    # 1C has no 4th state
    phi44  = tv ? base.Phi[4, 4] : NaN
    sig44  = tv ? base.Omega_state[4, 4] : NaN
    gam44  = tv ? model.Gamma[4, 4] : NaN

    set_params!(model, params)
    rmse, rw = oos_rmse(model, data, origins)

    println("\n", "="^78)
    println(label)
    println("="^78)
    @printf("  in-sample loglik      %12.3f\n", ll)
    @printf("  in-sample SSE         %12.3f    (1-step RRMSE vs RW %.4f)\n", sse, rr)
    @printf("  sigma^2 (obs var)     %12.6f\n", sigma2)
    @printf("  rho(Phi) %.5f   Phi44 %.5f   Sigma44 %.6f   Gamma44 %.6f\n",
            rho, phi44, sig44, gam44)
    @printf("  lambda path: min %.4f  median %.4f  max %.4f  sd %.4f  steps>0.5 %d/%d\n",
            minimum(lam), median(lam), maximum(lam), std(lam),
            count(>(0.5), lam), length(lam))
    println("  OOS RMSE by horizon (parameters fixed, origin rolled):")
    println("      h        model        RW     ratio")
    for h in (1, 3, 6, 12)
        @printf("    %3d  %10.4f %9.4f %9.4f\n", h, rmse[h], rw[h], rmse[h] / rw[h])
    end
    @printf("    pooled h=1..12 ratio %.4f\n",
            sqrt(sum(abs2, rmse) / sum(abs2, rw)))
    return (ll = ll, sse = sse, rr = rr, lam = lam, rmse = rmse, rw = rw,
            sigma2 = sigma2, rho = rho, phi44 = phi44, sig44 = sig44, gam44 = gam44)
end

function run()
    model, data, init = setup()
    origins = IN_END:(size(data, 2) - HORIZON)
    @printf("%s thread %s: in-sample 1:%d, %d forecast origins, horizon %d\n",
            MODEL, THREAD, IN_END, length(origins), HORIZON)

    println("\nEstimating under each objective (identical apart from OBJECTIVE[])...")
    p_ll   = fit(model, data, init, :loglik)
    p_sse  = fit(model, data, init, :sse)
    p_llfs = fit(model, data, init, :loglik; free_sigma = true)

    r_ll   = describe(model, data, p_ll,  "fitted by :loglik  (maximum likelihood)", origins)
    r_sse  = describe(model, data, p_sse, "fitted by :sse     (in-sample 1-step RMSE)", origins)
    r_llfs = describe(model, data, p_llfs, "fitted by :loglik, observation variance freed", origins)

    println("\n", "="^78)
    println("SUMMARY")
    println("="^78)
    function row(name, f, fmt)
        F = Printf.Format(fmt)
        @printf("  %-22s", name)
        for r in (r_ll, r_sse, r_llfs)
            print(Printf.format(F, f(r)))
        end
        println()
    end
    @printf("  %-22s %14s %14s %14s\n", "", ":loglik", ":sse", ":loglik+free s2")
    row("in-sample loglik",    r -> r.ll,            "%14.3f")
    row("in-sample SSE",       r -> r.sse,           "%14.3f")
    row("in-sample RRMSE/RW",  r -> r.rr,            "%14.4f")
    row("lambda sd",           r -> std(r.lam),      "%14.4f")
    row("lambda max",          r -> maximum(r.lam),  "%14.4f")
    row("Sigma44 (lam innov)", r -> r.sig44,         "%14.6f")
    row("rho(Phi)",            r -> r.rho,           "%14.5f")
    row("sigma^2 (obs var)",   r -> r.sigma2,        "%14.6f")
    for h in (1, 12)
        row("OOS RMSE/RW h=$h", r -> r.rmse[h] / r.rw[h], "%14.4f")
    end

    writedlm(joinpath(ROOT, "test_objective_ab_params.csv"),
             hcat(vec(init), p_ll, p_sse, p_llfs), ',')
    println("\nparameters written to test_objective_ab_params.csv (init, loglik, sse, loglik+free sigma^2)")
    return 0
end

exit(run())
