# Benchmark: analytic neural score (#9) vs the ForwardDiff score it replaced.
# Run from /home/skooiker/TVNets.
using Pkg
Pkg.activate("/home/skooiker/TVNets/TVNets-1.11.6")
using YieldFactorModels
const Y = YieldFactorModels
using BenchmarkTools, LinearAlgebra, Printf
cd(joinpath(@__DIR__, ".."))
LinearAlgebra.BLAS.set_num_threads(1)

function build_model(model_type)
    df, rl = Y.setup_data_paths(model_type, false, "", "6")
    data, mats = Y.load_data(df, "6")
    data = convert(Matrix{Float64}, data); mats = convert(Vector{Float64}, mats)
    N = length(mats); M = 3
    model, _ = Y.create_model(model_type, mats, N, M, Float64, "$rl$(model_type)/")
    p = convert(Matrix{Float64}, Y.load_initial_parameters!(model, model_type, Float64))[:, 1]
    Y.set_params!(model, p)
    return model, data, p
end

# Realistic (cache, beta, gamma, y) a few filter steps into the sample.
function realistic_point(model, data, p0)
    Y.set_params!(model, copy(p0))
    cache = Y.initialize_filter(model)
    base = model.base
    Y.update_factor_loadings!(model, base.gamma, base.Z)
    for t in 1:10
        Y.filter(model, data[:, t], cache)
    end
    Y.get_β_OLS!(base.beta, base.Z, data[:, 11], cache.ZtZ, cache.Zty)
    return cache, copy(base.beta), copy(base.gamma), copy(data[:, 11]), copy(base.Z)
end

println("="^74)
@printf("%-10s %-22s %-22s %-10s\n", "model", "analytic", "ForwardDiff", "speedup")
println("="^74)

for mt in ("1SD-NNS", "1SSD-NNS")
    model, data, p0 = build_model(mt)

    # ---- per-call score ----
    cache, beta, gamma, y, Z = realistic_point(model, data, p0)
    t_an = @belapsed Y._analytic_score_neural!($cache, $model, $beta, $gamma, $y)
    t_ad = @belapsed Y._grad_gamma_ad!($cache, $model, $beta, $gamma, $Z, $y)
    a_an = @ballocated Y._analytic_score_neural!($cache, $model, $beta, $gamma, $y)
    a_ad = @ballocated Y._grad_gamma_ad!($cache, $model, $beta, $gamma, $Z, $y)
    println("\n--- $mt : per score call ---")
    @printf("  analytic   : %8.3f µs   %6d bytes\n", t_an*1e6, a_an)
    @printf("  forwarddiff: %8.3f µs   %6d bytes\n", t_ad*1e6, a_ad)
    @printf("  SPEEDUP    : %.1fx faster, %.1fx less memory\n", t_ad/t_an, a_ad/max(a_an,1))

    # ---- end-to-end get_loss (score called once per timestep) ----
    nsteps = size(data, 2) - 1
    t_loss_an = @belapsed Y.get_loss($model, $data; K=3) setup=(Y.set_params!($model, copy($p0)))
    @printf("  get_loss(K=3) analytic   : %8.3f ms  (%d steps × 3 inits)\n", t_loss_an*1e3, nsteps)

    # force the ForwardDiff path and re-time
    @eval Y get_grad_gamma!(cache::GradientCache, m::AbstractNeuralMSEDrivenModel, beta, gamma, Z_proto, y) =
        _grad_gamma_ad!(cache, m, beta, gamma, Z_proto, y)
    t_loss_ad = @belapsed Y.get_loss($model, $data; K=3) setup=(Y.set_params!($model, copy($p0)))
    @printf("  get_loss(K=3) forwarddiff: %8.3f ms\n", t_loss_ad*1e3)
    @printf("  GET_LOSS SPEEDUP         : %.1fx faster\n", t_loss_ad/t_loss_an)

    # restore the analytic dispatch for the next model
    @eval Y get_grad_gamma!(cache::GradientCache, m::AbstractNeuralMSEDrivenModel, beta, gamma, Z_proto, y) =
        (m.transform_bool ? _analytic_score_neural!(cache, m, beta, gamma, y) :
                            _grad_gamma_ad!(cache, m, beta, gamma, Z_proto, y))
end
println("\n", "="^74)
