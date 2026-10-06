# Benchmark for the Kalman likelihood (PLAN_kalman_speedup.md, Phase 4).
#
# Reports median get_loss time and per-step filter! allocations for 1C, TVλ and
# AFNS. Run before and after a change and compare the two tables.
#
#   julia --project=<env> bench_kalman.jl

using Pkg
Pkg.activate("/home/skooiker/TVNets/TVNets-1.11.6")

using YieldFactorModels
using LinearAlgebra
using DelimitedFiles
using Printf
using Statistics

ENV["OPENBLAS_NUM_THREADS"] = "1"
LinearAlgebra.BLAS.set_num_threads(1)

const ROOT   = @__DIR__
const THREAD = "7"
const IN_END = 231
const REPS   = 50

function setup(model_type::String, float_type::Type=Float64)
    data_raw, mat_raw = load_data(joinpath(ROOT, "data") * "/", THREAD)
    data       = convert(Matrix{float_type}, data_raw)
    maturities = convert(Vector{float_type}, mat_raw)
    if model_type == "AFNS"
        data ./= float_type(100)
        maturities ./= float_type(12)
    end
    N, M = length(maturities), 3
    model, canonical = YieldFactorModels.create_model(
        model_type, maturities, N, M, float_type,
        joinpath(ROOT, "results_bench_scratch", model_type) * "/")
    raw = readdlm(joinpath(ROOT, "initializations", canonical,
                           "init_params_$(canonical).csv"), ',')
    raw = YieldFactorModels.reconcile_init_params(model, raw)
    return model, data, convert(Matrix{float_type}, raw)[:, 1]
end

println("Kalman likelihood benchmark — thread $THREAD, in-sample 1:$IN_END, $REPS reps\n")
@printf("%-6s %12s %12s %14s\n", "model", "median (ms)", "min (ms)", "alloc/step (B)")
println("-"^48)

for mt in ["1C", "TVλ", "AFNS"]
    model, data, params = setup(mt)
    insample = data[:, 1:IN_END]

    set_params!(model, params)
    get_loss(model, insample)                     # warm up / compile

    times = Float64[]
    for _ in 1:REPS
        set_params!(model, params)
        push!(times, @elapsed get_loss(model, insample))
    end

    # Allocation of a single steady-state filter! step, measured after the
    # filter has been initialized and run once.
    set_params!(model, params)
    YieldFactorModels.initialize_filter(model)
    y = @view insample[:, 1]
    YieldFactorModels.filter!(model, y, nothing)
    per_step = @allocated YieldFactorModels.filter!(model, y, nothing)

    @printf("%-6s %12.3f %12.3f %14d\n",
            mt, 1e3 * median(times), 1e3 * minimum(times), per_step)
end
