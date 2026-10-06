# Regression gate for the Kalman speed-up work (PLAN_kalman_speedup.md, Phase 0).
#
# Records get_loss / get_loss_array / predict for 1C, TVλ and AFNS at several
# points in parameter space. First run writes the reference; later runs compare.
#
#   julia --project=<env> test_equivalence.jl          # compare (or record if no ref)
#   julia --project=<env> test_equivalence.jl --record # force a new reference
#
# The reference file is the definition of "functionally the same" for every
# later phase. Do not regenerate it to make a failing phase pass.

using Pkg
Pkg.activate("/home/skooiker/TVNets/TVNets-1.11.6")

using YieldFactorModels
using LinearAlgebra
using DelimitedFiles
using Random
using Serialization
using Printf

ENV["OPENBLAS_NUM_THREADS"] = "1"
LinearAlgebra.BLAS.set_num_threads(1)

const ROOT     = @__DIR__
const REF_PATH = joinpath(ROOT, "test_equivalence_ref.jls")
const THREAD   = "7"
const MODELS   = ["1C", "TVλ", "AFNS"]
const N_DRAWS  = 3          # random parameter draws per model, on top of the init point
const IN_END   = 231        # in-sample end used by thread_.jl

const RTOL_LOSS  = 1e-8     # scalar log-likelihood
const RTOL_ARRAY = 1e-6     # loss array / predictions
const ATOL_ARRAY = 1e-8

"""
Build one model exactly as `run` does: data/maturities conversion, model
construction, init parameters (reconciled to this thread's N).
Returns (model, data, init_params_constrained).
"""
function setup(model_type::String, float_type::Type=Float64)
    data_raw, mat_raw = load_data(joinpath(ROOT, "data") * "/", THREAD)
    data       = convert(Matrix{float_type}, data_raw)
    maturities = convert(Vector{float_type}, mat_raw)

    # AFNS works in decimals and years (see `run`)
    if model_type == "AFNS"
        data ./= float_type(100)
        maturities ./= float_type(12)
    end

    N, M = length(maturities), 3
    model, canonical = YieldFactorModels.create_model(
        model_type, maturities, N, M, float_type,
        joinpath(ROOT, "results_equivalence_scratch", model_type) * "/")

    raw = readdlm(joinpath(ROOT, "initializations", canonical,
                           "init_params_$(canonical).csv"), ',')
    raw = YieldFactorModels.reconcile_init_params(model, raw)
    params = convert(Matrix{float_type}, raw)[:, 1]

    return model, data, params
end

"""
Parameter points to test: the init point, then `N_DRAWS` multiplicative
perturbations of it (fixed seed).

The jitter is multiplicative, and applied in constrained space, so it preserves
the sign of every entry — positive variances stay positive — without needing the
inverse transform. That matters because `untransform_params` throws on the TVλ
init point: its Phi diagonal holds -1.0166, outside the (-1, 1) domain of
`from_11_to_R`. Non-stationary points are still worth filtering here; the models
have their own fallbacks and the gate only cares that the numbers do not move.
"""
function param_points(model, init::Vector{T}) where {T<:Real}
    points = [copy(init)]
    rng = MersenneTwister(20260804)
    for _ in 1:N_DRAWS
        push!(points, init .* (one(T) .+ T(0.05) .* randn(rng, T, length(init))))
    end
    return points
end

"""
Every observable of one model at one parameter point. `set_params!` is repeated
before each call because the filter mutates beta/P/Z/gamma in place.
"""
function observe(model, data, params)
    insample = data[:, 1:IN_END]

    set_params!(model, params)
    loss = get_loss(model, insample)

    set_params!(model, params)
    loss_array = YieldFactorModels.get_loss_array(model, data; K=1)

    set_params!(model, params)
    pred = predict(model, insample)

    return Dict{Symbol,Any}(
        :loss              => loss,
        :loss_array        => loss_array,
        :preds             => pred.preds,
        :factors           => pred.factors,
        :states            => pred.states,
        :factor_loadings_1 => pred.factor_loadings_1,
        :factor_loadings_2 => pred.factor_loadings_2,
    )
end

function collect_all()
    out = Dict{String,Vector{Dict{Symbol,Any}}}()
    for mt in MODELS
        model, data, init = setup(mt)
        out[mt] = [observe(model, data, p) for p in param_points(model, init)]
        # Printed so a gate that has gone blind (every point degenerate) is visible.
        @printf("  %-5s losses: %s\n", mt,
                join((string(round(o[:loss]; digits=4)) for o in out[mt]), ", "))
    end
    return out
end

# NaN == NaN and Inf == Inf here: a degenerate point must stay degenerate.
function same(a, b; rtol, atol)
    size(a) == size(b) || return false
    for (x, y) in zip(a, b)
        if isnan(x) || isnan(y)
            isnan(x) && isnan(y) || return false
        elseif isinf(x) || isinf(y)
            x == y || return false
        else
            isapprox(x, y; rtol=rtol, atol=atol) || return false
        end
    end
    return true
end
same(a::Number, b::Number; rtol, atol) = same([a], [b]; rtol=rtol, atol=atol)

function compare(ref, cur)
    failures = String[]
    for mt in MODELS
        haskey(ref, mt) || (push!(failures, "$mt: missing from reference"); continue)
        length(ref[mt]) == length(cur[mt]) ||
            (push!(failures, "$mt: point count changed"); continue)

        for (i, (r, c)) in enumerate(zip(ref[mt], cur[mt]))
            for key in (:loss, :loss_array, :preds, :factors, :states,
                        :factor_loadings_1, :factor_loadings_2)
                rtol = key === :loss ? RTOL_LOSS : RTOL_ARRAY
                atol = key === :loss ? 0.0 : ATOL_ARRAY
                if !same(r[key], c[key]; rtol=rtol, atol=atol)
                    dev = try
                        maximum(abs.(vec(collect(r[key])) .- vec(collect(c[key]))))
                    catch
                        NaN
                    end
                    push!(failures,
                          @sprintf("%s point %d: %s differs (max abs dev %.3e)",
                                   mt, i, key, dev))
                end
            end
        end
    end
    return failures
end

function main()
    record = "--record" in ARGS || !isfile(REF_PATH)
    println(record ? "Recording reference..." : "Comparing against reference...")

    cur = collect_all()

    if record
        serialize(REF_PATH, cur)
        println("\nReference written to $(REF_PATH)")
        println("GATE: RECORDED")
        return 0
    end

    failures = compare(deserialize(REF_PATH), cur)
    if isempty(failures)
        println("\nGATE: PASS — all $(length(MODELS)) models match the reference")
        return 0
    end
    println("\nGATE: FAIL")
    foreach(f -> println("  ", f), failures)
    return 1
end

exit(main())
