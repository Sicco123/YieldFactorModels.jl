# Reproducibility check against results/thread_id__8.
#
# For each model: pick ONE moving-window task_id (a different one per model),
# rebuild exactly what run_forecast_window_database does for that task, re-run
# the estimation from its initialization, and compare against the parameters
# stored in that model's forecast database.
#
# Read-only: nothing is written to results/.
#
# Usage:
#   julia YieldFactorModels.jl/test_repro_thread8.jl              # all models
#   julia YieldFactorModels.jl/test_repro_thread8.jl 1C           # one model
#   julia YieldFactorModels.jl/test_repro_thread8.jl TVλ 233      # one model, fixed task

using Pkg
Pkg.activate("/home/skooiker/TVNets/TVNets-1.11.6")
Pkg.instantiate()

using YieldFactorModels
using Random
using LinearAlgebra
using Printf

ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["OMP_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"
LinearAlgebra.BLAS.set_num_threads(1)

const YFM = YieldFactorModels
const SQLite = YFM.SQLite
const DBInterface = YFM.DBInterface
const deserialize = YFM.Serialization.deserialize

# Settings of the thread-8 run (see test-thread_7.jl / test.jl)
const THREAD_ID       = "8"
const IN_SAMPLE_END   = 231
const IN_SAMPLE_START = 1
const FLOAT           = Float64
const MAX_GROUP_ITERS = 10
const GROUP_TOL       = 1e-6
const WINDOW_TYPE     = "moving"
const SEED            = 43          # run()'s default; estimation itself uses no RNG
const LOSS_TOL        = 1e-5   # relative: |dloss| / max(|loss_stored|,1)

const MODELS = ["1C", "NS", "NNS", "RW", "SD-NS", "SSD-NS",
                "1SD-NNS", "1SSD-NNS", "2SSD-NNS", "3SSD-NNS", "AFNS", "TVλ"]

results_dir(m) = "YieldFactorModels.jl/results/thread_id__$(THREAD_ID)/$(m)/"

"Task ids for which stored parameters exist: the merged db, else the leftover shards."
function available_tasks(m::String)
    dbdir = string(results_dir(m), "db/")
    merged = string(dbdir, "forecasts_$(WINDOW_TYPE)_merged.sqlite3")
    if isfile(merged)
        db = SQLite.DB(merged)
        ids = [row[1] for row in DBInterface.execute(db, "SELECT task_id FROM forecasts;")]
        SQLite.close(db)
        return sort(Int.(ids))
    end
    ids = Int[]
    for f in readdir(dbdir)
        mt = match(Regex("^forecasts_$(WINDOW_TYPE)_(\\d+)\\.sqlite3\$"), f)
        mt === nothing || push!(ids, parse(Int, mt.captures[1]))
    end
    return sort(ids)
end

"Stored parameter vector for one task, from the merged db or that task's shard."
function stored_params(m::String, task_id::Int)
    dbdir = string(results_dir(m), "db/")
    path = string(dbdir, "forecasts_$(WINDOW_TYPE)_merged.sqlite3")
    isfile(path) || (path = string(dbdir, "forecasts_$(WINDOW_TYPE)_$(task_id).sqlite3"))
    isfile(path) || return nothing
    db = SQLite.DB(path)
    p = nothing
    for row in DBInterface.execute(db, "SELECT params FROM forecasts WHERE task_id = ?;", (task_id,))
        p = vec(collect(deserialize(IOBuffer(row[1]))))
    end
    SQLite.close(db)
    return p
end

"Build model, data and initial parameters exactly as `run` does, before the task loop."
function setup(m::String)
    Random.seed!(SEED)
    data_folder, results_location = YFM.setup_data_paths(m, false, "", THREAD_ID)
    data, maturities = YFM.load_data(data_folder, THREAD_ID)
    data = convert(Matrix{FLOAT}, data)
    maturities = convert(Vector{FLOAT}, maturities)
    if m == "AFNS"
        data ./= FLOAT(100)
        maturities ./= FLOAT(12)
    end
    N, M = length(maturities), 3
    model, model_type = YFM.create_model(m, maturities, N, M, FLOAT, "$(results_location)$(m)/")
    param_groups = YFM.get_param_groups(model, String[])
    all_params = YFM.load_initial_parameters!(model, model_type, FLOAT)
    set_params!(model, all_params[:, 1])
    all_params[:, 1] = YFM.load_static_parameters!(model, model_type, results_location, THREAD_ID, all_params[:, 1])
    all_params = convert(Matrix{FLOAT}, all_params)
    return model, all_params, param_groups, data
end

"One iteration of the pipeline's task loop, mutating `model` and `all_params` as it does."
function fit_task!(model, all_params, param_groups, data, task_id::Int)
    all_params = YFM.read_static_params_from_db(model, task_id, all_params; window_type=WINDOW_TYPE)
    span = task_id - (IN_SAMPLE_END - IN_SAMPLE_START)
    window = data[:, span:task_id]
    _, _, fitted, _ = YFM.run_estimation!(model, window, size(window, 2), all_params,
                                          param_groups, MAX_GROUP_ITERS, GROUP_TOL; printing=false)
    set_params!(model, FLOAT.(fitted))     # the loop does this before predicting
    return vec(fitted), window, all_params
end

"Re-estimate model `m` on the moving window ending at `task_id`; return (fitted, stored, loss_fit, loss_stored)."
function reproduce(m::String, task_id::Int; chain::Vector{Int}=Int[])
    model, all_params, param_groups, data = setup(m)

    # Optionally replay preceding tasks in the same process first, as the pipeline would
    for t in chain
        _, _, all_params = fit_task!(model, all_params, param_groups, data, t)
    end

    fitted, window, _ = fit_task!(model, all_params, param_groups, data, task_id)

    old = stored_params(m, task_id)
    old === nothing && return (fitted, nothing, NaN, NaN)

    set_params!(model, FLOAT.(fitted));  loss_fit = get_loss(model, window)
    if length(old) == length(fitted)
        set_params!(model, FLOAT.(old)); loss_old = get_loss(model, window)
    else
        loss_old = NaN
    end
    return (fitted, old, loss_fit, loss_old)
end

function main(args)
    models = isempty(args) ? MODELS : [args[1]]
    fixed_task = length(args) >= 2 ? parse(Int, args[2]) : nothing

    # 3rd arg: an integer repeats the task N times in one process; "chain:a,b,c"
    # replays tasks a,b,c first, reusing model and all_params like the pipeline does.
    if length(args) >= 3
        m, task_id = args[1], parse(Int, args[2])
        if startswith(args[3], "chain:")
            ch = parse.(Int, split(args[3][7:end], ","))
            fitted, old, lf, lo = reproduce(m, task_id; chain=ch)
            @printf("chain %-18s %-10s %6d  loss_refit %-16.10g loss_stored %-16.10g  p1 %.10g\n",
                    args[3][7:end], m, task_id, lf, lo, fitted[1])
        else
            for k in 1:parse(Int, args[3])
                fitted, old, lf, lo = reproduce(m, task_id)
                @printf("repeat %d: %-10s %6d  loss_refit %-16.10g loss_stored %-16.10g  p1 %.10g\n",
                        k, m, task_id, lf, lo, fitted[1])
            end
        end
        exit(0)
    end

    rng = MersenneTwister(20260824)   # fixed draw so a failure can be re-run verbatim
    rows = String[]
    failed = false

    for m in MODELS                    # draw for every model, so a single-model run
        tasks = available_tasks(m)     # picks the same task as the full run
        pick = isempty(tasks) ? nothing : rand(rng, tasks)
        m in models || continue
        task_id = something(fixed_task, pick, -1)

        if task_id < 0
            push!(rows, @sprintf("%-10s %6s  %-10s %-10s %-12s  %s", m, "-", "-", "-", "-", "NO STORED PARAMS"))
            failed = true
            continue
        end

        println("\n=== $m  task_id=$task_id ===")
        local fitted, old, lf, lo
        try
            fitted, old, lf, lo = reproduce(m, task_id)
        catch e
            push!(rows, @sprintf("%-10s %6d  %-10s %-10s %-12s  ERROR: %s", m, task_id, "-", "-", "-", sprint(showerror, e)))
            failed = true
            continue
        end

        if old === nothing
            push!(rows, @sprintf("%-10s %6d  %-10s %-10s %-12s  NO STORED PARAMS", m, task_id, "-", "-", "-"))
            failed = true
            continue
        end
        if length(old) != length(fitted)
            push!(rows, @sprintf("%-10s %6d  %-10s %-10s %-12s  LAYOUT MISMATCH (stored %d vs now %d)",
                                 m, task_id, "-", "-", "-", length(old), length(fitted)))
            failed = true
            continue
        end

        maxabs = maximum(abs.(fitted .- old))
        maxrel = maximum(abs.(fitted .- old) ./ max.(abs.(old), 1e-8))
        dloss  = abs(lf - lo) / max(abs(lo), 1.0)   # relative: losses run into the thousands
        ok = isfinite(dloss) && dloss <= LOSS_TOL
        ok || (failed = true)
        push!(rows, @sprintf("%-10s %6d  %-10.3e %-10.3e %-12.3e %-14.7g %-14.7g  %s",
                             m, task_id, maxabs, maxrel, dloss, lf, lo, ok ? "PASS" : "FAIL"))
    end

    println("\n", "="^110)
    @printf("%-10s %6s  %-10s %-10s %-12s %-14s %-14s  %s\n", "model", "task", "maxabs", "maxrel", "reldloss", "loss_refit", "loss_stored", "verdict")
    println("="^110)
    foreach(println, rows)
    println("="^110)
    println("Tolerance: |loss(refit) - loss(stored)| / max(|loss_stored|,1) <= $LOSS_TOL on the same window.")
    println("Parameters may differ where the objective is flat; the loss is the check that matters.")

    exit(failed ? 1 : 0)
end

main(ARGS)
