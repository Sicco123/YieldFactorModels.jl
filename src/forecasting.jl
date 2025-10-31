



function starting_point(running_processes, num_tasks)
    #when running_processes == 1 we start at task 1, when running_processes == 2 we start at halfway point, 3 we start at quarter point, 4 we start at 3/4 point, 5 we start at 1/8, 6 we start at 3/8, 7 we start at 5/8, 8 we start at 7/8, 9 we start at 1/16, etc.
    if running_processes == 1
        return 1
    else
        denom = 2^(running_processes - 1)
        return div((running_processes - 1) * num_tasks, denom) + 1
    end
end

# Function to perform rolling forecasts with parallel processing
function run_rolling_forecasts(model::AbstractYieldFactorModel, data::AbstractMatrix,  thread_Id::String, in_sample_end::Int, in_sample_start::Int, forecast_horizon::Int,  init_params::AbstractMatrix; window_type::String="both",  param_groups::Vector{String}=String[], max_group_iters::Int=10, group_tol::Real=1e-8, reestimate::Bool=true)

    out_sample_end = size(data, 2)

 
    forecast_func = run_forecast_window_database

    if window_type == "both"
        # Sequential forecasts
        forecast_func(model, data,  thread_Id, in_sample_end, in_sample_start, out_sample_end, forecast_horizon, 
                        "expanding",   init_params; param_groups=param_groups,
                        max_group_iters=max_group_iters, group_tol=group_tol, reestimate=reestimate 
                        )

        forecast_func(model, data,  thread_Id, in_sample_end, in_sample_start, out_sample_end, forecast_horizon, 
                        "moving",   init_params; param_groups=param_groups,
                        max_group_iters=max_group_iters, group_tol=group_tol, reestimate=reestimate 
                        )
    elseif window_type == "expanding"
        forecast_func(model, data,  thread_Id, in_sample_end,in_sample_start, out_sample_end, forecast_horizon, 
                        "expanding",   init_params; param_groups=param_groups,
                        max_group_iters=max_group_iters, group_tol=group_tol, reestimate=reestimate
                        )
    elseif window_type == "moving"
        forecast_func(model, data,  thread_Id, in_sample_end,in_sample_start, out_sample_end, forecast_horizon, 
                        "moving",   init_params; param_groups=param_groups,
                        max_group_iters=max_group_iters, group_tol=group_tol, reestimate=reestimate)
    else
        error("Invalid window type")
    end
end

# atomic per-task lock using mkdir
function _acquire_task_lock(lockroot::AbstractString, window_type::AbstractString, task_id::Integer)
    lockdir = joinpath(lockroot, window_type, "task_$(task_id).lock")
    mkpath(dirname(lockdir))
    try
        mkdir(lockdir)                 # atomic
        return lockdir                 # we own the lock
    catch e
        # If the lock dir exists, someone else holds it → skip
        if isdir(lockdir)
            return nothing
        end
 
        if occursin("EEXIST", sprint(showerror, e)) || occursin("File exists", sprint(showerror, e))
            return nothing
        end
        rethrow(e)                     # real error → bubble up
    end
end

function _release_task_lock(lockdir::AbstractString)
    try
        isdir(lockdir) && rm(lockdir; recursive=true, force=true)
    catch
        # best-effort; ignore
    end
end

function run_forecast_window_database(model::AbstractYieldFactorModel, data::AbstractMatrix, thread_Id::String,
    in_sample_end::Int, in_sample_start::Int, out_sample_end::Int, forecast_horizon::Int, window_type::String,
     init_params::AbstractMatrix;
    param_groups::Vector{String}=String[], max_group_iters::Int=10, group_tol::Real=1e-8, reestimate::Bool=true)

    tasks = collect(in_sample_end:out_sample_end)
    # shuffle tasks 
    tasks = Random.shuffle(RandomDevice(), tasks)
   
    merged_path = "$(model.base.results_folder)db/forecasts_$(window_type)_merged.sqlite3"
    if isfile(merged_path)
        println("Merged forecasts already exist for $(window_type).")
        export_all_csv(model, tasks, thread_Id; window_type=window_type)
        return
    end

    forecast_db_base = "$(model.base.results_folder)db/forecasts_$(window_type).sqlite3"
    lockroot = "$(model.base.results_folder)db/locks"
    mkpath(joinpath(lockroot, window_type))

    all_params = init_params

    est_total_time = 0.0
    est_count = 0
    t = 0.0

    for task_id in tasks
        shard_path = _forecast_path(forecast_db_base, task_id)
        if isfile(shard_path)
            continue
        end

        lockdir = _acquire_task_lock(lockroot, window_type, task_id)
        if isnothing(lockdir)
            continue
        end

        try
            all_params = read_static_params_from_db(model, task_id, all_params; window_type=window_type)
            if window_type == "expanding"
                temp_forecast_data = hcat(data, fill(NaN, size(data,1), forecast_horizon))
                
                if reestimate
                    t = @elapsed begin
                        _, loss, params, _ = run_estimation!(model, data, task_id, all_params,
                                                             param_groups, max_group_iters, group_tol; printing=false)
                    end
                    est_total_time += t
                    est_count += 1
                else
                    loss = NaN; params = all_params[:,1]
                end

            elseif window_type == "moving"
                span = task_id - (in_sample_end - in_sample_start)
                temp_forecast_data = hcat(data[:, span:task_id], fill(NaN, size(data,1), forecast_horizon))

                if reestimate
                    t = @elapsed begin
                        _, loss, params, _ = run_estimation!(model, data, task_id, all_params,
                                                             param_groups, max_group_iters, group_tol; printing=false)
                    end
                    est_total_time += t
                    est_count += 1
                else
                    loss = NaN; params = all_params[:,1]
                end
            else
                error("Invalid window type")
            end

            set_params!(model, params)
            results = predict(model, temp_forecast_data)


            save_oos_forecast_sharded!(forecast_db_base, model, thread_Id, window_type,
                                       task_id, results, loss, params; forecast_horizon=forecast_horizon)

            if est_count > 0
                println("Thread $thread_Id estimation summary: performed $est_count estimations, average seconds per task: $(est_total_time / est_count), last: $t")
            else
                println("Thread $thread_Id estimation summary: no re-estimation performed.")
            end

        finally
            _release_task_lock(lockdir)
        end
    end

    # merge only if all shards exist
    shard_paths = [_forecast_path(forecast_db_base, t) for t in tasks]
    if all(isfile.(shard_paths))
        println("All shards exist. Merging...")
        merge_forecast_shards!(forecast_db_base; task_ids=tasks, delete_shards=true)
        export_all_csv(model, tasks, thread_Id; window_type=window_type)
    else
        println("Not all shards available for $(model.base.model_string). Skipping merge for now.")
    end

    return nothing
end