



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


# 3) use the DB handles; 4) avoid shadowing starting_point; 5) handle reestimate=false
function run_forecast_window_database(model::AbstractYieldFactorModel, data::AbstractMatrix, thread_Id::String,
    in_sample_end::Int, in_sample_start::Int, out_sample_end::Int, forecast_horizon::Int, window_type::String,
     init_params::AbstractMatrix;
    param_groups::Vector{String}=String[], max_group_iters::Int=10, group_tol::Real=1e-8, reestimate::Bool=true)

    total_tasks = in_sample_end:out_sample_end
    # if the merged file aready exists return 
    if isfile("$(model.base.results_folder)db/forecasts_$(window_type)_merged.sqlite3")
        println("The forecast for $(window_type) are already present in the db folder.")
        export_all_csv(model,total_tasks, thread_Id; window_type=window_type) 
        return 
    end

    all_params = init_params

    task_db_path         = "$(model.base.results_folder)db/task_states_$(window_type).sqlite3"
    proc_counter_db_path = "$(model.base.results_folder)db/proc_counter_$(window_type).sqlite3"
    forecast_db_path     = "$(model.base.results_folder)db/forecasts_$(window_type).sqlite3"

    # make db dir if it doesn't exist
    mkpath(dirname(task_db_path))

    task_db         = init_task_database(task_db_path, in_sample_end, out_sample_end)     # <-- capture handle
    proc_counter_db = init_proc_counter_db(proc_counter_db_path)                      # <-- capture handle
    #init_forecast_db(forecast_db_path)

    running_processes = incr_working!(proc_counter_db)
    println("Thread $thread_Id started. Running processes: $running_processes")

    # Timing for run_estimation! calls
    est_total_time = 0.0
    est_count = 0
    t = 0.0

    #try 
        start_idx = starting_point(running_processes, out_sample_end - in_sample_end)         # <-- renamed
        
        reordered_tasks = vcat(total_tasks[start_idx:end], total_tasks[1:start_idx-1])

        for task_id in reordered_tasks
            s = claim_task(task_db, task_id)
            println("Thread $thread_Id claimed task $task_id with status $s (window: $window_type) ")
            if s == 0
                # read static_params_from_db
            
                all_params = read_static_params_from_db(model, task_id, all_params; window_type=window_type)

                # compute window-specific data + estimation if requested
                if window_type == "expanding"
                    temp_forecast_data = hcat(data, fill(NaN, size(data,1), forecast_horizon))
                    if reestimate
                        t = @elapsed begin
                            old_params, loss, params, ir = run_estimation!(model, data, task_id, all_params, param_groups, max_group_iters, group_tol; printing=false)
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
                            old_params, loss, params, ir = run_estimation!(model, data, task_id, all_params, param_groups, max_group_iters, group_tol; printing=false)
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


                save_oos_forecast_sharded!(forecast_db_path, model, thread_Id, window_type,
                                        task_id, results, loss, params; forecast_horizon=forecast_horizon)
                if est_count > 0
                    avg = est_total_time / est_count
                    println("Thread $thread_Id estimation summary: performed $est_count estimations, average seconds per task: $avg, elapsed time for last task: $t s")
                else
                    println("Thread $thread_Id estimation summary: no re-estimation performed.")
                end
            else
                continue
            end

           
        end

        num_procs = decr_working!(proc_counter_db)   # post-decrement count
        println("Thread $thread_Id finished all tasks. Remaining processes: $num_procs")

        if est_count > 0
            avg = est_total_time / est_count
            println("Thread $thread_Id estimation summary: performed $est_count estimations, average seconds per task: $avg")
        else
            println("Thread $thread_Id estimation summary: no re-estimation performed.")
        end

        SQLite.close(task_db)
        SQLite.close(proc_counter_db)

        if num_procs == 0
            println("Merging forecast shards...")
            sleep(1)
            merge_forecast_shards!(forecast_db_path; task_ids=collect(in_sample_end:out_sample_end), delete_shards=true)
            export_all_csv(model, total_tasks,  thread_Id; window_type=window_type) 
        end
    # catch e
    #     num_procs = decr_working!(proc_counter_db)   # post-decrement
    #     println("Thread $thread_Id encountered an error: $e. Remaining processes: $num_procs")
    #     SQLite.close(task_db)
    #     SQLite.close(proc_counter_db)
    #     rethrow(e)
    # end
end