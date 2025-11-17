function read_static_params_from_db(model::AbstractYieldFactorModel, task_id::Int, all_params::AbstractMatrix; window_type::AbstractString="expanding")
    return all_params
end

function read_static_params_from_db(model::AbstractMSEDrivenModel, task_id::Int, all_params::AbstractMatrix ; window_type::AbstractString="expanding"   )
    static_model_name = get_static_model_type(model)
    # get parent directory of model.base.results_folder 
    results_dir = dirname(dirname(model.base.results_folder))
    static_db_folder = string(results_dir, "/$static_model_name/db/")

    # from forecasts_${window_type}_merged.sqlite3 get the params 
    forecast_db_path = string(static_db_folder, "forecasts_", window_type, "_merged.sqlite3")
    db = SQLite.DB(forecast_db_path)
    rows = DBInterface.execute(db, "SELECT params FROM forecasts WHERE task_id = ?;", (task_id,))

   
    try
        static_params = nothing
        for row in rows
            static_params = _deser(row[1])
            # expand dimension of static_params
            static_params = reshape(static_params, size(static_params, 1), 1)
           
        end
        all_params[:,1] .= initialize_with_static_params(model, all_params[:,1], static_params)

    catch e
        println("Error reading static params for task $task_id: $e")
        SQLite.close(db)
        rethrow(e)
    end
    SQLite.close(db)
    return all_params
end

function read_params_from_db(model::AbstractYieldFactorModel, task_id::Int, all_params::AbstractMatrix; window_type::AbstractString="expanding")
    model_name = model.base.model_string
    results_dir = dirname(dirname(model.base.results_folder))
    db_folder = string(results_dir, "/$model_name/db/")

    # from forecasts_${window_type}_merged.sqlite3 get the params
    forecast_db_path = string(db_folder, "forecasts_", window_type, "_merged.sqlite3")

    # if file exists else return all_params
    if !isfile(forecast_db_path)
        println("Forecast database not found at $forecast_db_path; returning default parameters.")
        return all_params
    end

    db = SQLite.DB(forecast_db_path)
    rows = DBInterface.execute(db, "SELECT params FROM forecasts WHERE task_id = ?;", (task_id,))

   
    try
        params = nothing
        for row in rows
            params = _deser(row[1])
            # expand dimension of params
            params = reshape(params, size(params, 1), 1)

        end
        all_params[:,1] .= params

    catch e
        println("Error reading static params for task $task_id: $e")
        SQLite.close(db)
        rethrow(e)
    end
    SQLite.close(db)

    return all_params
end

function init_task_database(db_path::String, a::Int, b::Int; max_attempts::Int=5)
    for i in 1:max_attempts
        try 
            if i == 1
                db = SQLite.DB(db_path)
                DBInterface.execute(db, "PRAGMA journal_mode=WAL;")
                DBInterface.execute(db, "PRAGMA synchronous=NORMAL;")
                DBInterface.execute(db, "PRAGMA temp_store=MEMORY;")
                DBInterface.execute(db, "PRAGMA busy_timeout=10000;")
                DBInterface.execute(db, "CREATE TABLE IF NOT EXISTS tasks(id INTEGER PRIMARY KEY, state INTEGER NOT NULL);")
                DBInterface.execute(db, """
                    WITH RECURSIVE seq(x) AS (
                        SELECT ?1
                        UNION ALL
                        SELECT x+1 FROM seq WHERE x < ?2
                    )
                    INSERT OR IGNORE INTO tasks(id, state)
                    SELECT x, 0 FROM seq;
                """, (min(a,b), max(a,b)))
                SQLite.close(db)
            else
                # try to read task with id a
                db = SQLite.DB(db_path)
                DBInterface.execute(db, "SELECT state FROM tasks WHERE id = ?;", (a,))
                SQLite.close(db)
            end
            break
        catch
            println("Attempt $i to initialize task database failed; retrying...")
            SQLite.close(db)
            sleep(5.0)  # exponential backoff
        end
    end
    db = SQLite.DB(db_path)
    return db
end

const _CLAIM = "UPDATE tasks SET state = 1 WHERE id = ?1 AND state = 0 RETURNING id;"

function claim_task(db::SQLite.DB, id::Integer)
    try 
        result = DBInterface.execute(db, _CLAIM, (id,))
        for row in result
            return 0  # Successfully claimed
        end
    catch e 
        return nothing 
    end
    return nothing  # Not found or already claimed
end

function init_proc_counter_db(db_path::String; max_attempts::Int=5)
    for i in 1:max_attempts
        try
            if i == 1
                db = SQLite.DB(db_path)
                DBInterface.execute(db, "PRAGMA journal_mode=WAL;")
                DBInterface.execute(db, "PRAGMA synchronous=NORMAL;")
                DBInterface.execute(db, "PRAGMA busy_timeout=10000;")
                DBInterface.execute(db, "PRAGMA temp_store=MEMORY;")
                DBInterface.execute(db, """
                    CREATE TABLE IF NOT EXISTS proc_counter (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        n  INTEGER NOT NULL
                    );
                """)
                DBInterface.execute(db, """
                    INSERT INTO proc_counter(id, n) VALUES (1, 0)
                    ON CONFLICT(id) DO NOTHING;
                """)
                SQLite.close(db)
            else
                # try to read proc_counter
                db = SQLite.DB(db_path)
                DBInterface.execute(db, "SELECT n FROM proc_counter WHERE id = 1;")
                SQLite.close(db)
            end
            break
        catch
            println("Attempt $i to initialize proc_counter database failed; retrying...")
            SQLite.close(db)
            sleep(5.0)  # exponential backoff
        end
    end
    db = SQLite.DB(db_path)

    return db
end

const _INC_RETURNING = "UPDATE proc_counter SET n = n + 1 WHERE id = 1 RETURNING n;"
const _DEC_RETURNING = "UPDATE proc_counter SET n = MAX(n - 1, 0) WHERE id = 1 RETURNING n;"

function incr_working!(db::SQLite.DB; max_attempts::Int=10)::Int
    for i in 1:max_attempts
        try
            for row in DBInterface.execute(db, _INC_RETURNING)
                return row[1]::Int
            end
        catch e
            println("Attempt $i to increment working counter failed: $e")
            sleep(0.2 * i + rand())  # exponential backoff
        end
    end
    error("Counter increment failed")
end

function decr_working!(db::SQLite.DB; max_attempts::Int=10)::Int
    for i in 1:max_attempts
        try
            for row in DBInterface.execute(db, _DEC_RETURNING)
                return row[1]::Int
            end
        catch e
            println("Attempt $i to decrement working counter failed: $e")
            sleep(0.2*i + rand())  # exponential backoff
        end
    end
    error("Counter decrement failed")
end

ser(x) = (io=IOBuffer(); serialize(io,x); take!(io))
const _DB_INIT_LOCKS = Dict{String, ReentrantLock}()
const _DB_INIT_LOCK = ReentrantLock()
function init_forecast_db(path::AbstractString)

    mkpath(dirname(path))
    
    # Get or create a lock for this specific DB path
    lock(_DB_INIT_LOCK) do
        if !haskey(_DB_INIT_LOCKS, path)
            _DB_INIT_LOCKS[path] = ReentrantLock()
        end
    end
    
    # Only one thread can initialize this DB at a time
    lock(_DB_INIT_LOCKS[path]) do
        newfile = !isfile(path)
        db = SQLite.DB(path)
        
        DBInterface.execute(db, "PRAGMA busy_timeout=10000;")
        DBInterface.execute(db, "PRAGMA temp_store=MEMORY;")
        
        if newfile
            mode = lowercase(String(first(DBInterface.execute(db, "PRAGMA journal_mode=WAL;"))[1]))
            if mode != "wal"
                @warn "WAL not enabled; got '$mode'. Using DELETE mode." path
                DBInterface.execute(db, "PRAGMA journal_mode=DELETE;")
            end
            DBInterface.execute(db, "PRAGMA synchronous=NORMAL;")
        end
        
        DBInterface.execute(db, """
            CREATE TABLE IF NOT EXISTS forecasts(
                model  TEXT NOT NULL,
                thread TEXT NOT NULL,
                window TEXT NOT NULL,
                task_id INTEGER NOT NULL,
                loss   REAL,
                params BLOB NOT NULL,
                preds  BLOB NOT NULL,
                fl1    BLOB NOT NULL,
                fl2    BLOB NOT NULL,
                factors BLOB NOT NULL,
                states  BLOB NOT NULL,
                PRIMARY KEY(model,thread,window,task_id)
            );
        """)
        return db
    end
end

_forecast_path(base::String, k::Int) = k==0 ? base : replace(base, ".sqlite3" => "_$k.sqlite3")

function save_oos_forecast_sharded!(base::String, model, thread::AbstractString, window::AbstractString,
                                    task_id::Int, results, loss::Real, params; forecast_horizon::Int, max_shards::Int=16)

    #params = round.(params, digits=6)
    results.preds .= round.(results.preds, digits=3)
    results.factor_loadings_1 .= round.(results.factor_loadings_1, digits=3)
    results.factor_loadings_2 .= round.(results.factor_loadings_2, digits=3)
    results.factors .= round.(results.factors, digits=3)
    results.states  .= round.(results.states,  digits=3)
    
    p = results.preds[:, end-forecast_horizon+1:end]
    f = results.factors[:, end-forecast_horizon+1:end]
    s = results.states[:, end-forecast_horizon+1:end]
    fl1 = results.factor_loadings_1[:, end-forecast_horizon+1:end]
    fl2 = results.factor_loadings_2[:, end-forecast_horizon+1:end]
    
    path = _forecast_path(base, task_id)
    db = init_forecast_db(path); SQLite.close(db)

    path = _forecast_path(base, task_id)
    db = init_forecast_db(path)

    try
        DBInterface.execute(db, "BEGIN IMMEDIATE;")
        DBInterface.execute(db, """
            INSERT OR REPLACE INTO forecasts(
                model,thread,window,task_id,loss,params,preds,fl1,fl2,factors,states
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """, (
            model.base.model_string, thread, window, task_id, loss,
            ser(params),
            ser(p),
            ser(fl1),
            ser(fl2),
            ser(f),
            ser(s),
        ))
        DBInterface.execute(db, "COMMIT;")
        SQLite.close(db)
        return path
    catch e
        println("Error saving forecast shard for task $task_id: $e")
        try DBInterface.execute(db, "ROLLBACK;") catch end
        SQLite.close(db)
        rethrow(e)
    end
end

function merge_forecast_shards!(base::String; out::String=replace(base, ".sqlite3"=>"_merged.sqlite3"),
                                task_ids::AbstractVector, delete_shards::Bool=false)
    # helpers

    merge_task_id = task_ids[1]

    src_path = _forecast_path(base, merge_task_id)
    
    for task_id in task_ids[2:end]
        #println("Merging forecast shard for task $task_id into $out")
        src_db = SQLite.DB(src_path)
        

        new_entry_path = _forecast_path(base, task_id)
        if !isfile(new_entry_path)
            println("Shard for task $task_id not found at $new_entry_path; skipping.")
            continue
        end
        
        
        # open new entry db and print the names of the table that are there
        new_entry_db = SQLite.DB(new_entry_path)

        # print the model.base.model_string entry in the db to check whether it was stored correctly 
        for row in DBInterface.execute(new_entry_db,
            "SELECT model, thread, window, task_id, loss, params, preds, fl1, fl2, factors, states
            FROM forecasts WHERE task_id = ?;", (task_id,))
            DBInterface.execute(src_db, """
                INSERT OR REPLACE INTO forecasts(
                    model,thread,window,task_id,loss,params,preds,fl1,fl2,factors,states
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """, (row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]))
        end

        # # print number of rows in src_db after each merge
        # row_count = first(DBInterface.execute(src_db, "SELECT COUNT(*) FROM forecasts;"))
        # println("Number of rows in merged db after merging task $task_id: ", row_count[1])

        SQLite.close(new_entry_db)
        SQLite.close(src_db)
    end

    # # print number of rows in merged db
    # src_db = SQLite.DB(src_path)
    # row_count = first(DBInterface.execute(src_db, "SELECT COUNT(*) FROM forecasts;"))
    # println("Number of rows in merged db after merging: ", row_count[1])
    # SQLite.close(src_db)



    # # show params where task_id == 250 
    # src_db = SQLite.DB(src_path)
    # for row in DBInterface.execute(src_db, "SELECT task_id, params FROM forecasts WHERE task_id = ?;", (250,))
    #     println("Params for task_id 250: ", row)
    # end
    # SQLite.close(src_db)

    # Rename
    mv(src_path, out; force=true)
    if delete_shards
        for task_id in task_ids
            shard_path = _forecast_path(base, task_id)
            isfile(shard_path) && rm(shard_path)
        end
    end

  

    return out
end


# --- paths (assumes merged DB is: db/forecasts_<window>_merged.sqlite3) ---
_forecast_merged_path(model, window_type) =
    string(model.base.results_folder, "db/forecasts_", window_type, "_merged.sqlite3")

_legacy_forecast_csv_path(model, thread_Id, window_type) =
    string(model.base.results_folder, model.base.model_string, "__thread_id__", thread_Id, "__", window_type, "_window_forecasts.csv")

_legacy_fitted_params_csv_path(model, thread_Id, window_type) =
    string(model.base.results_folder, model.base.model_string,  "__thread_id__", thread_Id, "__", window_type, "_window_fitted_params.csv")

 _legacy_factors_csv_path(model, thread_Id, window_type) =
    string(model.base.results_folder, model.base.model_string, "__thread_id__", thread_Id, "__", window_type, "_window_factors.csv")

_legacy_states_csv_path(model, thread_Id, window_type) =
    string(model.base.results_folder, model.base.model_string, "__thread_id__", thread_Id, "__", window_type, "_window_states.csv")

_deser(x) = deserialize(IOBuffer(x))

"""
export_forecast_csv(model, thread_Id; window_type="expanding")

Reads preds from merged DB and writes legacy forecasts CSV:
rows = i, i+h, <K variables...>, sorted by col1 then col2.
"""
function export_forecast_csv(model,thread_Id::AbstractString, tasks, ; window_type::AbstractString="expanding")
    

    tmp = _legacy_forecast_csv_path(model, thread_Id, window_type) * "temp.csv"
    isfile(tmp) && rm(tmp; force=true)
    preds_blob = nothing
    for task_id in tasks

        db = SQLite.DB(_forecast_merged_path(model, window_type))
        row = DBInterface.execute(db, "SELECT task_id,  preds FROM forecasts WHERE task_id = ?;", (task_id,))
        task_id = nothing 
        for r in row 
            task_id = r[1]
            preds_blob = r[2]
        end
        #println(row)
        SQLite.close(db)
        

        P = _deser(preds_blob)              # K x H
        K, H = size(P)
        open(tmp, "a") do io
            for h in 1:H
                row = Vector{Float64}(undef, 2 + K)
                row[1] = float(task_id)
                row[2] = float(task_id + h)
                @views row[3:end] .= P[:, h]
                writedlm(io, [row], ',')
            end
        end
    end



    final = _legacy_forecast_csv_path(model, thread_Id, window_type)
    forecastingResults = readdlm(tmp, ',')
    try
        forecastingResults = convert.(Float64, forecastingResults)
    catch
        forecastingResults = convert.(Float32, forecastingResults)
    end
    forecastingResults = forecastingResults[sortperm(forecastingResults[:,2]), :]
    forecastingResults = forecastingResults[sortperm(forecastingResults[:,1]), :]
    writedlm(final, forecastingResults, ',')
    rm(tmp; force=true)
    return final
end

# --- extra paths for fl1 / fl2 ---
_legacy_fl1_csv_path(model, thread_Id, window_type) =
    string(model.base.results_folder, model.base.model_string,  "__thread_id__", thread_Id, "__", window_type, "_window_fl1.csv")

_legacy_fl2_csv_path(model, thread_Id, window_type) =
    string(model.base.results_folder, model.base.model_string, "__thread_id__", thread_Id, "__", window_type, "_window_fl2.csv")

# Helper to (re)load, coerce numeric, sort by first column, and write final CSV
function _finalize_simple_table(tmp_path::AbstractString, final_path::AbstractString)
    tbl = readdlm(tmp_path, ',')
    try
        tbl = convert.(Float64, tbl)
    catch
        tbl = convert.(Float32, tbl)
    end
    # sort by task_id (first column)
    tbl = tbl[sortperm(tbl[:, 1]), :]
    writedlm(final_path, tbl, ',')
    rm(tmp_path; force=true)
    return final_path
end

"""
export_fitted_params_csv(model, tasks; window_type="expanding")

Reads `params` from merged DB and writes a legacy params CSV:
rows = task_id, params...
Extraction mirrors export_forecast_csv: iterates over `tasks`, appends to a temp CSV, then sorts by task_id.
"""
function export_fitted_params_csv(model, thread_Id, tasks; window_type::AbstractString="expanding")
    tmp = _legacy_fitted_params_csv_path(model, thread_Id, window_type) * "temp.csv"
    isfile(tmp) && rm(tmp; force=true)

    for task_id in tasks
        db = SQLite.DB(_forecast_merged_path(model, window_type))
        row = DBInterface.execute(db, "SELECT task_id, params FROM forecasts WHERE task_id = ?;", (task_id,))
        got_task = nothing
        params_blob = nothing
        for r in row
            got_task   = r[1]
            params_blob = r[2]
        end
        SQLite.close(db)

        isnothing(params_blob) && continue
        p = vec(collect(_deser(params_blob)))     # make sure it's a 1-D vector

        open(tmp, "a") do io
            # first entry = task_id, then params...
            out = Vector{Float64}(undef, 1 + length(p))
            out[1] = float(got_task)
            @views out[2:end] .= Float64.(p)
            writedlm(io, [out], ',')
        end
    end

    final = _legacy_fitted_params_csv_path(model, thread_Id, window_type)
    isfile(final) && rm(final; force=true)
    return _finalize_simple_table(tmp, final)
end

"""
export_fl1_csv(model, tasks; window_type="expanding")

Reads `fl1` from merged DB and writes a legacy CSV:
rows = task_id, fl1...
Extraction mirrors export_forecast_csv.
"""
function export_fl1_csv(model, thread_Id, tasks; window_type::AbstractString="expanding")
    tmp = _legacy_fl1_csv_path(model, thread_Id, window_type) * "temp.csv"
    isfile(tmp) && rm(tmp; force=true)

    for task_id in tasks
        db = SQLite.DB(_forecast_merged_path(model, window_type))
        row = DBInterface.execute(db, "SELECT task_id, fl1 FROM forecasts WHERE task_id = ?;", (task_id,))
        got_task = nothing
        fl_blob = nothing
        for r in row
            got_task = r[1]
            fl_blob  = r[2]
        end
        SQLite.close(db)

        isnothing(fl_blob) && continue
        P = _deser(fl_blob)        # ensure 1-D vector

        K, H = size(P)
        open(tmp, "a") do io
            for h in 1:H
                row = Vector{Float64}(undef, 2 + K)
                row[1] = float(task_id)
                row[2] = float(task_id + h)
                @views row[3:end] .= P[:, h]
                writedlm(io, [row], ',')
            end
        end
    end

    final = _legacy_fl1_csv_path(model, thread_Id, window_type)
    isfile(final) && rm(final; force=true)
    return _finalize_simple_table(tmp, final)
end

"""
export_fl2_csv(model, tasks; window_type="expanding")

Reads `fl2` from merged DB and writes a legacy CSV:
rows = task_id, fl2...
Extraction mirrors export_forecast_csv.
"""
function export_fl2_csv(model, thread_Id, tasks; window_type::AbstractString="expanding")
    tmp = _legacy_fl2_csv_path(model, thread_Id, window_type) * "temp.csv"
    isfile(tmp) && rm(tmp; force=true)

    for task_id in tasks
        db = SQLite.DB(_forecast_merged_path(model, window_type))
        row = DBInterface.execute(db, "SELECT task_id, fl2 FROM forecasts WHERE task_id = ?;", (task_id,))
        got_task = nothing
        fl_blob = nothing
        for r in row
            got_task = r[1]
            fl_blob  = r[2]
        end
        SQLite.close(db)


        isnothing(fl_blob) && continue
        P = _deser(fl_blob)        # ensure 1-D vector

        K, H = size(P)
        open(tmp, "a") do io
            for h in 1:H
                row = Vector{Float64}(undef, 2 + K)
                row[1] = float(task_id)
                row[2] = float(task_id + h)
                @views row[3:end] .= P[:, h]
                writedlm(io, [row], ',')
            end
        end
    end

    final = _legacy_fl2_csv_path(model, thread_Id, window_type)
    isfile(final) && rm(final; force=true)
    return _finalize_simple_table(tmp, final)
end

function _append_array_rows!(io, task_id::Real, A)
    if A isa AbstractVector
        out = Vector{Float64}(undef, 1 + length(A))
        out[1] = float(task_id)
        @views out[2:end] .= Float64.(A)
        writedlm(io, [out], ',')
    else
        K, H = size(A)
        for h in 1:H
            row = Vector{Float64}(undef, 2 + K)
            row[1] = float(task_id)
            row[2] = float(task_id + h)
            @views row[3:end] .= Float64.(A[:, h])
            writedlm(io, [row], ',')
        end
    end
end

function export_factors_csv(model, thread_Id, tasks; window_type::AbstractString="expanding")
    tmp = _legacy_factors_csv_path(model, thread_Id, window_type) * "temp.csv"
    isfile(tmp) && rm(tmp; force=true)

    for task_id in tasks
        db = SQLite.DB(_forecast_merged_path(model, window_type))
        row = DBInterface.execute(db, "SELECT task_id, factors FROM forecasts WHERE task_id = ?;", (task_id,))
        got_task, blob = nothing, nothing
        for r in row
            got_task = r[1]; blob = r[2]
        end
        SQLite.close(db)
        isnothing(blob) && continue
        A = _deser(blob)

        open(tmp, "a") do io
            _append_array_rows!(io, got_task, A)
        end
    end

    final = _legacy_factors_csv_path(model, thread_Id, window_type)
    isfile(final) && rm(final; force=true)
    return _finalize_simple_table(tmp, final)
end

function export_states_csv(model, thread_Id, tasks; window_type::AbstractString="expanding")
    tmp = _legacy_states_csv_path(model, thread_Id, window_type) * "temp.csv"
    isfile(tmp) && rm(tmp; force=true)

    for task_id in tasks
        db = SQLite.DB(_forecast_merged_path(model, window_type))
        row = DBInterface.execute(db, "SELECT task_id, states FROM forecasts WHERE task_id = ?;", (task_id,))
        got_task, blob = nothing, nothing
        for r in row
            got_task = r[1]; blob = r[2]
        end
        SQLite.close(db)
        isnothing(blob) && continue
        A = _deser(blob)

        open(tmp, "a") do io
            _append_array_rows!(io, got_task, A)
        end
    end

    final = _legacy_states_csv_path(model, thread_Id, window_type)
    isfile(final) && rm(final; force=true)
    return _finalize_simple_table(tmp, final)
end

# Optional: convenience bundler that leaves export_forecast_csv untouched
export_all_csv(model, thread_Id, tasks; window_type="expanding") = (
    forecasts      = export_forecast_csv(model, thread_Id, tasks; window_type=window_type),
    fitted_params  = export_fitted_params_csv(model, thread_Id, tasks; window_type=window_type),
    fl1            = export_fl1_csv(model, thread_Id, tasks; window_type=window_type),
    fl2            = export_fl2_csv(model, thread_Id, tasks; window_type=window_type),
    factors        = export_factors_csv(model, thread_Id, tasks; window_type=window_type),  # NEW
    states         = export_states_csv(model, thread_Id, tasks; window_type=window_type),   # NEW
)

