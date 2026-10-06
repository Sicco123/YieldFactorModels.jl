module YieldFactorModels
    # ============================================================================
    # Dependencies
    # ============================================================================
    using LinearAlgebra
    using Optim
    using Statistics
    using DelimitedFiles
    using Printf
    using Lux
    using LineSearches
    using Zygote
    using ForwardDiff
    using DiffResults
    using Distributed
    using Base.Threads 
    using StaticArrays
    using Random 
    using SQLite
    using DBInterface
    using Serialization

    # ============================================================================
    # Include Files
    # ============================================================================
    # Model files
    include("models/msedriven/msebasemodel.jl")
    include("models/parameteroperations.jl")
    include("models/msedriven/mseneural.jl")
    include("models/msedriven/mselambda.jl")
    include("models/msedriven/paramteroperations.jl")
    include("models/static/staticbasemodel.jl")
    include("models/static/staticneural.jl")
    include("models/static/staticlambda.jl")
    include("models/static/randomwalk.jl")
    include("models/static/paramteroperations.jl")
    include("models/kalman/kalmanbasemodel.jl")
    include("models/kalman/dns.jl")
    include("models/kalman/filter.jl")
    include("models/kalman/paramoperations.jl")
    include("models/filter.jl")
    include("models/kalman/tvλdns.jl")
    include("models/kalman/afns.jl")

    # Utility files
    include("utils/transformations.jl")
    include("utils/neural_network_transform.jl")
    include("utils/data_management.jl")
    
    # db files 
    include("databaseoperations/databaseoperations.jl")

    # Core functionality files
    include("io.jl")
    include("optimization.jl")
    include("forecasting.jl")
    include("model_dictionary.jl")
    # ============================================================================
    # Exports
    # ============================================================================
    # Main types
    export AbstractYieldFactorModel, AbstractMSEDrivenModel, AbstractStaticModel
    export AbstractNeuralMSEDrivenModel, AbstractλMSEDrivenModel
    export MSEDrivenBaseModel, MSEDNeuralModel, MSEDλModel
    export StaticBaseModel, StaticNeuralModel, StaticλModel

    # Parameter operations
    export get_params!, set_params!
    export initialize_with_static_params!, get_new_initial_params
    export expand_params, get_unique_params

    # Factor loadings
    export update_factor_loadings!
    
    # Core functions
    export run, save_results, load_data, predict, get_loss
    export init_params_path
    export estimate!, estimate_steps!#, run_rolling_forecasts

    # ============================================================================
    # Helper Functions
    # ============================================================================
    
    """
        setup_data_paths(model_type, simulation, scratch_dir)
    
    Configure data folder and results location paths based on simulation settings.
    """
    function setup_data_paths( model_type::String, simulation::Bool, scratch_dir::String, thread_id::String)
        if simulation
            data_folder = "$(scratch_dir)YieldFactorModels.jl/data_simulation/"
            results_location = "$(scratch_dir)YieldFactorModels.jl/results_simulation/thread_id__$(thread_id)/"
           
        else
            data_folder = "$(scratch_dir)YieldFactorModels.jl/data/"
            results_location = "$(scratch_dir)YieldFactorModels.jl/results/thread_id__$(thread_id)/"
            
        end
        return data_folder, results_location
    end

    

    """
        load_static_parameters!(model, model_type, results_location, thread_id)
    
    Load and initialize static parameters for neural network and NS models.
    """
    function load_static_parameters!(model, model_type::String, results_location::String, thread_id::String, params)
        static_model_name = static_counterpart(model)
        try 
            
            static_params = readdlm(
                string(results_location,  static_model_name, "/", 
                        static_model_name, "__thread_id__", thread_id, "__out_params.csv"), 
                ','
            )
            params = initialize_with_static_params(model, params, static_params)
        catch err
            @warn "Static parameters for $(model_type) not found, using default initialization."
        end
        return params        
    end
    
    """
        load_initial_parameters!(model, model_type, float_type, thread_id; simulation=false)
    
    Load initial parameters for the specified model type, preferring this thread's own
    file over the shared default, generating random parameters and saving to disk if
    neither is found.
    """


    """
        init_params_path(init_folder, model_type, thread_id; simulation=false)

    The init file to read: a thread-specific one when it exists, otherwise the shared
    default `init_params_<model_type>.csv`. Thread files are named the same way results
    are (`init_params_1C__thread_id__7.csv`) and are written by
    `simulation_exp_1/long_path_initializations.jl`, one per thread, so a per-thread
    re-initialization never overwrites the default other threads still read.

    When no candidate exists the plain default path is returned, which is what
    `load_initial_parameters!` then writes its random start to.
    """
    function init_params_path(init_folder::String, model_type::String, thread_id::String;
                              simulation::Bool=false)
        # A diagonal model reads init_params_<type>-diag__thread_id__<id>.csv, and falls
        # back to the unrestricted model's file when it has not been re-initialized yet:
        # the off-diagonal Φ slots in it are pinned to zero on load anyway.
        names = model_type == base_model_name(model_type) ? [model_type] :
                                                            [model_type, base_model_name(model_type)]
        candidates = String[]
        for nm in names
            base = string(init_folder, "init_params_", nm)
            append!(candidates, simulation ?
                [string(base, "__thread_id__", thread_id, "_simulation.csv"),
                 string(base, "__thread_id__", thread_id, ".csv"),
                 string(base, "_simulation.csv"),
                 string(base, ".csv")] :
                [string(base, "__thread_id__", thread_id, ".csv"),
                 string(base, ".csv")])
        end
        i = findfirst(isfile, candidates)
        return isnothing(i) ? last(candidates) : candidates[i]
    end

    function load_initial_parameters!(model, model_type::String, float_type::Type,
                                      thread_id::String; simulation=false)
        path = init_params_path(model.base.init_folder, model_type, thread_id; simulation=simulation)
        all_params = nothing
        try
            all_params = readdlm(path, ',')
            println("Initial parameters read from: $path")
        catch err
            @warn ("Initial parameters for $(model_type) not found at $path. Writing file with random initial parameters...")
            num_params = length(get_params(model))
            println("Number of parameters: $num_params")
            all_params = rand(float_type, num_params, 1)
            # mk dir if needed 
            isdir(model.base.init_folder) || mkpath(model.base.init_folder)
            writedlm(path, all_params, ',')
        end
        # Resize stored params to this thread's N where the layout allows it.
        all_params = reconcile_init_params(model, all_params)
        return all_params
    end
    """
        run_estimation!(model, data, in_sample_end, all_params, param_groups, 
                       max_group_iters, group_tol)
    
    Run model estimation with either grouped or standard optimization.
    """
    function run_estimation!(model, data::Matrix, in_sample_end::Int, all_params::Matrix,
                            param_groups::Vector{String}, max_group_iters::Int, 
                            group_tol::Real; printing::Bool=true)
        if !isempty(param_groups)
            @assert size(all_params, 1) == length(param_groups) ||
                    error("length(param_groups) = $(length(param_groups)) must equal number of rows in all_params = $(size(all_params, 1))")
            init_params, loss, params, ir = estimate_steps!(
                model,
                data[:, 1:in_sample_end], 
                all_params,
                param_groups;
                max_group_iters = max_group_iters,
                tol = group_tol,
                printing = printing
            )
        else
            init_params, loss, params, ir = estimate!(
                model, 
                data[:, 1:in_sample_end], 
                all_params;
                printing = printing
            )
        end
        return init_params, loss, params, ir
    end

    # ============================================================================
    # Main Run Function
    # ============================================================================
    
    """
        run(thread_id, in_sample_end, forecast_horizon, run_rolling, model_type, float_type; 
            num_procs, window_type, in_sample_start, param_groups, max_group_iters, 
            group_tol, run_optimization, save_results_bool, simulation, reestimate, scratch_dir)
    
    Main function to run yield factor model estimation, filtering, and forecasting.
    
    # Arguments
    - `thread_id::String="1"`: Thread identifier for parallel processing
    - `in_sample_end::Int=100`: End of in-sample period
    - `forecast_horizon::Int=12`: Number of periods to forecast ahead
    - `run_rolling::Bool=true`: Whether to run rolling window forecasts
    - `model_type::String="1C"`: Type of model to estimate
    - `float_type::Type=Float32`: Floating point precision
    - `num_procs::Int=1`: Number of processes for parallel computing
    - `window_type::String="both"`: Type of rolling window ("expanding", "rolling", or "both")
    - `in_sample_start::Int=1`: Start of in-sample period
    - `param_groups::Vector{String}=String[]`: Parameter groups for grouped estimation
    - `max_group_iters::Int=10`: Maximum iterations for grouped estimation
    - `group_tol::Real=1e-8`: Tolerance for grouped estimation convergence
    - `run_optimization::Bool=true`: Whether to run parameter optimization
    - `save_results_bool::Bool=true`: Whether to save results to disk
    - `simulation::Bool=false`: Whether running on simulated data
    - `reestimate::Bool=true`: Whether to re-estimate in rolling forecasts
    - `scratch_dir::String="/scratch-shared/skooiker/YieldFactorModels.jl/"`: Scratch directory for cluster computing

    # Returns
    - `model`: The estimated model object
    """
    function run(
        thread_id::String="1",
        in_sample_end::Int=100,
        forecast_horizon::Int=12,
        run_rolling::Bool=true,
        model_type::String="1C",
        float_type::Type=Float32;
        window_type::String="both",
        in_sample_start::Int=1,
        param_groups::Vector{String}=String[],      
        max_group_iters::Int=10,              
        group_tol::Real=1e-8,
        run_optimization::Bool=true,
        save_results_bool::Bool=true, 
        simulation::Bool=false, 
        reestimate::Bool=true,
        scratch_dir::String="",
        seed::Int=43,
        analytic_score::Bool=true,
        diagonal::Bool=false,
        static_seed::Bool=true
    )

        # Closed-form neural score: ON by default (~2x faster). Set analytic_score=false
        # to use ForwardDiff. Only affects anchored neural models (transform_bool=true);
        # a no-op otherwise.
        ANALYTIC_SCORE[] = analytic_score

        # false: score-driven models keep their own initialization instead of inheriting
        # the static counterpart's per-window fit. See STATIC_SEED.
        STATIC_SEED[] = static_seed

        if simulation
            window_type = "simulation"
            run_optimization = false
            run_rolling = true
            save_results_bool = false
        end 

        Random.seed!(seed)
        # ========================================================================
        # Setup paths and load data
        # ========================================================================
        data_folder, results_location = setup_data_paths( model_type, simulation, scratch_dir, thread_id)
        data, maturities = load_data(data_folder, thread_id)
        data = convert(Matrix{float_type}, data)
        maturities = convert(Vector{float_type}, maturities)

        if model_type == "AFNS" || model_type == "33"
            data ./= float_type(100)
            maturities ./= float_type(12)
        end

        # ========================================================================
        # Initialize model
        # ========================================================================
        N = length(maturities)  # Number of maturities
        M = 3                    # Number of factors
       
        model, model_type = create_model(model_type, maturities, N, M, float_type,
                                         "$results_location$(model_type)$(diagonal ? "-diag" : "")/";
                                         diagonal=diagonal)
 
        # ========================================================================
        # Load and set parameters
        # ========================================================================
        param_groups = get_param_groups(model, param_groups)
        all_params = load_initial_parameters!(model, model_type, float_type, thread_id; simulation=simulation)
        set_params!(model, all_params[:, 1])
        

        # Load static parameters if applicable
        if STATIC_SEED[]
            all_params[:,1] = load_static_parameters!(model, model_type, results_location, thread_id, all_params[:,1])
        end

        # Convert parameters to appropriate float type
        all_params = convert(Matrix{float_type}, all_params)
   
    
        # ========================================================================
        # Run optimization
        # ========================================================================
        if run_optimization
            println("The param groups are : ", param_groups)
            init_params, loss, params, ir = run_estimation!(
                model, data, in_sample_end, all_params, param_groups, 
                max_group_iters, group_tol; printing = true
            )
        else  
            init_params = all_params[:, 1]
            params = all_params[:, 1]
            loss = 0.0
            ir = 0.0
        end

        # ========================================================================
        # Compute in-sample loss
        # ========================================================================
 

        # ========================================================================
        # Save results
        # ========================================================================
        if save_results_bool
            # In-sample results
            # set_params! FIRST: estimate! leaves the model at the last point the
            # optimizer evaluated, not at the returned optimum, so filtering before
            # this line wrote *_filtered_insample.csv at the wrong parameters.
            set_params!(model, params)
            results = predict(model, data[:, 1:in_sample_end])

            save_results(model, results, loss, thread_id, "insample")

            loss = get_loss(model, data[:, 1:in_sample_end])
            println("In-sample loss: $loss")

            # Out-of-sample filtering
            results = predict(model, data[:, 1:end])
            save_results(model, results, loss, thread_id, "outofsample")
            
            loss_array = get_loss_array(model, data[:, 1:end]; K = 1)
            out_of_sample_loss_array = loss_array[in_sample_end+1:end]

            # ponytail: in_sample_end == T (a full-sample fit) leaves nothing out of sample
            if !isempty(out_of_sample_loss_array)
            # print first sum of 10 percent of out-of-sample losses, 25%, 50% of the sample and then 75 percent and 100 percent
            println("Out-of-sample loss array (first 10%): ", mean(out_of_sample_loss_array[1:Int(floor(0.1*length(out_of_sample_loss_array)))]))
            println("Out-of-sample loss array (first 25%): ", mean(out_of_sample_loss_array[1:Int(floor(0.25*length(out_of_sample_loss_array)))]))
            println("Out-of-sample loss array (first 50%): ", mean(out_of_sample_loss_array[1:Int(floor(0.5*length(out_of_sample_loss_array)))]))
            println("Out-of-sample loss array (first 75%): ", mean(out_of_sample_loss_array[1:Int(floor(0.75*length(out_of_sample_loss_array)))]))
            println("Out-of-sample loss array (full): ", mean(out_of_sample_loss_array))
            end

        end

        # ========================================================================
        # Rolling window forecasts
        # ========================================================================
        if run_rolling
            println("Forecasting...")
            run_rolling_forecasts(
                model, data, thread_id, in_sample_end, in_sample_start, forecast_horizon, 
                all_params; 
                window_type=window_type, 
                param_groups=param_groups, 
                max_group_iters=max_group_iters, 
                group_tol=group_tol, 
                reestimate=reestimate
            )
        end
        
        return model
    end
end
