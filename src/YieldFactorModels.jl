module YieldFactorModels
    # ============================================================================
    # Dependencies
    # ============================================================================
    using LinearAlgebra
    using Optim
    using Statistics
    using DelimitedFiles
    using Printf
    using Flux
    using LineSearches
    using Flux: destructure
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
    include("models/filter.jl")

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
        static_model_name = get_static_model_type(model)
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
        load_initial_parameters!(model, model_type)
    
    Load initial parameters for the specified model type, generating random
    parameters and saving to disk if not found.
    """


    function load_initial_parameters!(model, model_type::String, float_type::Type)
        all_params = nothing
        try
            all_params = readdlm(string(model.base.init_folder, "init_params_", model_type, ".csv"), ',')
        catch err
            @warn ("Initial parameters for $(model_type) not found in $(model.base.init_folder). Writing file with random initial parameters...")
            num_params = length(get_params(model))
            println("Number of parameters: $num_params")
            all_params = rand(float_type, num_params, 1)
            # mk dir if needed 
            isdir(model.base.init_folder) || mkpath(model.base.init_folder)
            writedlm(string(model.base.init_folder, "init_params_", model_type, ".csv"), all_params, ',')
        end
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
        seed::Int=43
    )

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

        # ========================================================================
        # Initialize model
        # ========================================================================
        N = length(maturities)  # Number of maturities
        M = 3                    # Number of factors
       
        model, model_type = create_model(model_type, maturities, N, M, float_type, "$results_location$(model_type)/" )
 
        # ========================================================================
        # Load and set parameters
        # ========================================================================
        param_groups = get_param_groups(model, param_groups)
        all_params = load_initial_parameters!(model, model_type, float_type)
        set_params!(model, all_params[:, 1])
        

        # Load static parameters if applicable
        all_params[:,1] = load_static_parameters!(model, model_type, results_location, thread_id, all_params[:,1])

        # Convert parameters to appropriate float type
        all_params = convert(Matrix{float_type}, all_params)
    
        # ========================================================================
        # Run optimization
        # ========================================================================
        if run_optimization
            println("The param groups are : ", param_groups)
            init_params, loss, params, ir = run_estimation!(
                model, data, in_sample_end, all_params, param_groups, 
                max_group_iters, group_tol
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
        set_params!(model, params)
        loss = get_loss(model, data[:, 1:in_sample_end])
        println("In-sample loss: $loss")

        # ========================================================================
        # Save results
        # ========================================================================
        if save_results_bool
            # In-sample results
            results = predict(model, data[:, 1:in_sample_end])
            save_results(model, results, loss, thread_id, "insample")

            # Out-of-sample filtering
            results = predict(model, data[:, 1:end])
            save_results(model, results, loss, thread_id, "outofsample")
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
