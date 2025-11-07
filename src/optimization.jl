"""
    compute_loss(model::AbstractYieldFactorModel, data, params)

Compute the loss for optimization.
Transforms parameters, updates model, and returns scaled negative loss.

# Returns
- Negative ll (for minimization)
"""
function compute_loss(model::AbstractYieldFactorModel, data, params)
    # Transform parameters to constrained space
    transformed_params = transform_params(model, params)
    
    #temp_model = get_temp_model(model, transformed_params)
    set_params!(model, transformed_params)
   
    # Compute loss via state space filter
    loss= get_loss(model, data)

    # Return negative for minimization (optimizer minimizes)
    return -loss
end



"""
    try_initializations(best_params, model::AbstractYieldFactorModel)

Default initialization strategy - returns parameters unchanged.
Override this method for specific model types requiring custom initialization.
"""
function try_initializations(best_params, model::AbstractYieldFactorModel, data)
    return best_params
end

function try_initializations(best_params, model::AbstractStaticModel, data; max_tries=1)
    # stack all new intializations in matrix 
    all_params = zeros(eltype(best_params), length(best_params), max_tries + 1)
    all_params[:, 1] = best_params
    for trial in 1:max_tries
        all_params[:, trial+1] = get_new_initial_params(model, best_params, trial)
    end
    return all_params
end

function try_initializations(best_params, model::AbstractRandomWalkModel, data; max_tries=1)
    # stack all new intializations in matrix 
    all_params = zeros(eltype(best_params), length(best_params), 1)
    all_params[:, 1] = best_params
    return all_params
end



"""
    try_initializations(best_params, model::AbstractMSEDrivenModel, data)

Attempt multiple random initializations for MSE-driven models.
Tries 1000 random parameter configurations and returns the best one.

# Arguments
- `best_params`: Initial parameter vector
- `model`: MSE-driven model instance
- `data`: Observation data matrix

# Returns
- Best parameter vector found across all trials
"""
function try_initializations(best_params, model::AbstractMSEDrivenModel, data)
    # Set up model dimensions
    set_params!(model, best_params)
    T = size(data, 2)
    NT = length(data)
    
    # Compute initial loss
    init_loss= get_loss(model, data)
    init_loss *= -1  # Negate for maximization comparison
    
    trial_params = copy(best_params)

    # Try multiple random initializations
    for trial in 1:1000
        initial_p = get_new_initial_params(model, trial_params, trial)

        # Skip if no valid initialization generated
        if initial_p === nothing
            break
        end
        
        # Update model with trial parameters
        T = size(data, 2)
        NT = length(data)
        set_params!(model, initial_p)
        
        # Evaluate loss
        loss= get_loss(model, data)
        loss *= -1

 
        # Keep best parameters
        if loss < init_loss
            init_loss = loss
            best_params = copy(initial_p)
        end
    end

    # expand best_params 
    best_params = reshape(best_params, (length(best_params), 1))
    return best_params
end 

"""
    estimate_steps!(model, data, all_params, param_groups; kwargs...)

Estimate model parameters using block-coordinate descent.
Optimizes parameter groups sequentially, allowing different optimizers per group.

# Arguments
- `model`: Model instance
- `data`: Observation data matrix
- `all_params`: Matrix of initial parameters (each column is a starting point)
- `param_groups`: Vector mapping each parameter to a group ID

# Keyword Arguments
- `max_group_iters`: Maximum iterations of group-wise optimization (default: 10)
- `tol`: Convergence tolerance for ll change (default: 1e-8)
- `opts`: Custom optimizer options (optional)
- `optimizers`: Custom optimizer instances (optional)

# Returns
- Tuple of (initial_params, ll, best_params, iteration_count)
"""
function estimate_steps!(
    model::AbstractYieldFactorModel,
    data::Matrix{T},
    all_params::Matrix{T},
    param_groups::Vector{String};
    max_group_iters::Int = 10,
    tol::Real = 1e-8,
    opts = nothing,
    optimizers = nothing,
    printing::Bool=true
) where T<:Real

   
    group_ids = sort(collect(Set(param_groups)))
    
    # Try improved initializations for first starting point
    all_params = try_initializations(all_params[:, 1], model, data)
    n_starts = size(all_params, 2)
 
    # Transform to unconstrained space for optimization
    for j in 1:n_starts
        all_params[:, j:j] = untransform_params(model, all_params[:, j:j])
        # santize # Clean invalid values
        all_params[:, j:j] = _sanitize_parameters(all_params[:, j:j], T)

    end
    
    # Create loss function closure
    loss_wrapper(p) = compute_loss(model, data, p)
   


    # Validate initial parameters
    ll = -loss_wrapper(all_params[:, 1])

    for i in 1:10
        if !isfinite(ll) || isnan(ll)
            # Perturb parameters slightly to find valid starting point
            all_params[:, 1] .*= T(0.95)
            ll = -loss_wrapper(all_params[:, 1])
        else
            if printing
                println("✓ Found valid initial parameters after $(i-1) perturbations")
            end
            break
        end
    end



    # Storage for results from each starting point
    all_results = Vector{Tuple{Vector{T}, T, Vector{T}, Int}}(undef, n_starts)
    best_ll_overall = -Inf
    best_j = 0
    
  
    # Set up optimizer configurations
    optimizer_dict = _create_optimizer_dict(opts, optimizers, T; printing=printing)


    if printing
        println("\n" * "="^60)
        println("Starting block-coordinate optimization")
        println("="^60)
    end 

    best_j = 1
    for j in 1:n_starts
        if printing
            println("\n--- Starting point $j/$n_starts ---")
        end

        p = copy(all_params[:, j])
        # make a trainer params that is vector with data types Real
   

        prev_ll = -Inf
        abort_group = false
    
        # Iteratively optimize each parameter group
        for iter in 1:max_group_iters
            for g in group_ids
                # Skip placeholder group
                if g == "-1"
                    continue
                end
                
                # Get optimizer configuration for this group
                optimizer, opt, _ = _get_optimizer_for_group(g, optimizer_dict)
                
                # Find parameters belonging to this group
                inds = findall(param_groups .== g)
             
                # Create objective for this parameter block
                subobj = x_sub -> begin
                    p_temp = similar(x_sub, length(p))  # Dual during AD, Float64 otherwise
                    p_temp .= p                         # broadcasts p into Dual if needed
                    p_temp[inds] .= x_sub               # overwrite the active block
                    
                    loss_wrapper(p_temp)
                end
    
                x0 = p[inds]

           
                # Optimize this parameter block
                try
                    # without autodiff
                    res = optimize(subobj, x0, optimizer, opt)
                 
                    p[inds] .= Optim.minimizer(res)
                catch e
                    println("  ⚠️  Error optimizing group $g on iter $iter: $e")
                    if iter == 1
                        rethrow(e)  # No fallback on first iteration
                    end
                    println("  Using last parameters & skipping remaining iterations")
                    abort_group = true
                    break
                end

                # garbage collect 
                GC.gc()
            end
            
            # Exit if error occurred
            if abort_group
                break
            end
            
            # Check convergence
            ll = -loss_wrapper(p)
            Δll = ll - prev_ll
            
            if abs(Δll) < tol
                if printing
                    println("  ✓ Converged after $iter iterations (ΔLL = $Δll)")
                end
                prev_ll = ll
                break
            end
            prev_ll = ll

        end
        
 
        # Store results
        all_results[j] = (all_params[:, j], prev_ll, copy(p), 0)
        
        if prev_ll > best_ll_overall
            best_ll_overall = prev_ll
            best_j = j
        end
        if printing
            println("✓ LL = $prev_ll from start $j")
        end
        
    end
    
    # Extract best results
    init_p, ll, best_p, ir = all_results[best_j]
    
    # Transform back to constrained space
    best_p = transform_params(model, best_p)
    init_p = transform_params(model, init_p)


    if printing
        println("\n" * "="^60)
        println("✓ Best overall LL = $ll from start $best_j")
        println("="^60 * "\n")
    end

    return init_p, ll, best_p, ir
end


"""
    estimate!(model::AbstractYieldFactorModel, data, all_params)

Main estimation function using LBFGS optimization.
Tries multiple starting points and returns the best solution.

# Arguments
- `model`: DNS model instance
- `data`: Observation data matrix
- `all_params`: Matrix of initial parameters (each column is a starting point)

# Returns
- Tuple of (initial_params, ll, best_params, iteration_count)
"""
function estimate!(model::AbstractYieldFactorModel, data, all_params::Matrix{T}; printing::Bool=true) where T <: Real
    n_starts = size(all_params, 2)
    all_params_results = Vector{Any}(undef, n_starts)
    best_j = 0
    best_ll = -Inf
    
    # Configure optimizer
    opt = Optim.Options(
        iterations = 1000,
        g_tol = 1e-6,
        f_abstol = 1e-6,
        show_trace = false,
        show_every = 1,
        store_trace = false,
        extended_trace = false,
        allow_f_increases = true
    )
    
    # Transform to unconstrained space
    all_params = untransform_params(model, all_params)
    
    # Create loss function
    loss_wrapper(p) = compute_loss(model, data, p)

    if printing
        println("\n" * "="^60)
        println("Starting optimization with $n_starts starting point(s)")
        println("="^60)
    end

    for j in 1:n_starts
        if printing
            println("\n--- Starting point $j/$n_starts ---")
        end

        params = all_params[:, j]
        
        # Optimize using LBFGS
        td = TwiceDifferentiable(loss_wrapper, params; autodiff=:forward)
        result = optimize(
            td, 
            params,
            Optim.LBFGS(linesearch = LineSearches.BackTracking(order=3)),
            opt
        )
        
        # Extract results
        ll = -Optim.minimum(result)
        params = Optim.minimizer(result)
        converged = Optim.converged(result)
        
        # Store results for this starting point
        all_params_results[j] = (all_params[:, j], ll, params, converged ? 0 : 1)
        
        # Track best solution
        if ll > best_ll
            best_ll = ll
            best_j = j
        end
        
        # Report progress
        status = converged ? "✓" : "⚠"
        if printing
            println("$status LL = $ll (converged: $converged)")
        end
    end
    
    # Extract best results
    init_params, ll, best_params, ir = all_params_results[best_j]
    
    # Transform back to constrained space
    best_params = transform_params(model, best_params)
    init_params = transform_params(model, init_params)

    if printing
        println("\n" * "="^60)
        println("✓ Best LL = $ll from starting point $best_j")
        println("="^60 * "\n")
    end

    return init_params, ll, best_params, ir
end


# ============================================================================
# Helper Functions
# ============================================================================

"""
    _sanitize_parameters(params, T)

Replace NaN and infinite values with zeros.
"""
function _sanitize_parameters(params::Matrix{T}, ::Type{T}) where T<:Real
    n_params, n_starts = size(params)
    for j in 1:n_starts
        for i in 1:n_params
            if !isfinite(params[i, j]) || isnan(params[i, j])
                params[i, j] = T(0.0)
            end
        end
    end
    return params
end

"""
    _create_optimizer_dict(opts, optimizers, T)

Create dictionary mapping parameter groups to optimizer configurations.
"""
function _create_optimizer_dict(opts, optimizers, T; printing::Bool=true)
    if opts === nothing || optimizers === nothing
        # Default optimization configurations
        opt1 = Optim.Options(
            iterations = 500, #500
            g_tol = 1e-6,
            f_abstol = 1e-6,
            show_trace = printing,
            show_every = 100,
            store_trace = false,
            extended_trace = false,
            allow_f_increases = true
        )
        
        opt2 = Optim.Options(
            iterations = 250, #250
            g_tol = 1e-6,
            f_abstol = 1e-6,
            show_trace = printing,
            show_every = 100,
            store_trace = false,
            extended_trace = false,
            allow_f_increases = false
        )
        
        opt3 = Optim.Options(
            iterations = 5000,
            show_trace = printing,
            show_every = 100,
            store_trace = false
        )
        
        opt5 = Optim.Options(
            iterations = 10000,
            show_trace = printing,
            show_every = 100,
            store_trace = false
        )
        
        # Default optimizers
        optimizer1 = Optim.NelderMead()
        optimizer2 = Optim.LBFGS(linesearch = LineSearches.BackTracking(order=3))
        optimizer3 = Optim.Adam(alpha = T(0.001))
        optimizer4 = Optim.NelderMead()
        
        return Dict(
            "1" => (optimizer1, opt1, "NelderMead"),
            "2" => (optimizer2, opt2, "LBFGS"), #LBFGS
            "3" => (optimizer3, opt3, "Adam"),
            "4" => (optimizer4, opt1, "NelderMead"),
            "5" => (optimizer3, opt5, "Adam-Long")
        )
    else
        return opts  # User-provided configurations
    end
end

"""
    _get_optimizer_for_group(group_id, optimizer_dict)

Retrieve optimizer configuration for a specific parameter group.
"""
function _get_optimizer_for_group(group_id, optimizer_dict)
    if haskey(optimizer_dict, group_id)
        return optimizer_dict[group_id]
    else
        # Default fallback
        return optimizer_dict["1"]
    end
end