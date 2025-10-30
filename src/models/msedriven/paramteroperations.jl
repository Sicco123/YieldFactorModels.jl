
# Get parameters from model
function get_params(model::AbstractMSEDrivenModel)
    base = model.base
    
    # Collect unique parameters from A, B, omega, delta, Phi into a single vector
    T = eltype(base.A)
    params = Vector{T}()
    
    # Get unique A and B based on duplicator
    append!(params, get_unique_params(base.A, base.duplicator))
    if !isempty(base.B)
        append!(params, get_unique_params(base.B, base.duplicator))
    end
    
    # omega, delta, Phi have no sharing
    append!(params, vec(base.omega))
    append!(params, vec(base.delta))
    append!(params, vec(base.Phi))
    
    return params
end

# Set parameters to model
function set_params!(model::AbstractMSEDrivenModel, params::AbstractVector)
    base = model.base
    
    # Set parameters from a single vector of unique parameters back into model
    index = 1
    
    # Number of unique parameters for A and B
    n_unique = maximum(base.duplicator)
    
    # Set A (expand unique parameters)
    unique_A = params[index:index + n_unique - 1]
    base.A .= expand_params(unique_A, base.duplicator)
    index += n_unique
    
    # Set B (expand unique parameters if not random walk)
    if !isempty(base.B)
        unique_B = params[index:index + n_unique - 1]
        base.B .= expand_params(unique_B, base.duplicator)
        index += n_unique
    end
    
    # Set omega, delta, Phi (no sharing)
    len_omega = length(base.omega)
    base.omega .= params[index:index + len_omega - 1]
    index += len_omega

    len_delta = length(base.delta)
    base.delta .= params[index:index + len_delta - 1]
    index += len_delta

    len_Phi = length(base.Phi)
    base.Phi .= reshape(params[index:index + len_Phi - 1], size(base.Phi))

    # Set beta, gamma, mu, nu accordingly
    base.beta .= base.delta
    base.gamma .= base.omega
    base.mu .= (I(base.M) - base.Phi) * base.delta
    base.nu .= isempty(base.B) ? zeros(eltype(base.gamma), length(base.omega)) : (ones(eltype(base.gamma), length(base.omega)) .- base.B) .* base.omega
    # Set factor loadings
    update_factor_loadings!(model, base.gamma, base.Z)
end

function get_temp_model(model::AbstractMSEDrivenModel, params::AbstractVector)
    base = model.base
    
    # Set parameters from a single vector of unique parameters back into model
    index = 1
    
    # Number of unique parameters for A and B
    n_unique = maximum(base.duplicator)
    
    # Set A (expand unique parameters)
    unique_A = params[index:index + n_unique - 1]
    A = expand_params(unique_A, base.duplicator)
    index += n_unique
    
    # Set B (expand unique parameters if not random walk)
    if !isempty(base.B)
        unique_B = params[index:index + n_unique - 1]
        B = expand_params(unique_B, base.duplicator)
        index += n_unique
    else 
        B = zeros(eltype(A), 0)
    end
    
    # Set omega, delta, Phi (no sharing)
    len_omega = length(base.omega)
    omega = [i for i in params[index:index + len_omega - 1]]
    index += len_omega
  
    len_delta = length(base.delta)
    T = typeof(params[index])
    delta = Vector{T}(params[index:index + len_delta - 1])
    index += len_delta

    len_Phi = length(base.Phi)
    T = typeof(params[index])
    vec_Phi = Vector{T}(params[index:index + len_Phi - 1])
    Phi = reshape(vec_Phi, size(base.Phi))

    beta = copy(delta)
 
    gamma = copy(omega)
 
    mu = [i for i in (I(base.M) - Phi) * delta]

    nu = isempty(B) ? zeros(eltype(gamma), length(omega)) : (ones(eltype(gamma), length(omega)) .- B) .* omega
    Z = ones(eltype(gamma), base.N, base.M)

    temp_model = build(model, Z, beta, gamma, Phi, delta, mu, A, B, omega, nu) 
    update_factor_loadings!(temp_model, temp_model.base.gamma, temp_model.base.Z)

    
    return temp_model
end



# Initialize with static parameters (for factor forecasting)
function initialize_with_static_params(model::AbstractMSEDrivenModel, params::AbstractVector, static_params::AbstractMatrix)
    base = model.base
    params[end - length(static_params)+1:end] .= static_params[:,1]
    return params
end

# Get new initial parameters for optimization trials
function get_new_initial_params(model::AbstractMSEDrivenModel, params::AbstractVector, trial::Int)
    base = model.base
    
    num_A = length(base.A_guesses)
    num_B = isempty(base.B) ? 0 : length(base.B_guesses)
    n_unique = maximum(base.duplicator)

    has_B = num_B > 0
    
    # Calculate total combinations
    total_combinations = if has_B
        n_unique == 1 ? num_A * num_B : num_A^2 * num_B^2
    else
        n_unique == 1 ? num_A : num_A^2
    end
    
    trial > total_combinations && return nothing
    
    # Split indices for multi-unique case
    if n_unique == 1
        # Single unique value
        if has_B
            A_idx = div(trial - 1, num_B) + 1
            B_idx = mod(trial - 1, num_B) + 1
            params[1] = base.A_guesses[A_idx]
            params[2] = base.B_guesses[B_idx]
        else
            params[1] = base.A_guesses[trial]
        end
    else
        # Multiple unique values: split into two halves
        half = div(n_unique, 2)
        
        if has_B
            A_idx_1 = div(trial - 1, num_A * num_B^2) + 1
            remainder = mod(trial - 1, num_A * num_B^2)
            A_idx_2 = div(remainder, num_B^2) + 1
            remainder = mod(remainder, num_B^2)
            B_idx_1 = div(remainder, num_B) + 1
            B_idx_2 = mod(remainder, num_B) + 1
            
            params[1:half] .= base.A_guesses[A_idx_1]
            params[half+1:n_unique] .= base.A_guesses[A_idx_2]
            params[n_unique+1:n_unique+half] .= base.B_guesses[B_idx_1]
            params[n_unique+half+1:2*n_unique] .= base.B_guesses[B_idx_2]
        else
            A_idx_1 = div(trial - 1, num_A) + 1
            A_idx_2 = mod(trial - 1, num_A) + 1
            
            params[1:half] .= base.A_guesses[A_idx_1]
            params[half+1:n_unique] .= base.A_guesses[A_idx_2]
        end
    end
    
    return params
end