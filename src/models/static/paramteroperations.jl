
# Get parameters from model
function get_params(model::AbstractStaticModel)
    base = model.base
    
    # Collect unique parameters from A, B, omega, delta, Phi into a single vector
    T = eltype(base.Phi)
    params = Vector{T}()
    
    # omega, delta, Phi have no sharing
    append!(params, vec(base.gamma))
    append!(params, vec(base.delta))
    append!(params, vec(base.Phi))
    
    return params
end

# Set parameters to model
function set_params!(model::AbstractStaticModel, params::AbstractVector)
    base = model.base
    
    # Set parameters from a single vector of unique parameters back into model
    index = 1
    
    
    # Set gamma, delta, Phi (no sharing)
    len_gamma = length(base.gamma)
    base.gamma .= params[index:index + len_gamma - 1]
    index += len_gamma

    len_delta = length(base.delta)
    base.delta .= params[index:index + len_delta - 1]
    index += len_delta

    len_Phi = length(base.Phi)
    base.Phi .= reshape(params[index:index + len_Phi - 1], size(base.Phi))

    # Set beta, gamma, mu, nu accordingly
    base.beta .= base.delta
    base.mu .= (I(base.M) - base.Phi) * base.delta
    # Set factor loadings
    update_factor_loadings!(model, base.gamma, base.Z)
end

function get_temp_model(model::AbstractStaticModel, params::AbstractVector)
    base = model.base
    
    # Set parameters from a single vector of unique parameters back into model
    index = 1

    # Set gamma, delta, Phi (no sharing)
    len_gamma = length(base.gamma)
    gamma = [i for i in params[index:index + len_gamma - 1]]
    index += len_gamma

    len_delta = length(base.delta)
    T = typeof(params[index])
    delta = Vector{T}(params[index:index + len_delta - 1])
    index += len_delta

    len_Phi = length(base.Phi)
    T = typeof(params[index])
    vec_Phi = Vector{T}(params[index:index + len_Phi - 1])
    Phi = reshape(vec_Phi, size(base.Phi))

    beta = copy(delta)

    mu = [i for i in (I(base.M) - Phi) * delta]

    Z = ones(eltype(gamma), base.N, base.M)

    temp_model = build(model, Z, beta, gamma, Phi, delta, mu) 
    update_factor_loadings!(temp_model, temp_model.base.gamma, temp_model.base.Z)

    return temp_model
end



# Initialize with static parameters (for factor forecasting)
function initialize_with_static_params!(model::AbstractStaticModel, static_params::AbstractVector)
    base = model.base
    params = get_params!(model)
    params[end - base.M*(base.M +1)+1:end] .= static_params
    set_params!(model, params)
end

# Get new initial parameters for optimization trials
function get_new_initial_params(model::AbstractStaticModel, params::AbstractVector, trial::Int)
    base = model.base
    
    # set rng 
    Random.seed!(trial)
    random_draws = rand(length(params))
    params .= random_draws
    return params
end