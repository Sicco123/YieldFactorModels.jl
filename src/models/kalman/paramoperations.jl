function get_params(model::AbstractKalmanModel)
    
    return model.base.flat_params
end

function set_params!(model::AbstractDNSModel, params::AbstractVector)
    model.base.flat_params .= params

    T = typeof(params[1])

    k = 1
    model.base.gamma .= params[k]

    k += 1
    
    model.base.Omega_obs .= Matrix(I, model.base.N, model.base.N) * params[k]
    k += 1
    
    # Fill Omega_state matrix using double loop
    M = model.base.M
    for j in 1:M
        for i in 1:M
            if i == j
                # Diagonal elements: use exp transformation
                model.base.Omega_state[i, j] = params[k]
                k += 1
            elseif i < j
                # Upper triangular off-diagonal elements
                model.base.Omega_state[i, j] = params[k]
                k += 1
            else
                # Lower triangular part: set to zero
                model.base.Omega_state[i, j] = T(0.0)
            end
        end
    end

    model.base.Omega_state .= model.base.Omega_state' * model.base.Omega_state
    model.base.delta .= params[k:k+M-1]
    k += M
    model.base.Phi .= reshape(params[k:k+M*M-1], M, M)'
    k += M*M
    update_factor_loadings!(model, model.base.gamma, model.base.Z)

end

# Initialize with static parameters (for factor forecasting)
function initialize_with_static_params!(model::AbstractKalmanModel, static_params::AbstractVector)
    # placeholder
    a = 1
end

function get_new_initial_params(model::AbstractKalmanModel, params::AbstractVector, trial::Int)
    # placeholder
    Random.seed!(trial)

    params .= randn(length(params)) 
end