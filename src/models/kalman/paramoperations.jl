function get_params(model::AbstractKalmanModel)
    
    return model.base.flat_params
end

function set_params!(base::KalmanBaseModel, params::AbstractVector)
    base.flat_params[end-length(params)+1:end] .= params

    T = typeof(params[1])

    k = 1
 
    base.Omega_obs .= Matrix(I, base.N, base.N) * params[k]
    k += 1
    
    # Fill Omega_state matrix using double loop
    M = base.M
    for j in 1:M
        for i in 1:M
            if i == j
                # Diagonal elements: use exp transformation
                base.Omega_state[i, j] = params[k]
                k += 1
            elseif i < j
                # Upper triangular off-diagonal elements
                base.Omega_state[i, j] = params[k]
                k += 1
            else
                # Lower triangular part: set to zero
                base.Omega_state[i, j] = T(0.0)
            end
        end
    end

    base.Omega_state .= base.Omega_state' * base.Omega_state
    base.delta .= params[k:k+M-1]
    k += M
    base.Phi .= reshape(params[k:k+M*M-1], M, M)'
    k += M*M
   
end



function set_params!(model::AbstractDNSModel, params::AbstractVector)
   
    model.base.flat_params .= params

   
    k = 1
    model.base.gamma .= params[k]

    k += 1

    set_params!(model.base, params[k:end])

    update_factor_loadings!(model, model.base.gamma, model.base.Z)

end

function set_params!(model::AbstractTVλDNSModel, params::AbstractVector)

    model.base.flat_params .= params

    set_params!(model.base, params)

    
end

# Initialize with static parameters (for factor forecasting)
function initialize_with_static_params!(model::AbstractKalmanModel, static_params::AbstractVector)
    # placeholder
    a = 1
end



function initialize_with_static_params(model::AbstractTVλDNSModel, params::AbstractVector, static_params::AbstractMatrix)
    base = model.base

    # phi matrix
    params[1:1] .= static_params[2:2]
    params[2:7] .= static_params[end-17:end-12]
    params[12:14] .= static_params[end-11:end-9]
    params[16:18] .= static_params[end-8:end-6]
    params[20:22] .= static_params[end-5:end-3] 
    params[24:26] .= static_params[end-2:end,1]
    return params
end


function get_new_initial_params(model::AbstractKalmanModel, params::AbstractVector, trial::Int)
    # placeholder
    Random.seed!(trial)

    params .= randn(length(params)) 
end