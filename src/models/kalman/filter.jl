function filter!(m::AbstractKalmanModel, y::AbstractVector{T}, cache) where T<:Real
    
    if any(isnan.(y))
        m.beta .= m.delta .+ m.Phi * m.beta
        m.P = m.Phi * m.P * m.Phi' .+ m.Omega_state
        return m.beta, m.P
    end
    
    # Error Calculation
    m.y_pred .= m.Z * m.beta
    m.v .= y .- m.y_pred
    
    # Update F
    m.F .= m.Z * m.P * m.Z' .+ m.Omega_obs 

    try 
        m.F_inv .= inv(m.F)
    catch
        println("Error inverting F")
        return nothing
    end

    # Kalman Gain
    K = m.P * m.Z' * m.F_inv

    # Update state estimate
    m.beta .= m.delta .+ m.Phi * (m.beta .+ K * m.v)
    # Update error covariance
    KZ = K * m.Z
    m.P .= m.Phi * ((m.In .- KZ) * m.P) * m.Phi' .+ m.Omega_state

    return nothing
end


function get_loss(model::AbstractKalmanModel, data::Matrix{T}) where T<:Real
    base = model.base
    nobs = size(data, 2)
    cache = initialize_filter(model)

    loglik = T(0.0)
    logdet_2pi = base.N * log(2π)
    
    for t in 1:nobs-1
        @views filter!(model, data[:, t], cache)
        
        loglik += T(0.5) * (logdet(model.F) + (model.v' * model.F_inv * model.v)[1] + logdet_2pi)

        if isinf(loglik) || isnan(loglik)
            return Inf
        end
    
    end
   
    return loglik
end

function predict(model::AbstractKalmanModel, data::Matrix{T}; K::Int=3) where T<:Real
    base = model.base
    nobs = size(data, 2)

    cache = initialize_filter(model)

    preds = Matrix{T}(undef, size(data))
    factors = Matrix{T}(undef, model.base.M, nobs)
    states = Matrix{T}(undef, model.base.L, nobs)
    factor_loadings_1 = Matrix{T}(undef, model.base.N,  nobs)
    factor_loadings_2 = Matrix{T}(undef, model.base.N,  nobs)

    for t in 1:nobs
        pred = filter(model, data[:, t], cache)
        preds[:, t] = pred
        factors[:, t] = base.beta
        states[:, t] = base.gamma
        factor_loadings_1[:, t] = copy(base.Z[:, 2])
        factor_loadings_2[:, t] = copy(base.Z[:, 3])
    end

    return (preds=preds, factors=factors, states=states, factor_loadings_1=factor_loadings_1, factor_loadings_2=factor_loadings_2)
end