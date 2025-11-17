function initialize_filter(model::AbstractKalmanModel)
    base = model.base
    # Initialize beta with unconditional mean: (I - Phi)^(-1) * delta
    base.beta .= (I - base.Phi) \ base.delta

    # Initialize P with unconditional covariance
    base.P .= reshape(inv(I - kron(base.Phi, base.Phi)) * vec(base.Omega_state), base.M, base.M)
    
    return nothing
end

function filter!(m::AbstractKalmanModel, y::AbstractVector{T}, cache) where T<:Real

    
    if any(isnan.(y))
        # Use preallocated buffers to avoid allocations
        @views temp_beta = m.base.temp_MxN[:, 1]  # reuse temp_MxN first column
        mul!(temp_beta, m.base.Phi, m.base.beta)
        m.base.beta .= m.base.delta .+ temp_beta
        
        # Use preallocated buffers for P update
        @views temp_P = m.base.temp_NxM[1:m.base.M, 1:m.base.M]  # reuse temp_NxM
        mul!(temp_P, m.base.Phi, m.base.P)
        mul!(m.base.P, temp_P, m.base.Phi')
        m.base.P .+= m.base.Omega_state
        return nothing
    end
    
    # Error Calculation
    mul!(m.base.y_pred, m.base.Z, m.base.beta)
    m.base.v .= y .- m.base.y_pred
   
    # Update F
    m.base.F .= mul!(m.base.F, mul!(m.base.temp_NxM, m.base.Z, m.base.P), m.base.Z') .+ m.base.Omega_obs 

    try 
        m.base.F_inv .= inv(m.base.F)
    catch
        println("Error inverting F")
        return nothing
    end

    # Kalman Gain transposed stored in 
    m.base.temp_MxN .= mul!(m.base.temp_MxN, mul!(m.base.temp_NxM, m.base.Z,  m.base.P')', m.base.F_inv)
    @views K = m.base.temp_MxN                      # reuse temp_MxN as K


    @views m.base.beta .+= mul!(m.base.temp_NxM[1:m.base.M, 1], K, m.base.v)
    @views m.base.beta .= m.base.delta .+ mul!(m.base.temp_NxM[1:m.base.M, 1], m.base.Phi, m.base.beta)
    # Update error covariance: P = Phi * (I - K*Z) * P * Phi' + Omega_state
    @views KZ = m.base.temp_NxM[1:m.base.M, :]               # reuse temp_NxM top M rows
    mul!(KZ, m.base.temp_MxN, m.base.Z)                    # KZ := K*Z
    @views temp_IKZ = m.base.temp_MxN[:, 1:m.base.M]         # reuse temp_MxN first M cols
    temp_IKZ .= m.base.In .- KZ                          # temp_IKZ := I - K*Z
    @views temp_prod = m.base.temp_NxM[m.base.M+1:2*m.base.M, 1:m.base.M]  # reuse next M rows
    mul!(temp_prod, temp_IKZ, m.base.P)                  # temp_prod := (I - K*Z)*P
    mul!(KZ, m.base.Phi, temp_prod)                      # KZ := Phi*(I - K*Z)*P (reuse KZ)
    mul!(m.base.P, KZ, m.base.Phi')                      # P := Phi*(I - K*Z)*P*Phi'
    m.base.P .+= m.base.Omega_state                      # P += Omega_state

    return nothing
end


function get_loss(model::AbstractKalmanModel, data::Matrix{T}) where T<:Real
    base = model.base
    nobs = size(data, 2)
    cache = initialize_filter(model)

    loglik = T(0.0)
    logdet_2pi = base.N * log(2π)
    
    for t in 1:nobs-1
        filter!(model, data[:, t], cache)
        
        try 
            if t > 1
                loglik -= T(0.5) * (logdet(base.F) + (base.v' * base.F_inv * base.v)[1] + logdet_2pi)
            end
        catch
            println("Error computing loglik at time $t")
            return -Inf
        end

        if isinf(loglik) || isnan(loglik)
            return -Inf
        end
    
    end
   
    return loglik
end

function get_loss_array(model::AbstractKalmanModel, data::Matrix{T}; K::Int=1 ) where T<:Real
    base = model.base
    nobs = size(data, 2)
    cache = initialize_filter(model)

    mse = Vector{T}(undef, nobs -1)
    fill!(mse, 0.0)
    
    catched_params = similar(get_params(model))

    @inbounds for k in 0:K-1
        catch_point = Int(floor(nobs*((0.25) + 0.75*(k)/K)))
        if k > 1
            set_params!(model, catched_params) 
        end 
        for t in 1:nobs-1
            @views filter!(model, data[:, t], cache)
            @views @. base.v = data[:, t] - base.y_pred

            if t > 1
                mse[t] -= dot(base.v, base.v)
            end

            if isinf(mse[t]) || isnan(mse[t])
                return -Inf
            end
        
            if t == catch_point
                catched_params = copy(get_params(model))
            end
        end
    end

    # In-place division to avoid allocation
    @. mse = mse / base.N / K
    return mse
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
        @views filter!(model, data[:, t], cache)
        if t >1
            preds[:, t-1] = base.y_pred
            factors[:, t-1] = base.beta
            states[:, t-1] = base.gamma
            factor_loadings_1[:, t-1] = copy(base.Z[:, 2])
            factor_loadings_2[:, t-1] = copy(base.Z[:, 3])
        end
    end

    return (preds=preds, factors=factors, states=states, factor_loadings_1=factor_loadings_1, factor_loadings_2=factor_loadings_2)
end