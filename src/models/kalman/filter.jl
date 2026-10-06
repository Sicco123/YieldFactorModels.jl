"""
    initialize_filter(model::AbstractKalmanModel)

Start the filter at the state's unconditional moments, with a fallback for the
case where those moments do not exist.

The unconditional moments require a stable Phi. `inv(I - Phi⊗Phi)` does not error
when Phi is explosive — it silently returns an indefinite `P`, which makes `F`
have a negative determinant and collapses the likelihood a few observations in.
Falling back to the one-step-ahead moments (mean and variance of the state after
one transition from zero) keeps the prior finite, PSD, and on the model's own
scale.

Estimation never reaches the fallback: `compute_loss` gates on the same spectral
radius for every Kalman model. It exists for `predict` and `get_loss_array`,
which are also the paths that can be handed explosive parameters stored by an
earlier run.

TVλ has its own method below and never runs this one.
"""
function initialize_filter(model::AbstractKalmanModel)
    base = model.base

    if all(isfinite, base.Phi) && all(isfinite, base.delta) && is_stable(base.Phi)
        base.beta .= (I - base.Phi) \ base.delta
        base.P .= reshape(inv(I - kron(base.Phi, base.Phi)) * vec(base.Omega_state), base.M, base.M)
    else
        base.beta .= base.delta
        base.P .= base.Omega_state
    end

    return nothing
end


function initialize_filter(model::AbstractTVλDNSModel)
    base = model.base
    base.beta .= (I - base.Phi) \ base.delta     # δ = (I-Φ)μ, so this is μ
    base.P .= model.Gamma
    return nothing
end

"""
    spectral_radius(Phi)

Largest eigenvalue modulus. `is_stable` is the stationarity check the Kalman
methods gate on; both assume `Phi` is already known to be finite.
"""
spectral_radius(Phi::AbstractMatrix) = maximum(abs, eigvals(Phi))
is_stable(Phi::AbstractMatrix) = spectral_radius(Phi) < 1

"""
    predict_observation!(m)

Default (DNS/1C): loadings are fixed by lambda, so `y_pred = Z * beta`.
"""
function predict_observation!(m::AbstractKalmanModel)
    mul!(m.base.y_pred, m.base.Z, m.base.beta)
    return nothing
end

# TVλ: lambda is the 4th state, so the loadings are rebuilt from beta[4] every
# step and only the first three columns enter the prediction.
function predict_observation!(m::AbstractTVλDNSModel)
    @views update_factor_loadings!(m, m.base.beta[4:4], m.base.Z)
    @views mul!(m.base.y_pred, m.base.Z[:, 1:3], m.base.beta[1:3])
    return nothing
end

"""
    update_measurement_jacobian!(m)

Write any state-dependent columns of the measurement matrix that the update needs
beyond the loadings themselves. Only TVλ has any; for every other model `base.Z`
is already the full Jacobian.
"""
update_measurement_jacobian!(::AbstractKalmanModel) = nothing

# TVλ: extended-Kalman linearization of y_pred with respect to the lambda state,
# stored in the 4th column of Z. Uses the first two columns of temp_NxM as
# scratch
function update_measurement_jacobian!(m::AbstractTVλDNSModel)
    base = m.base
    λ = m.lambda[1]
    dlambda_da4 = dtvλ_link(λ)   # dλ/dβ₄ for the exponential link (see tvλdns.jl)

    @views dZ1_dlambda = base.temp_NxM[1:base.N, 1]
    @views dZ2_dlambda = base.temp_NxM[1:base.N, 2]

    @. dZ1_dlambda = (m.z_i - base.Z[:, 2]) / λ
    @. dZ2_dlambda = base.maturities * m.z_i

    @. base.Z[:, 4] = ((base.beta[2] + base.beta[3]) * (dZ1_dlambda) + base.beta[3] * (dZ2_dlambda)) * dlambda_da4
    return nothing
end

"""
    filter!(m::AbstractKalmanModel, y, cache)

One Kalman step, shared by every Kalman model. What differs per model lives in
`predict_observation!` and `update_measurement_jacobian!`; the prediction and
measurement updates below were identical in all three implementations.

`cache` is unused here (the filter works out of the model's own preallocated
buffers) and is kept for signature compatibility with the score-driven `filter`.
"""

function filter!(m::AbstractKalmanModel, y::AbstractVector{T}, cache) where T<:Real
    base = m.base
    predict_observation!(m)

    if any(isnan, y)
        # Prediction step only (no measurement update due to missing data)
        @views temp_beta = base.temp_MxN[:, 1]  # reuse temp_MxN first column
        mul!(temp_beta, base.Phi, base.beta)
        base.beta .= base.delta .+ temp_beta

        # Use preallocated buffers for P update
        @views temp_P = base.temp_NxM[1:base.M, 1:base.M]  # reuse temp_NxM
        mul!(temp_P, base.Phi, base.P)
        mul!(base.P, temp_P, base.Phi')
        base.P .+= base.Omega_state
        return nothing
    end

    # Error Calculation
    base.v .= y .- base.y_pred
    update_measurement_jacobian!(m)

    # Update F
    base.F .= mul!(base.F, mul!(base.temp_NxM, base.Z, base.P), base.Z') .+ base.Omega_obs

    copyto!(base.F_chol, base.F)
    C = cholesky!(Symmetric(base.F_chol); check=false)
    if !issuccess(C)
        base.logdetF[1] = NaN
        fill!(base.Finv_v, NaN)
        return nothing
    end
    base.logdetF[1] = logdet(C)

    # F^-1 v, the only piece of F^-1 the log-likelihood needs (see get_loss)
    copyto!(base.Finv_v, base.v)
    ldiv!(C, base.Finv_v)

    # Kalman gain K = P Z' F^-1, formed as (F^-1 (Z P'))' since F is symmetric
    mul!(base.temp_NxM, base.Z, base.P')
    ldiv!(C, base.temp_NxM)
    transpose!(base.temp_MxN, base.temp_NxM)
    @views K = base.temp_MxN                             # reuse temp_MxN as K

    @views base.beta .+= mul!(base.temp_NxM[1:base.M, 1], K, base.v)
    @views temp_beta = base.temp_NxM[1:base.M, 1]
    mul!(temp_beta, base.Phi, base.beta)
    base.beta .= base.delta .+ temp_beta

    # Update error covariance: P = Phi * (I - K*Z) * P * Phi' + Omega_state
    @views KZ = base.temp_NxM[1:base.M, :]               # reuse temp_NxM top M rows
    mul!(KZ, base.temp_MxN, base.Z)                      # KZ := K*Z
    @views temp_IKZ = base.temp_MxN[:, 1:base.M]         # reuse temp_MxN first M cols
    temp_IKZ .= base.In .- KZ                            # temp_IKZ := I - K*Z
    @views temp_prod = base.temp_NxM[base.M+1:2*base.M, 1:base.M]  # reuse next M rows
    mul!(temp_prod, temp_IKZ, base.P)                    # temp_prod := (I - K*Z)*P
    mul!(KZ, base.Phi, temp_prod)                        # KZ := Phi*(I - K*Z)*P (reuse KZ)
    mul!(base.P, KZ, base.Phi')                          # P := Phi*(I - K*Z)*P*Phi'
    base.P .+= base.Omega_state                          # P += Omega_state

    return nothing
end



loss_objective(::AbstractKalmanModel) = :loglik
loss_objective(::AbstractTVλDNSModel) = :sse


const OBJECTIVE = Ref(:auto)

function get_loss(model::AbstractKalmanModel, data::Matrix{T}) where T<:Real
    base = model.base
    nobs = size(data, 2)
    cache = initialize_filter(model)
    
    loglik = T(0.0)
    logdet_2pi = base.N * log(2π)
    objective = OBJECTIVE[] === :auto ? loss_objective(model) : OBJECTIVE[]

    # t runs to nobs: every observation is filtered and, past the t=1 burn-in
    # (whose innovation is against the unconditional prior), scored. The loop
    # previously stopped at nobs-1, so the last observation of every window —
    # the forecast origin — never informed the parameters.
    for t in 1:nobs-1
        @views filter!(model, data[:, t], cache)

        try
            if t > 1
                loglik -= objective === :sse ? dot(base.v, base.v) :
                          T(0.5) * (base.logdetF[1] + dot(base.v, base.Finv_v) + logdet_2pi)
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


function rrmse_vs_rw(model::AbstractKalmanModel, data::AbstractMatrix{T}) where T<:Real
    base = model.base
    cache = initialize_filter(model)

    sse_model = zero(T)
    sse_rw = zero(T)
    for t in 1:size(data, 2)-1
        @views filter!(model, data[:, t], cache)
        t > 1 || continue
        y = @view data[:, t]
        yprev = @view data[:, t-1]
        (any(isnan, y) || any(isnan, yprev)) && continue
        sse_model += dot(base.v, base.v)          # v = y_t - E[y_t | t-1], set by filter!
        sse_rw += sum(abs2, y .- yprev)
    end
    return sqrt(sse_model / sse_rw)
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

    # prediction step
    filter!(model, fill(NaN, size(data, 1)) , cache)
    preds[:, end] = base.y_pred
    factors[:, end] = base.beta
    states[:, end] = base.gamma
    factor_loadings_1[:, end] = copy(base.Z[:, 2])
    factor_loadings_2[:, end] = copy(base.Z[:, 3])

    return (preds=preds, factors=factors, states=states, factor_loadings_1=factor_loadings_1, factor_loadings_2=factor_loadings_2)
end