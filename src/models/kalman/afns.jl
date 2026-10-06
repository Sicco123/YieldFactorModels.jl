### AFNSModel - Independent-factor Arbitrage-Free Nelson-Siegel model
#
# Estimation by Kalman filter + maximum likelihood, following
#
#     Christensen, Diebold & Rudebusch (2011),
#     "The affine arbitrage-free class of Nelson-Siegel term structure models",
#     Journal of Econometrics 164, 4-20.
#


struct AFNSModel{Fl <: Real, Fβ <: Real} <: AbstractAFNSModel
    # Base
    base::KalmanBaseModel{Fl, Fβ}
    lambda::Vector{Fl}
    A::Vector{Fl}
    dt::Fl

   
    function AFNSModel{T}(maturities::Vector{T}, N::Int, M::Int;
                          model_string::String = "AFNS",
                          results_location::String = "results/",
                          dt::T = T(1.0)) where T<:Real

        # one observation-driven-style parameter slot for lambda (identity, like DNS)
        specific_transformations   = Function[identity]
        specific_untransformations = Function[identity]

        L = 1

        base = KalmanBaseModel{T}(maturities, N, M, L,
                                  specific_transformations, specific_untransformations,
                                  model_string;
                                  results_location=results_location)

        # Diagonal-factor / per-series-variance variant: rebuild the parameter
        # transforms so AFNS has N measurement-error variances and a diagonal
        # Sigma. Other models keep the base layout untouched.
        rebuild_afns_transforms!(base, N, M)

        lambda = fill(T(0.0), 1)
        A      = zeros(T, N)

        new{T, T}(base, lambda, A, dt)
    end
end

const AFNS_OBSVAR_FLOOR = 1e-8
const AFNS_SIGMA_FLOOR  = 5e-3
const AFNS_SIGMA_CAP    = 5e-2

# Parameter layout: [lambda | N obs variances | M diagonal-Sigma | M thetaP | M diagonal-KP].
function rebuild_afns_transforms!(base::KalmanBaseModel, N::Int, M::Int)
    trans   = Function[identity]        # lambda
    untrans = Function[identity]
    for _ in 1:N                        # N obs variances, floored
        push!(trans,   x -> AFNS_OBSVAR_FLOOR + exp(x))
        # max: a stored value at/below the floor must untransform finitely, not to -Inf/NaN
        push!(untrans, x -> log(max(x - AFNS_OBSVAR_FLOOR, AFNS_OBSVAR_FLOOR)))
    end
    for _ in 1:M                        # M diagonal Sigma, in (floor, cap)
        push!(trans,   x -> AFNS_SIGMA_FLOOR + (AFNS_SIGMA_CAP - AFNS_SIGMA_FLOOR) / (1 + exp(-x)))
        # clamp: stored values at/outside the bounds must untransform finitely
        push!(untrans, x -> begin
            z = clamp((x - AFNS_SIGMA_FLOOR) / (AFNS_SIGMA_CAP - AFNS_SIGMA_FLOOR), 1e-6, 1 - 1e-6)
            log(z / (1 - z))
        end)
    end
    for _ in 1:M                        # thetaP
        push!(trans, identity); push!(untrans, identity)
    end
    for _ in 1:M                        # diagonal KP, positive (so Phi is always stationary)
        push!(trans, from_R_to_pos); push!(untrans, from_pos_to_R)
    end

    resize!(base.transformations,   length(trans));   base.transformations   .= trans
    resize!(base.untransformations, length(untrans)); base.untransformations .= untrans
    resize!(base.flat_params,       length(trans));   fill!(base.flat_params, zero(eltype(base.flat_params)))
    return
end

function get_static_model_type(model::AbstractAFNSModel)
    return "AFNS"
end

function reconcile_init_params(model::AbstractAFNSModel, all_params::AbstractMatrix)
    N    = model.base.N
    M    = model.base.M
    tail = 3M
    n0   = size(all_params, 1)
    n0 == 1 + N + tail && return all_params      # already matches

    # Stored file from the old correlated-KP layout at this N (tail was 2M+M*M):
    # keep lambda/obs/Sigma/theta, take the diagonal of the stored KP.
    if n0 == 1 + N + 2M + M * M
        println("AFNS init params use the old correlated-KP layout; taking diag(KP).")
        out = Matrix{eltype(all_params)}(undef, 1 + N + tail, size(all_params, 2))
        kK = 1 + N + 2M
        @views for j in axes(all_params, 2)
            out[1:kK, j] .= all_params[1:kK, j]
            for i in 1:M
                out[kK+i, j] = all_params[kK+(i-1)*M+i, j]
            end
        end
        return out
    end

    n0 <= 1 + tail && return all_params           # no obs block to read; let caller handle

    out = Matrix{eltype(all_params)}(undef, 1 + N + tail, size(all_params, 2))
    @views for j in axes(all_params, 2)
        col     = all_params[:, j]
        obs     = col[2:n0-tail]                   # stored per-maturity variances
        out[1, j]          = col[1]                # lambda
        out[2:1+N, j]     .= sum(obs) / length(obs)
        out[2+N:end, j]   .= col[n0-tail+1:n0]     # diag-Sigma, thetaP, diag-KP
    end
    return out
end


function update_factor_loadings!(model::AbstractAFNSModel, lambda::AbstractVector{R}, Z::AbstractMatrix) where {R<:Real}
    λ = lambda[1]
    tau_maturities = λ .* model.base.maturities
    z_i = exp.(-tau_maturities)

    Z[:, 1] .= one(eltype(Z))
    Z[:, 2] .= (1 .- z_i) ./ tau_maturities
    Z[:, 3] .= Z[:, 2] .- z_i
    return
end

# Yield-adjustment intercept A_i = -A(tau_i)/tau_i, using the analytical formula 
function update_yield_adjustment!(model::AbstractAFNSModel, lambda::Real, Sigma::AbstractMatrix)
    s = Sigma
    cA = s[1, 1]^2
    cB = s[2, 1]^2 + s[2, 2]^2
    cC = s[3, 1]^2 + s[3, 2]^2 + s[3, 3]^2
    cD = s[1, 1] * s[2, 1]
    cE = s[1, 1] * s[3, 1]
    cF = s[2, 1] * s[3, 1] + s[2, 2] * s[3, 2]

    l  = lambda
    l2 = l^2
    l3 = l^3

    @inbounds for i in eachindex(model.base.maturities)
        t  = model.base.maturities[i]
        e1 = exp(-l * t)
        e2 = exp(-2 * l * t)

        tA = t^2 / 6
        tB = 1 / (2l2) - (1 / l3) * (1 - e1) / t + (1 / (4l3)) * (1 - e2) / t
        tC = (1 / (2l2) + (1 / l2) * e1 - (1 / (4l)) * t * e2
              - (3 / (4l2)) * e2 - (2 / l3) * (1 - e1) / t
              + (5 / (8l3)) * (1 - e2) / t)
        tD = (1 / (2l)) * t + (1 / l2) * e1 - (1 / l3) * (1 - e1) / t
        tE = ((3 / l2) * e1 + (1 / (2l)) * t + (1 / l) * t * e1
              - (3 / l3) * (1 - e1) / t)
        tF = (1 / l2 + (1 / l2) * e1 - (1 / (2l2)) * e2
              - (3 / l3) * (1 - e1) / t + (3 / (4l3)) * (1 - e2) / t)

        A_over_tau = cA * tA + cB * tB + cC * tC + cD * tD + cE * tE + cF * tF
        model.A[i] = -A_over_tau
    end
    return
end

function get_param_groups(model::AbstractAFNSModel, param_groups::Vector{String})
    base = model.base
    length_total_params = length(get_params(model))
    if length(param_groups) == length_total_params
        return param_groups
    else
        println("Default param groups assigned (group \"5\": NM).")
        return vcat(fill("5", length_total_params))
    end
end

function update_transition!(model::AbstractAFNSModel, KP::AbstractMatrix, Sigma::AbstractMatrix)
    n  = model.base.M
    dt = model.dt
    R  = eltype(KP)

    SS = Sigma * Sigma'
    VL = zeros(R, 2n, 2n)
    @views VL[1:n, 1:n]       .= -KP
    @views VL[1:n, n+1:2n]    .= SS
    @views VL[n+1:2n, n+1:2n] .= KP'
    VL .*= dt

    # search backtracks.
    if !all(isfinite, VL)
        fill!(model.base.Phi, R(NaN))
        fill!(model.base.Omega_state, R(NaN))
        return model.base.Phi
    end

    eVL = try
        exp(VL)                             
    catch
        fill!(model.base.Phi, R(NaN))
        fill!(model.base.Omega_state, R(NaN))
        return model.base.Phi
    end
    Phi = eVL[1:n, 1:n]
    Q   = eVL[1:n, n+1:2n] * Phi'

    model.base.Phi .= Phi
    model.base.Omega_state .= (Q .+ Q') ./ 2   
    return Phi
end

function set_params!(model::AbstractAFNSModel, params::AbstractVector)
    model.base.flat_params .= params
    R = eltype(params)
    N = model.base.N
    M = model.base.M

    # [1] lambda
    model.base.gamma[1] = params[1]
    model.lambda[1]     = R(1e-2) + exp(params[1])

    # [2 : N+1] measurement-error variance, one per maturity
    fill!(model.base.Omega_obs, zero(R))
    @inbounds for i in 1:N
        model.base.Omega_obs[i, i] = params[1 + i]
    end

    # [N+2 : N+1+M] diagonal Sigma
    Sigma = zeros(R, M, M)
    @inbounds for i in 1:M
        Sigma[i, i] = params[N + 1 + i]
    end

    # [N+2+M : N+1+2M] theta^P
    kθ = N + 2 + M
    thetaP = @view params[kθ:kθ+M-1]

    # [N+2+2M : end] K^P — diagonal (independent-factor AFNS)
    kK = kθ + M
    KP = zeros(R, M, M)
    @inbounds for i in 1:M
        KP[i, i] = params[kK+i-1]
    end

    # cross-section
    update_factor_loadings!(model, model.lambda, model.base.Z)
    update_yield_adjustment!(model, model.lambda[1], Sigma)

    # transition: Phi, Q and delta = (I - Phi) theta^P
    Phi = update_transition!(model, KP, Sigma)
    model.base.delta .= (I - Phi) * thetaP
    return
end


# AFNS carries the arbitrage-free yield-adjustment intercept on top of the
# Nelson-Siegel loadings; the rest of the step is the shared kernel in
# models/kalman/filter.jl.
function predict_observation!(m::AbstractAFNSModel)
    mul!(m.base.y_pred, m.base.Z, m.base.beta)
    m.base.y_pred .+= m.A
    return nothing
end


function get_loss(model::AbstractAFNSModel, data::Matrix{T}) where T<:Real
    if !all(isfinite, model.base.Phi) || !all(isfinite, model.base.Omega_state)
        return T(-Inf)
    end
    try
        return invoke(get_loss, Tuple{AbstractKalmanModel, Matrix{T}}, model, data)
    catch
        return T(-Inf)
    end
end

function predict(model::AbstractAFNSModel, data::Matrix{T}; K::Int=3) where T<:Real
    r = invoke(predict, Tuple{AbstractKalmanModel, Matrix{T}}, model, data; K=K)
    return merge(r, (preds = T(100) .* r.preds,))
end

function get_loss_array(model::AbstractAFNSModel, data::Matrix{T}; K::Int=1) where T<:Real
    return T(100)^2 .* invoke(get_loss_array, Tuple{AbstractKalmanModel, Matrix{T}}, model, data; K=K)
end
