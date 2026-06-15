# ---------------------------------------------------------------------------
# Concentrated likelihood for the neural MSE-driven model.
#
# Given the neural/score parameters (A, B, omega), the loadings path Z_{t+1|t}
# and the OLS factor path beta_{t|t} are INDEPENDENT of the linear VAR params
# (delta, Phi): those only enter the one-step prediction
#     v_t = y_{t+1} - Z_{t+1|t} (mu + Phi beta_{t|t}),    mu = (I - Phi) delta
# which is linear in theta = [mu; vec(Phi)].  So the optimal (mu, Phi) is a
# single least-squares solve, and the optimiser only needs to vary the neural
# params.  Enabled by CONCENTRATE[] (set by `run(...; concentrate=true)`); when
# false nothing here runs and the regular get_loss path is used unchanged.
#
# Validated against the filter to machine precision in exp_concentrate.jl.
# ---------------------------------------------------------------------------

# Toggle (set by run()). Default off so existing behaviour is unchanged.
const CONCENTRATE = Ref(false)

# Design block for one step: X = [Z | Z*(b' ⊗ I_M)]  (N × M(M+1)).
# Column for Phi[i,k] is Z[:,i] .* b[k]; columns 1:M are Z (for mu).
@inline function _concentrate_X!(X, Z, b, M)
    @inbounds @views X[:, 1:M] .= Z
    @inbounds for k in 1:M, i in 1:M
        @views X[:, M + (k-1)*M + i] .= Z[:, i] .* b[k]
    end
    return X
end

# Single pass over the score recursion (mirroring get_loss's num_inits
# re-seeding): build the design block X_t on the fly and accumulate the pooled
# normal equations G = ΣX'X, c = ΣX'y, and sy = Σ‖y‖² — no path storage, no
# second pass. The minimised residual is then sy - 2θ'c + θ'Gθ in closed form.
function _concentrate_accumulate(model, data, num_inits::Int)
    base = model.base
    T = eltype(base.gamma)
    nobs = size(data, 2); N = base.N; M = base.M; nlin = M * (M + 1); nt = nobs - 1
    gamma0 = copy(base.gamma)
    gamma_hist = Matrix{T}(undef, length(base.gamma), nobs)
    lo = max(1, div(nobs, 5)); hi = max(lo, div(nobs, 2))

    G = zeros(T, nlin, nlin); c = zeros(T, nlin); sy = zero(T)
    X = zeros(T, N, nlin); Gt = zeros(T, nlin, nlin); ct = zeros(T, nlin); bbuf = zeros(T, M)
    cache = initialize_filter(model)
    for init in 1:num_inits
        if init == 1
            base.gamma .= gamma0
        else
            ridx = rand(Random.MersenneTwister(42 + init), lo:hi)
            @views base.gamma .= gamma_hist[:, ridx]
        end
        update_factor_loadings!(model, base.gamma, base.Z)
        for t in 1:nobs
            y = view(data, :, t)
            @views gamma_hist[:, t] .= base.gamma
            get_β_OLS!(base.beta, base.Z, y, cache.ZtZ, cache.Zty)   # beta_t|t (pre)
            grad = get_grad_gamma!(cache, model, base.beta, base.gamma, base.Z, y)
            update_gamma_with_grad!(model, grad, Val(base.scale_grad))
            update_factor_loadings!(model, base.gamma, base.Z)
            get_β_OLS!(base.beta, base.Z, y, cache.ZtZ, cache.Zty)   # beta_t|t (refined)
            (t <= nt) && (bbuf .= base.beta)                          # snapshot beta_t|t
            if !isempty(base.B)                                       # gamma_{t+1|t}
                base.gamma .= base.nu .+ base.B .* base.gamma
                update_factor_loadings!(model, base.gamma, base.Z)
            end
            if t <= nt                                                # base.Z = Z_{t+1|t}
                _concentrate_X!(X, base.Z, bbuf, M)
                yt = view(data, :, t + 1)
                mul!(Gt, X', X); G .+= Gt
                mul!(ct, X', yt); c .+= ct
                sy += dot(yt, yt)
            end
        end
    end
    return G, c, sy
end

@inline _concentrate_theta(G, c, ::Type{T}) where T = (G + T(1e-8) * I) \ c

# Concentrated loss in the same sign/normalisation as get_loss (negative MSE).
function get_loss_concentrated(model, data::Matrix{T}; num_inits::Int=3) where T
    base = model.base
    nobs = size(data, 2)
    G, c, sy = _concentrate_accumulate(model, data, num_inits)
    theta = _concentrate_theta(G, c, T)
    rss = sy - 2 * dot(theta, c) + dot(theta, G * theta)             # Σ‖y - Xθ‖², closed form
    return -rss / base.N / nobs / num_inits
end

# After optimisation: solve the optimal (delta, Phi) for the current neural
# params and write a clean full parameter vector back into the model.
function set_concentrated_params!(model, data; num_inits::Int=3)
    base = model.base; M = base.M; T = eltype(base.gamma)
    p = get_params(model)                       # snapshot (neural params correct)
    G, c, sy = _concentrate_accumulate(model, data, num_inits)
    theta = _concentrate_theta(G, c, T)
    Phi = reshape(theta[M+1:end], M, M)
    delta = (I - Phi) \ theta[1:M]
    nPhi = length(base.Phi); ndelta = length(base.delta)
    p[end-nPhi-ndelta+1:end-nPhi] .= delta
    p[end-nPhi+1:end] .= vec(Phi)
    # Constrain to the model's transform domain (Phi diagonal -> (-1,1)) so the
    # saved vector stays valid and the VAR can't be explosive. Identity-mapped
    # params (delta, off-diagonals) pass through unchanged.
    p = transform_params(model, vec(untransform_params(model, reshape(p, length(p), 1))))
    set_params!(model, p)
    return base.delta, base.Phi
end

# ---------------------------------------------------------------------------
# Static neural models (NNS, NNS-Not-Anchored, …): loadings Z are FIXED
# (Z = net(gamma), no time variation), so beta_t = OLS(Z, y_t) and the one-step
# error y_{t+1} - Z(mu + Phi beta_t) is again linear in theta = [mu; vec(Phi)].
# Same closed-form solve, simpler forward (no score recursion, single pass).
# ---------------------------------------------------------------------------
function _concentrate_accumulate_static(model, data)
    base = model.base
    T = eltype(base.delta)
    nobs = size(data, 2); N = base.N; M = base.M; nlin = M * (M + 1); nt = nobs - 1
    update_factor_loadings!(model, base.gamma, base.Z)         # fixed Z from gamma
    cache = initialize_filter(model)
    G = zeros(T, nlin, nlin); c = zeros(T, nlin); sy = zero(T)
    X = zeros(T, N, nlin); Gt = zeros(T, nlin, nlin); ct = zeros(T, nlin); bbuf = zeros(T, M)
    for t in 1:nobs
        y = view(data, :, t)
        get_β_OLS!(base.beta, base.Z, y, cache.ZtZ, cache.Zty) # beta_t = OLS(Z, y_t)
        if t <= nt
            bbuf .= base.beta
            _concentrate_X!(X, base.Z, bbuf, M)                # Z_{t+1|t} = Z (fixed)
            yt = view(data, :, t + 1)
            mul!(Gt, X', X); G .+= Gt
            mul!(ct, X', yt); c .+= ct
            sy += dot(yt, yt)
        end
    end
    return G, c, sy
end

function get_loss_concentrated_static(model, data::Matrix{T}) where T
    base = model.base; nobs = size(data, 2)
    G, c, sy = _concentrate_accumulate_static(model, data)
    theta = _concentrate_theta(G, c, T)
    rss = sy - 2 * dot(theta, c) + dot(theta, G * theta)
    return -rss / base.N / nobs
end

function set_concentrated_params_static!(model, data)
    base = model.base; M = base.M; T = eltype(base.delta)
    p = get_params(model)
    G, c, sy = _concentrate_accumulate_static(model, data)
    theta = _concentrate_theta(G, c, T)
    Phi = reshape(theta[M+1:end], M, M)
    delta = (I - Phi) \ theta[1:M]
    nPhi = length(base.Phi); ndelta = length(base.delta)
    p[end-nPhi-ndelta+1:end-nPhi] .= delta
    p[end-nPhi+1:end] .= vec(Phi)
    # Constrain to the model's transform domain (Phi diagonal -> (-1,1)).
    p = transform_params(model, vec(untransform_params(model, reshape(p, length(p), 1))))
    set_params!(model, p)
    return base.delta, base.Phi
end

# Static neural get_loss: concentrate when toggled, else the generic path.
function get_loss(model::AbstractStaticNeuralModel, data::Matrix{T}; K::Int=1) where T<:Real
    CONCENTRATE[] && return get_loss_concentrated_static(model, data)
    return invoke(get_loss, Tuple{AbstractYieldFactorModel, Matrix{T}}, model, data; K=K)
end

# Which models support concentrate-out, and how to persist (delta, Phi).
_can_concentrate(::AbstractYieldFactorModel) = false
_can_concentrate(m::AbstractNeuralMSEDrivenModel) = m.transform_bool
_can_concentrate(::AbstractStaticNeuralModel) = true

_persist_concentrate!(::AbstractYieldFactorModel, data) = nothing
_persist_concentrate!(m::AbstractNeuralMSEDrivenModel, data) =
    set_concentrated_params!(m, data; num_inits=_default_num_inits(m))
_persist_concentrate!(m::AbstractStaticNeuralModel, data) =
    set_concentrated_params_static!(m, data)
