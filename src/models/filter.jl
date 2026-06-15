# Cache struct for gradient computation - don't preallocate Dual arrays
struct GradientCache{C,T}
    cfg::C
    Ztemp_any::Base.RefValue{Any}   # will hold Matrix{<:Real or <:Dual}
    v_any::Base.RefValue{Any}       # will hold Vector{<:Real or <:Dual}
    ZtZ::Matrix{T}                  # Pre-allocated for Z'Z
    Zty::Vector{T}                  # Pre-allocated for Z'y
    H::Matrix{T}                    # 3 × 2N: analytic-score hidden activations (h1 | h2)
    S::Matrix{T}                    # N × 10: analytic-score column scratch
    grad::Vector{T}                 # length-L analytic-score output buffer
    pred::Vector{T}                 # N: one-step prediction buffer (Z*beta)
    tmp::Vector{T}                  # M: transition buffer (Phi*beta)
end

function GradientCache(gamma::AbstractVector{T}, M::Int, N::Int) where T
    cfg = ForwardDiff.GradientConfig(nothing, gamma, ForwardDiff.Chunk{length(gamma)}())
    ZtZ = Matrix{T}(undef, M, M)
    Zty = Vector{T}(undef, M)
    H = Matrix{T}(undef, 3, 2N)
    S = Matrix{T}(undef, N, 10)
    grad = Vector{T}(undef, length(gamma))
    pred = Vector{T}(undef, N)
    tmp = Vector{T}(undef, M)
    return GradientCache{typeof(cfg),T}(cfg, Ref{Any}(nothing), Ref{Any}(nothing),
                                        ZtZ, Zty, H, S, grad, pred, tmp)
end



function initialize_filter(model::AbstractYieldFactorModel)
    base = model.base
    if hasfield(typeof(base), :grad_EWMA)
       base.grad_EWMA .= zeros(eltype(base.gamma), length(base.gamma))
       base.grad_EWMA_count .= [0]
    end
    return GradientCache(base.gamma, base.M, base.N)
end


@inline function update_gamma_with_grad!(m::AbstractMSEDrivenModel, grad_gamma::AbstractVector, ::Val{true})
    # When gradient scaling is active, update the EWMA of squared gradients
    # and apply bias correction before scaling the current gradient.
    β = m.base.forget_factor
    # Update EWMA
    m.base.grad_EWMA .= β .* m.base.grad_EWMA .+ (1 - β) .* (grad_gamma .* grad_gamma)
    # Increment counter for bias correction
    m.base.grad_EWMA_count .+= 1
    # Bias‑corrected second moment
    denom_factor = 1 - β^(m.base.grad_EWMA_count[1])
    ε = eps(eltype(m.base.gamma))
    grad_gamma .= grad_gamma ./ (sqrt.(m.base.grad_EWMA ./ denom_factor) .+ ε)
    # Apply update using step sizes A
    m.base.gamma .+= grad_gamma .* m.base.A
    return nothing
end

@inline function update_gamma_with_grad!(m::AbstractMSEDrivenModel, grad_gamma::AbstractVector, ::Val{false})
    # When gradient scaling is not active, perform the original update directly.
    m.base.gamma .+= grad_gamma .* m.base.A
    return nothing
end

function filter(m::AbstractMSEDrivenModel, y::AbstractVector{T}, cache) where T<:Real
    if isnan(y[1])
        if !isempty(m.base.B)
            m.base.gamma .= m.base.nu .+ m.base.B .* m.base.gamma
            update_factor_loadings!(m, m.base.gamma, m.base.Z)
        end
        mul!(cache.tmp, m.base.Phi, m.base.beta)
        m.base.beta .= m.base.mu .+ cache.tmp
        mul!(cache.pred, m.base.Z, m.base.beta)
        return cache.pred
    end
    # 1) Get beta OLS
    try
        get_β_OLS!(m.base.beta, m.base.Z, y, cache.ZtZ, cache.Zty)
    catch e
        println("Error in OLS calculation: ", e)
        return -Inf
    end

    # 2) Get gamma t|t
    grad_gamma = get_grad_gamma!(cache, m, m.base.beta, m.base.gamma, m.base.Z, y)
    # Update gamma using dispatch on whether gradient scaling is active.
    update_gamma_with_grad!(m, grad_gamma, Val(m.base.scale_grad))
    update_factor_loadings!(m, m.base.gamma, m.base.Z)

    # 3) Get beta OLS
    try
        get_β_OLS!(m.base.beta, m.base.Z, y, cache.ZtZ, cache.Zty)
    catch e
        println("Error in OLS calculation: ", e)
        return -Inf
    end

    # 4) Predictions
    if !isempty(m.base.B)
        m.base.gamma .= m.base.nu .+ m.base.B .* m.base.gamma
        update_factor_loadings!(m, m.base.gamma, m.base.Z)
    end
    mul!(cache.tmp, m.base.Phi, m.base.beta)
    m.base.beta .= m.base.mu .+ cache.tmp
    mul!(cache.pred, m.base.Z, m.base.beta)
    return cache.pred
end

function filter(m::AbstractStaticModel, y::AbstractVector{T}, cache) where T<:Real

    if isnan(y[1])
        m.base.beta .= m.base.mu  .+ m.base.Phi * m.base.beta
        return m.base.Z * m.base.beta
    end
    # 1) Get beta OLS
    try
        get_β_OLS!(m.base.beta, m.base.Z, y, cache.ZtZ, cache.Zty)
    catch e
        println("Error in OLS calculation: ", e)
        return -Inf
    end

    # 4) Predictions 
    m.base.beta .= m.base.mu  .+ m.base.Phi * m.base.beta
    return m.base.Z * m.base.beta
end

function filter(m::AbstractRandomWalkModel, y::AbstractVector{T}, cache) where T<:Real
    if isnan(y[1])
        return copy(m.last_y)
    end

    m.last_y .= y

    return copy(m.last_y)
end

function get_β_OLS!(beta, Z, y, ZtZ, Zty)
    try
        mul!(ZtZ, Z', Z)
        mul!(Zty, Z', y)
        F = cholesky!(Symmetric(ZtZ))
        ldiv!(beta, F, Zty)
    catch e
        M = size(Z, 2)
        mul!(ZtZ, Z', Z)
        mul!(Zty, Z', y)
        @views ZtZ[diagind(ZtZ)] .+= 1e-3
        F = cholesky!(Symmetric(ZtZ))
        ldiv!(beta, F, Zty)
    end
    return nothing
end

function get_β_OLS!(beta, Z, y)
    try
        ldiv!(beta, cholesky!(Z'Z), Z'y)
    catch e
        M = size(Z, 2)
        F = cholesky!(Z'Z + 1e-3*I(M))
        Zty = Z'y
        ldiv!(beta, F, Zty)
    end
    return nothing
end

function _buffers!(cache::GradientCache, Z_proto::AbstractMatrix, y_proto::AbstractVector, p)
    ET = eltype(p)  # Real or ForwardDiff.Dual
    Zt = cache.Ztemp_any[]
    vt = cache.v_any[]


    if !(Zt isa AbstractMatrix{ET}) || size(Zt) != size(Z_proto)
        Zt = similar(Z_proto, ET)
        cache.Ztemp_any[] = Zt
    end
    if !(vt isa AbstractVector{ET}) || length(vt) != length(y_proto)
        vt = similar(y_proto, ET)
        cache.v_any[] = vt
    end
    return Zt, vt
end

# Generic score: ∂(-‖y - Z(γ)β‖²)/∂γ via ForwardDiff (fallback for any model).
function _grad_gamma_ad!(cache::GradientCache, m, beta, gamma, Z_proto, y)
    cfg = cache.cfg

    # scalar loss (log-likelihood) over gamma, reusing scratch
    function llf(p)
        Z_temp, v = _buffers!(cache, ForwardDiff.value.(Z_proto), y, p)
        update_factor_loadings!(m, p, Z_temp)  # writes into Z_temp
        mul!(v, Z_temp, ForwardDiff.value.(beta))                  # v = Z_temp * beta
        @. v = y - v
        # return a plain scalar; ForwardDiff is fine with this
        return -dot(v, v)
    end

    res = DiffResults.GradientResult(gamma)
    ForwardDiff.gradient!(res, llf, gamma, cfg)
    return DiffResults.gradient(res)
end

get_grad_gamma!(cache::GradientCache, m, beta, gamma, Z_proto, y) =
    _grad_gamma_ad!(cache, m, beta, gamma, Z_proto, y)

# Neural models: use the closed-form score for the anchored loadings
# (transform_bool = true); fall back to ForwardDiff otherwise.
function get_grad_gamma!(cache::GradientCache, m::AbstractNeuralMSEDrivenModel, beta, gamma, Z_proto, y)
    return m.transform_bool ? _analytic_score_neural!(cache, m, beta, gamma, y) :
                              _grad_gamma_ad!(cache, m, beta, gamma, Z_proto, y)
end

# Closed-form ∂(-‖y - Z(γ)β‖²)/∂γ for the anchored two-net loadings, replacing
# the per-step ForwardDiff. The map is γ_a → net1 → transform_net_1! → Z[:,2]
# (and γ_b → net2 → transform_net_2! → Z[:,3]); each net is a fixed 1→3→1 MLP
# (tanh hidden, no output bias). We reverse-mode (VJP) the whole chain by hand:
# seed 2β₂v on Z[:,2] and 2β₃v on Z[:,3], backprop through the transforms, then
# through the nets. Must match the AD score to machine precision (verified in
# bench_hotloop.jl) so the optimization trajectory is unchanged.
function _analytic_score_neural!(cache::GradientCache, m, beta, gamma, y)
    base = m.base
    N = base.N
    T = eltype(gamma)
    x = m.net_input                       # 1 × N maturities (fixed)
    b1 = beta[1]; b2 = beta[2]; b3 = beta[3]
    ε = T(1e-7)
    scale = T(0.9610)

    # preallocated scratch (cache.H 3×2N, cache.S N×10, cache.grad length-L)
    h1 = view(cache.H, :, 1:N);      h2 = view(cache.H, :, N+1:2N)
    r2 = view(cache.S, :, 1);  z2 = view(cache.S, :, 2);  t2 = view(cache.S, :, 3)
    r3 = view(cache.S, :, 4);  z3 = view(cache.S, :, 5);  p  = view(cache.S, :, 6)
    resid = view(cache.S, :, 7); v = view(cache.S, :, 8)
    gr2 = view(cache.S, :, 9);  gr3 = view(cache.S, :, 10)
    grad = cache.grad

    # ---------------- net 1 forward (γ[1:9]) ----------------
    @inbounds for j in 1:N
        xj = x[1, j]; s1 = zero(T)
        for k in 1:3
            hk = tanh(gamma[k]*xj + gamma[3+k]); h1[k, j] = hk
            s1 += gamma[6+k]*hk
        end
        r2[j] = s1
    end
    # transform_net_1! (Val{true}): z2[i]=((r2[i]-r2[N-1])·s)² on 2:N-2; ends fixed
    raw_last1 = r2[N-1]
    s = inv(r2[1] - r2[N-1] + ε)
    @inbounds for i in 2:N-2
        ti = (r2[i] - raw_last1)*s; t2[i] = ti; z2[i] = ti*ti
    end
    z2[1] = one(T); z2[N-1] = zero(T); z2[N] = zero(T)

    # ---------------- net 2 forward (γ[10:18]) ----------------
    @inbounds for j in 1:N
        xj = x[1, j]; s1 = zero(T)
        for k in 1:3
            hk = tanh(gamma[9+k]*xj + gamma[12+k]); h2[k, j] = hk
            s1 += gamma[15+k]*hk
        end
        r3[j] = s1
    end
    # transform_net_2! (Val{true}): detrend vs endpoint line, square, normalise
    x1 = x[1, 1]; xN = x[1, N]; invdx = inv(xN - x1)
    slope = (r3[N] - r3[1])*invdx
    intercept = r3[1] - slope*x1
    sum_sq = zero(T)
    @inbounds for i in 2:N-1
        ri = r3[i] - (slope*x[1, i] + intercept); resid[i] = ri
        pi = ri*ri; p[i] = pi; sum_sq += pi*pi
    end
    sqrtS = sqrt(sum_sq)
    denom = sqrtS/scale + ε
    inv_denom = inv(denom)
    z3[1] = zero(T); z3[N] = zero(T)
    @inbounds for i in 2:N-1
        z3[i] = p[i]*inv_denom
    end

    # ---------------- residual (seeds are gz2=2β₂v, gz3=2β₃v, applied inline) ----
    @inbounds for j in 1:N
        v[j] = y[j] - (b1 + b2*z2[j] + b3*z3[j])
        gr2[j] = zero(T); gr3[j] = zero(T)       # zero accumulators (buffers reused)
    end

    # ---------------- transform 1 backward → gr2 ----------------
    @inbounds for i in 2:N-2
        ti = t2[i]; gi = 2*b2*v[i]
        gr2[i]   += gi*(2*ti*s)
        gr2[1]   += gi*(-2*ti*ti*s)
        gr2[N-1] += gi*(2*ti*s*(ti - 1))
    end
    # net 1 backward → grad[1:9]
    @inbounds for k in 1:3
        gW2 = zero(T); gW1 = zero(T); gb = zero(T); W2k = gamma[6+k]
        for j in 1:N
            hkj = h1[k, j]; gj = gr2[j]
            gW2 += gj*hkj
            d = gj*W2k*(1 - hkj*hkj)
            gb += d; gW1 += d*x[1, j]
        end
        grad[k] = gW1; grad[3+k] = gb; grad[6+k] = gW2
    end

    # ---------------- transform 2 backward → gr3 ----------------
    A = zero(T)
    @inbounds for i in 2:N-1
        A += (2*b3*v[i])*p[i]
    end
    secondc = denom*denom*scale*sqrtS
    inv_secondc = secondc > 0 ? inv(secondc) : zero(T)
    @inbounds for k in 2:N-1
        gp = (2*b3*v[k])*inv_denom - p[k]*A*inv_secondc
        gresid = gp*2*resid[k]
        wk = (x[1, k] - x1)*invdx
        gr3[k] += gresid
        gr3[1] += gresid*(-(1 - wk))
        gr3[N] += gresid*(-wk)
    end
    # net 2 backward → grad[10:18]
    @inbounds for k in 1:3
        gW2 = zero(T); gW1 = zero(T); gb = zero(T); W2k = gamma[15+k]
        for j in 1:N
            hkj = h2[k, j]; gj = gr3[j]
            gW2 += gj*hkj
            d = gj*W2k*(1 - hkj*hkj)
            gb += d; gW1 += d*x[1, j]
        end
        grad[9+k] = gW1; grad[12+k] = gb; grad[15+k] = gW2
    end

    return grad
end

function get_grad_gamma_alt!(cache::GradientCache, m, beta, gamma, Z_proto, y)
    cfg = cache.cfg
    
    # scalar loss (log-likelihood) over gamma, reusing scratch
    function llf(p)
        Z_temp, v = _buffers!(cache, ForwardDiff.value.(Z_proto), y, p)
        update_factor_loadings!(m, p, Z_temp)  # writes into Z_temp
        # Promote y to dual type to track derivatives
        y_dual = similar(v)
        y_dual .= y
        
        # Solve for beta (more stable than inv)
        beta_temp = Z_temp \ y_dual
        mul!(v, Z_temp, beta_temp)                  # v = Z_temp * beta_temp
        @. v = y_dual - v
        return -dot(v, v)
    end

    res = DiffResults.GradientResult(gamma)
    ForwardDiff.gradient!(res, llf, gamma, cfg)
    return DiffResults.gradient(res)
end

function get_loss(model::AbstractYieldFactorModel, data::Matrix{T}; K::Int=1 ) where T<:Real
    base = model.base
    nobs = size(data, 2)
    cache = initialize_filter(model)

    mse = T(0.0)
    
    catched_params = similar(get_params(model))
    pred = similar(data[:,1])
    v = similar(data[:,1])

    for k in 0:K-1
        catch_point = Int(floor(nobs*((0.25) + 0.75*(k)/K)))
        if k > 1
            set_params!(model, catched_params) 
        end 
        for t in 1:nobs-1
            @views pred .= filter(model, data[:, t], cache)
            @views v .= data[:, t+1] .- pred

           
            mse -= dot(v, v)

            if isinf(mse) || isnan(mse)
                return -Inf
            end
        
            if t == catch_point
                catched_params = copy(get_params(model))
            end
        end
    end

    return mse/base.N/nobs/K
end

# Multi-initialization one-step-ahead loss for score-driven models.
_default_num_inits(::AbstractMSEDrivenModel) = 3
_default_num_inits(::AbstractλMSEDrivenModel) = 1

function get_loss(model::AbstractMSEDrivenModel, data::Matrix{T}; K::Int=1, num_inits::Int=_default_num_inits(model)) where T<:Real
    # Concentrated path: solve (delta, Phi) in closed form (neural anchored model
    # only). Additive — only taken when run(...; concentrate=true) set CONCENTRATE[].
    if CONCENTRATE[] && model isa AbstractNeuralMSEDrivenModel && model.transform_bool
        return get_loss_concentrated(model, data; num_inits=num_inits)
    end
    base = model.base
    nobs = size(data, 2)

    # Score state implied by the current parameters (set in set_params!).
    gamma0 = copy(base.gamma)
    # Score-state trajectory of the most recent run; re-seed source for runs > 1.
    gamma_hist = Matrix{eltype(base.gamma)}(undef, length(base.gamma), nobs)

    # Re-seed period range (matches DNS: nobs/5 .. nobs/2), guarded for short samples.
    lo = max(1, div(nobs, 5))
    hi = max(lo, div(nobs, 2))

    pred = similar(data[:, 1])
    v    = similar(data[:, 1])

    mse = T(0.0)
    for init in 1:num_inits
        # (Re)initialize the score state, then refresh loadings and the filter cache.
        if init == 1
            base.gamma .= gamma0
        else
            ridx = rand(Random.MersenneTwister(42 + init), lo:hi)
            @views base.gamma .= gamma_hist[:, ridx]
        end
        update_factor_loadings!(model, base.gamma, base.Z)
        cache = initialize_filter(model)

        for t in 1:nobs-1
            @views gamma_hist[:, t] .= base.gamma     # score state used to predict t+1
            @views pred .= filter(model, data[:, t], cache)
            @views v .= data[:, t+1] .- pred
            mse -= dot(v, v)
            if isinf(mse) || isnan(mse)
                return -Inf
            end
        end
    end

    return mse / base.N / nobs / num_inits / K
end

function get_loss_array(model::AbstractYieldFactorModel, data::Matrix{T}; K::Int=1 ) where T<:Real
    base = model.base
    nobs = size(data, 2)
    cache = initialize_filter(model)

    mse = Vector{T}(undef, nobs -1)
    fill!(mse, 0.0)
    
    catched_params = similar(get_params(model))
    pred = similar(data[:,1])
    v = similar(data[:,1])

    @inbounds for k in 0:K-1
        catch_point = Int(floor(nobs*((0.25) + 0.75*(k)/K)))
        if k > 1
            set_params!(model, catched_params) 
        end 
        for t in 1:nobs-1
            @views pred .= filter(model, data[:, t], cache)
            @views @. v = data[:, t+1] - pred

            mse[t] -= dot(v, v)

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


function predict(model::AbstractYieldFactorModel, data::Matrix{T}; K::Int=3) where T<:Real
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

