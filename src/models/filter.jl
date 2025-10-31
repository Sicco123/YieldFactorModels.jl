# Cache struct for gradient computation - don't preallocate Dual arrays
struct GradientCache{C}
    cfg::C
    Ztemp_any::Base.RefValue{Any}   # will hold Matrix{<:Real or <:Dual}
    v_any::Base.RefValue{Any}       # will hold Vector{<:Real or <:Dual}
end

function GradientCache(gamma::AbstractVector{T}) where T
    cfg = ForwardDiff.GradientConfig(nothing, gamma)
    return GradientCache{typeof(cfg)}(cfg, Ref{Any}(nothing), Ref{Any}(nothing))
end



function initialize_filter(model::AbstractYieldFactorModel)
    base = model.base
    if hasfield(typeof(base), :grad_EWMA)
       base.grad_EWMA .= zeros(eltype(base.gamma), length(base.gamma))
       base.grad_EWMA_count .= [0]
    end
    return GradientCache(base.gamma)
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

function filter(m::AbstractMSEDrivenModel, y::Vector{T}, cache) where T<:Real

    if isnan(y[1])
        if !isempty(m.base.B)
            m.base.gamma .= m.base.nu .+ m.base.B .* m.base.gamma
            update_factor_loadings!(m, m.base.gamma, m.base.Z)
        end
        m.base.beta .= m.base.mu  .+ m.base.Phi * m.base.beta
        return m.base.Z * m.base.beta
    end
    # 1) Get beta OLS
    try
        get_β_OLS!(m.base.beta, m.base.Z, y)
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
        get_β_OLS!(m.base.beta, m.base.Z, y)
    catch e
        println("Error in OLS calculation: ", e)
        return -Inf
    end

    # 4) Predictions 
    if !isempty(m.base.B)
        m.base.gamma .= m.base.nu .+ m.base.B .* m.base.gamma
        update_factor_loadings!(m, m.base.gamma, m.base.Z)
    end
    m.base.beta .= m.base.mu  .+ m.base.Phi * m.base.beta

    return m.base.Z * m.base.beta
end

function filter(m::AbstractStaticModel, y::Vector{T}, cache) where T<:Real

    if isnan(y[1])
        m.base.beta .= m.base.mu  .+ m.base.Phi * m.base.beta
        return m.base.Z * m.base.beta
    end
    # 1) Get beta OLS
    try
        get_β_OLS!(m.base.beta, m.base.Z, y)
    catch e
        println("Error in OLS calculation: ", e)
        return -Inf
    end

    # 4) Predictions 
    m.base.beta .= m.base.mu  .+ m.base.Phi * m.base.beta
    return m.base.Z * m.base.beta
end

function filter(m::AbstractRandomWalkModel, y::Vector{T}, cache) where T<:Real
    if isnan(y[1])
        return copy(m.last_y)
    end

    m.last_y .= y

    return copy(m.last_y)
end

function get_β_OLS!(beta, Z, y)
    try
        beta .= cholesky(Z'Z)\(Z'y) #(Z' * Z) \ (Z' * y)
    catch e
        M = size(Z, 2)
        beta .= ((Z' * Z) + 1e-3 * I(M)) \ (Z' * y)
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

function get_grad_gamma!(cache::GradientCache, m, beta, gamma, Z_proto, y)
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

function get_loss(model::AbstractYieldFactorModel, data::Matrix{T}) where T<:Real
    base = model.base
    nobs = size(data, 2)
    cache = initialize_filter(model)

    mse = 0.0

    for t in 1:nobs-1
        pred = filter(model,  data[:, t], cache)
        v = data[:, t+1] .- pred
        
        mse -= dot(v, v)
        
        if isinf(mse) || isnan(mse)
            return -Inf
        end
    end

    return mse/base.N/nobs
end


function predict(model::AbstractYieldFactorModel, data::Matrix{T}) where T<:Real
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