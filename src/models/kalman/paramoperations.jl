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
    if is_diagonal(base)
        base.Phi .= Diagonal(base.Phi)
        # Keep the stored vector in step with the model, so get_params, natural_params
        # and every file written from them show the zeros the filter actually used.
        @views base.flat_params[end-M*M+1:end] .= vec(permutedims(base.Phi))
    end
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

    # Φ = L Mk L⁻¹ with L = chol(Γ), so a diagonal A alone does not make Φ diagonal —
    # Γ has to be diagonal too. The independent-factor TVλ therefore pins chol(Γ)'s
    # off-diagonals as well, which also decouples the λ state (the fourth) from the
    # three yield factors. Pinning the stored slots rather than the matrices keeps
    # get_params and the init file it writes telling the same story.
    if is_diagonal(model)
        params = pin_slots!(collect(params), chol_offdiag_slots(model.base.M))
    end

    model.base.flat_params .= params

    set_params!(model.base, params)

    base = model.base
    A = copy(base.Phi)
    Γ = model.Gamma
    Γ .= base.Omega_state                       # base's set_params! left Γ = U'U here
    C = cholesky(Symmetric(Γ); check = false)
    if !issuccess(C)
        fill!(base.Phi, NaN)
        return
    end
    L = C.L
    B = cholesky(Symmetric(I + A * A')).L

    Mk = A' / B'                                # = L⁻¹ Φ L, with ‖Mk‖₂ < 1
    base.Phi .= L * Mk / L
    Σ = L * (I - Mk * Mk') * L'
    base.Omega_state .= (Σ .+ Σ') ./ 2          # drop the asymmetry rounding adds
    base.delta .= (I - base.Phi) * base.delta
    return
end

function reconcile_init_params(model::AbstractTVλDNSModel, all_params::AbstractMatrix)
    n = model.base.M
    out = copy(all_params)
    Ex = reverse(Matrix{eltype(out)}(I, n, n), dims = 2)   # exchange matrix

    for col in axes(out, 2)
        p = @view out[:, col]

        Γ = zeros(eltype(p), n, n)
        k = 2
        for j in 1:n, i in 1:j
            Γ[i, j] = Γ[j, i] = p[k]
            k += 1
        end
        Φ = permutedims(reshape(p[k+n:k+n+n*n-1], n, n))

        CΓ = cholesky(Symmetric(Γ); check = false)
        issuccess(CΓ) || error("TVλ init column $col: Γ (rows 2:$(k-1)) is not positive definite")

        P = CΓ.L \ Φ * CΓ.L
        CS = cholesky(Symmetric(I - P'P); check = false)
        # Φ and Γ are not free of each other: the implied innovation covariance
        # Σ = Γ - ΦΓΦ' = L(I - PP')L' has to be positive definite, which caps how
        # persistent Φ may be given Γ. Stationarity alone is not enough.
        issuccess(CS) || error("TVλ init column $col: Σ = Γ - ΦΓΦ' is not positive definite — Φ is too persistent for this Γ")
        Ub = Ex * cholesky(Symmetric(Ex * inv(CS) * Ex)).L * Ex   # upper triangular, Ub*Ub' = (I - P'P)⁻¹
        A = permutedims(P * Ub)

        k = 2
        for j in 1:n, i in 1:j
            p[k] = CΓ.U[i, j]
            k += 1
        end
        p[k+n:k+n+n*n-1] .= vec(permutedims(A))
    end
    return out
end

function natural_params(model::AbstractTVλDNSModel)
    base = model.base
    n = base.M
    p = copy(base.flat_params)

    k = 2
    for j in 1:n, i in 1:j
        p[k] = model.Gamma[i, j]        # Γ = U'U, upper triangle column-major
        k += 1
    end
    p[k+n:k+n+n*n-1] .= vec(permutedims(base.Phi))   # Φ itself, row-major, not A
    return p
end

# Initialize with static parameters (for factor forecasting)
function initialize_with_static_params!(model::AbstractKalmanModel, static_params::AbstractVector)
    # placeholder
    a = 1
end



# TVλ is initialized purely from its own init CSV; it is deliberately NOT seeded from
# 1C's fitted observation variance and factor mean the way the MSE-driven models are.
initialize_with_static_params(::AbstractTVλDNSModel, params::AbstractVector, ::AbstractMatrix) = params


function get_new_initial_params(model::AbstractKalmanModel, params::AbstractVector, trial::Int)
    # placeholder
    Random.seed!(trial)

    params .= randn(length(params)) 
end