### TVλDNSModel - Lambda Parameter Based MSED Model

struct TVλDNSModel{Fl <: Real, Fβ <: Real} <: AbstractTVλDNSModel
    # Base
    base::KalmanBaseModel{Fl, Fβ}

    # λ parameter
    lambda::Vector{Fl}
    z_i::Vector{Fl}
    tau_maturities::Vector{Fl}
    # Unconditional state covariance Γ, the parametrization's own primitive.
    # Kept because it IS the filter's initial P — see initialize_filter.
    Gamma::Matrix{Fl}
    # Constructor
    function TVλDNSModel{T}(maturities::Vector{T}, N::Int, M::Int;
                           model_string::String = "TVλDNS",
                           results_location::String = "results/") where T<:Real

        specific_transformations = Function[]
        specific_untransformations = Function[]

        L = 1  # Lambda model has 1 observation driven parameter
        
    

        # Create base model, passing optional gradient scaling parameters
        base = KalmanBaseModel{T}(maturities, N, M+1, L, 
                                     specific_transformations, specific_untransformations,
                                      model_string;
                                     results_location=results_location)

        nstates = M + 1
        base.transformations[end-nstates^2+1:end] .= identity
        base.untransformations[end-nstates^2+1:end] .= identity


        ncov = nstates * (nstates + 1) ÷ 2
        cov_slots = (lastindex(base.transformations) - nstates^2 - nstates - ncov + 1):(lastindex(base.transformations) - nstates^2 - nstates)
        base.transformations[cov_slots] .= identity
        base.untransformations[cov_slots] .= identity

        lambda = fill(T(0.0), 1)
        z_i = fill(T(0.0), N)
        tau_maturities = fill(T(0.0), N)
        Gamma = zeros(T, nstates, nstates)


        new{T, T}(base, lambda, z_i, tau_maturities, Gamma)
    end

    TVλDNSModel{T}(base::KalmanBaseModel{T, Fβ}, lambda::T) where {T<:Real, Fβ<:Real} = new{T, Fβ}(base, lambda)

end

function build(model::AbstractTVλDNSModel, Z::Matrix{Fβ}, beta::Vector{Fβ}, gamma::Vector{Fβ}, Phi::Matrix{Fβ}, delta::Vector{Fβ}, mu::Vector{Fβ}, A::Vector{Fβ}, B::Vector{Fβ}, omega::Vector{Fβ}, nu::Vector{Fβ}) where {Fβ<:Real}
    base = build(model.base, Z, beta, gamma, Phi, delta, mu, A, B, omega, nu)
    return TVλDNSModel{typeof(base.maturities[1])}(
       base, model.lambda
    )
end

function get_static_model_type(model::AbstractTVλDNSModel)
    return "1C"
end


const TVΛ_MIN = 0.01
tvλ_link(γ::R) where {R<:Real} = R(TVΛ_MIN) + exp(γ)

# dλ/dγ = exp(γ) = λ - TVΛ_MIN, written in λ so the caller need not keep γ around.
dtvλ_link(λ::R) where {R<:Real} = λ - R(TVΛ_MIN)


function get_param_groups(model::AbstractTVλDNSModel, param_groups::Vector{String})
    n = length(get_params(model))
    length(param_groups) == n && return param_groups

    ns = model.base.M                        # 4 states: 3 yield factors + λ
    ncov = ns * (ns + 1) ÷ 2
    A0 = 1 + ncov + ns                       # last slot before A's block

    groups = pin_diagonal_groups!(model, fill("2", n))
    # Everything initialize_with_static_params copies over from the 1C fit stays put:
    # the observation variance (slot 1) and the three yield-factor means μ[1:ns-1].
    # groups[1] = "-1"
    # groups[(1 + ncov + 1):(1 + ncov + ns - 1)] .= "-1"  # μ[1:ns-1]
    # groups[(1 + ncov - ns + 1):(ncov)] .= "-1"          # Γ[1:ns-1, ns]
    # for i in 1:ns-1
    #     groups[A0 + ns * (i - 1) + ns] = "-1"           # A[i, ns]
    #     groups[A0 + ns * (ns - 1) + i] = "-1"           # A[ns, i]
    # end

    #println("TVλ param groups: λ couplings pinned at init (slots $(findall(==("-1"), groups))), rest group \"2\" (LBFGS).")
    return groups
end


function trace_callback(model::AbstractTVλDNSModel, data, show_every::Int)
    return function (os)
        iter = os isa AbstractArray ? length(os) : os.iteration
        if iter % show_every == 0
            rr = try
                rrmse_vs_rw(model, data)
            catch
                NaN
            end
            #println("            RRMSE vs RW: ", round(rr; digits = 4))
        end
        return false
    end
end

function update_factor_loadings!(model::AbstractTVλDNSModel, gamma::AbstractVector{R}, Z::AbstractMatrix{R}) where {R<:Real}
    model.lambda .= tvλ_link.(gamma)

    model.tau_maturities .= model.lambda .* model.base.maturities
    model.z_i .= exp.(-model.tau_maturities)

    Z[:, 2] .= (1 .- model.z_i) ./ model.tau_maturities
    Z[:, 3] .= Z[:, 2] .- model.z_i
    return 
end