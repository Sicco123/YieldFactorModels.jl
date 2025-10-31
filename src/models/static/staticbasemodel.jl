### YieldFactorModels Base
abstract type AbstractStaticModel <: AbstractYieldFactorModel end
abstract type AbstractStaticNeuralModel <: AbstractStaticModel end
abstract type AbstractStaticλModel <: AbstractStaticModel end
abstract type AbstractRandomWalkModel <: AbstractStaticModel end


struct StaticBaseModel{Fl<:Real,Fβ<:Real,Fγ<:Real} <: AbstractYieldFactorModel
    maturities::Vector{Fl}
    N::Int
    M::Int
    L::Int

    Z::Matrix{Fβ}
    beta::Vector{Fβ}

    Phi::Matrix{Fβ}
    delta::Vector{Fβ}
    mu::Vector{Fβ}

    gamma::Vector{Fγ}
    
    transformations::Vector{Function}
    untransformations::Vector{Function}

    init_folder::String
    results_folder::String
    model_string::String

    # inner constructor (OK to use `new` here)
    function StaticBaseModel{Fl}(
        maturities::Vector{Fl}, N::Int, M::Int, L::Int,
        specific_transformations::Vector{Function},
        specific_untransformations::Vector{Function},
        model_string;
        results_location::String = "results/"
    ) where {Fl<:Real}

        Z     = ones(Fl, N, M)
        beta  = zeros(Fl, M)
        Phi   = zeros(Fl, M, M)
        delta = zeros(Fl, M)
        mu    = zeros(Fl, M)

        gamma = zeros(Fl, L)

        transformations = [
            specific_transformations...,
            fill(identity, 3)...,
            from_R_to_11, fill(identity, 3)...,
            from_R_to_11, fill(identity, 3)...,
            from_R_to_11
        ]
        untransformations = [
            specific_untransformations...,
            fill(identity, 3)...,
            from_11_to_R, fill(identity, 3)...,
            from_11_to_R, fill(identity, 3)...,
            from_11_to_R
        ]

        init_folder    = "YieldFactorModels.jl/initializations/$(model_string)/"
        results_folder = results_location

        return new{Fl,Fl,Fl}(maturities, N, M, L, Z, beta, Phi, delta, mu, gamma,
                              transformations, untransformations,
                              init_folder, results_folder, model_string)
    end

    # Full-fields inner constructor (lets you call with fields directly)
    StaticBaseModel{Fl,Fβ,Fγ}(
        maturities::Vector{Fl}, N::Int, M::Int, L::Int,
        Z::Matrix{Fβ}, beta::Vector{Fβ},
        Phi::Matrix{Fβ}, delta::Vector{Fβ}, mu::Vector{Fβ},
        gamma::Vector{Fγ},transformations::Vector{Function},
        untransformations::Vector{Function},
        init_folder::String, results_folder::String, model_string::String
    ) where {Fl<:Real,Fβ<:Real,Fγ<:Real} =
    new{Fl,Fβ,Fγ}(maturities, N, M, L, Z, beta, Phi, delta, mu,
                   gamma, 
                   transformations, untransformations,
                   init_folder, results_folder, model_string)
end


# outer helper that builds a new instance (NO `new` here)
function build(model::StaticBaseModel{Fl,Fl2,Fl3},
               Z::AbstractMatrix{Fβ}, beta::AbstractVector{Fβ},
               gamma::AbstractVector{Fγ}, Phi::AbstractMatrix{Fβ},
               delta::AbstractVector{Fβ}, mu::AbstractVector{Fβ},
) where {Fl<:Real,Fβ<:Real,Fγ<:Real, Fl2<:Real, Fl3<:Real}

    return StaticBaseModel{Fl,Fβ,Fγ}(
        # use the *field constructor* with all fields in order:
        model.maturities, model.N, model.M, model.L,
        Z, beta, Phi, delta, mu,
        gamma,
        model.transformations, model.untransformations,
        model.init_folder, model.results_folder, model.model_string
    )
end

function get_param_groups(model::AbstractStaticModel, param_groups::Vector{String})
    base = model.base
    length_total_params = length(get_params(model))
    if length(param_groups) == length_total_params
        return param_groups
    else
        println("Default param groups assigned.")
        return vcat(fill("1", length_total_params - base.M*(base.M +1)), fill("2", base.M*(base.M +1)))
    end
end