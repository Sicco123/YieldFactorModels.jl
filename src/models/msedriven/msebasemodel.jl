### YieldFactorModels Base
abstract type AbstractYieldFactorModel end
abstract type AbstractMSEDrivenModel <: AbstractYieldFactorModel end
abstract type AbstractNeuralMSEDrivenModel <: AbstractMSEDrivenModel end
abstract type AbstractλMSEDrivenModel <: AbstractMSEDrivenModel end


struct MSEDrivenBaseModel{Fl<:Real,Fβ<:Real,Fγ<:Real} <: AbstractYieldFactorModel
    maturities::Vector{Fl}
    N::Int
    M::Int
    L::Int

    Z::Matrix{Fβ}
    beta::Vector{Fβ}

    Phi::Matrix{Fβ}
    delta::Vector{Fβ}
    mu::Vector{Fβ}

    duplicator::Vector{Int}
    gamma::Vector{Fγ}
    A::Vector{Fγ}
    B::Vector{Fγ}
    omega::Vector{Fγ}
    nu::Vector{Fγ}

    transformations::Vector{Function}
    untransformations::Vector{Function}

    A_guesses::Vector{Fl}
    B_guesses::Vector{Fl}

    init_folder::String
    results_folder::String
    model_string::String

    # Optional gradient scaling parameters. When `scale_grad` is true, the
    # filter uses an exponentially weighted moving average (EWMA) of the
    # squared gradient to scale updates on `gamma`. `forget_factor`
    # governs the decay rate of the EWMA. `grad_EWMA` accumulates the
    # second moment estimate, and `grad_EWMA_count` tracks the number
    # of updates for bias correction. These fields incur no overhead
    # when `scale_grad` is false.
    scale_grad::Bool
    forget_factor::Fl
    grad_EWMA::Vector{Fγ}
    grad_EWMA_count::Vector{Int}

    # inner constructor (OK to use `new` here). Accepts optional
    # `scale_grad` and `forget_factor` keyword arguments for gradient
    # scaling behaviour. Defaults disable scaling.
    function MSEDrivenBaseModel{Fl}(
        maturities::Vector{Fl}, N::Int, M::Int, L::Int,
        duplicator::Vector{Int}, random_walk::Bool,
        specific_transformations::Vector{Function},
        specific_untransformations::Vector{Function},
        A_guesses::Vector{Fl}, B_guesses::Vector{Fl}, model_string; results_location::String = "results/",
        scale_grad::Bool = false,
        forget_factor::Fl = Fl(0.9)
    ) where {Fl<:Real}

        Z     = ones(Fl, N, M)
        beta  = zeros(Fl, M)
        Phi   = zeros(Fl, M, M)
        delta = zeros(Fl, M)
        mu    = zeros(Fl, M)

        gamma = zeros(Fl, L)
    

        A = ones(Fl, L)
        B = random_walk ? zeros(Fl, 0) : ones(Fl, L)   # decide policy
        omega = zeros(Fl, L)
        nu    = zeros(Fl, L)

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
        results_folder = results_location #"YieldFactorModels.jl/results/$(model_string)/"

        # Allocate EWMA accumulator and update counter for gradient scaling
        grad_EWMA = zeros(Fl, L)
        grad_count = [0]
        return new{Fl,Fl,Fl}(maturities, N, M, L, Z, beta, Phi, delta, mu, duplicator,
                              gamma, A, B, omega, nu,
                              transformations, untransformations,
                              A_guesses, B_guesses,
                              init_folder, results_folder, model_string,
                              scale_grad, forget_factor, grad_EWMA, grad_count)
    end

    # Full-fields inner constructor (lets you call with fields directly)
    MSEDrivenBaseModel{Fl,Fβ,Fγ}(
        maturities::Vector{Fl}, N::Int, M::Int, L::Int,
        Z::Matrix{Fβ}, beta::Vector{Fβ},
        Phi::Matrix{Fβ}, delta::Vector{Fβ}, mu::Vector{Fβ},
        duplicator::Vector{Int},
        gamma::Vector{Fγ}, A::Vector{Fγ}, B::Vector{Fγ}, omega::Vector{Fγ}, nu::Vector{Fγ},
        transformations::Vector{Function},
        untransformations::Vector{Function},
        A_guesses::Vector{Fl}, B_guesses::Vector{Fl},
        init_folder::String, results_folder::String, model_string::String,
        scale_grad::Bool, forget_factor::Fl, grad_EWMA::Vector{Fγ}, grad_EWMA_count::Vector{Int}
    ) where {Fl<:Real,Fβ<:Real,Fγ<:Real} =
    new{Fl,Fβ,Fγ}(maturities, N, M, L, Z, beta, Phi, delta, mu, duplicator,
                   gamma, A, B, omega, nu,
                   transformations, untransformations,
                   A_guesses, B_guesses,
                   init_folder, results_folder, model_string,
                   scale_grad, forget_factor, grad_EWMA, grad_EWMA_count)
end


# outer helper that builds a new instance (NO `new` here)
function build(model::MSEDrivenBaseModel{Fl,Fl2,Fl3},
               Z::AbstractMatrix{Fβ}, beta::AbstractVector{Fβ},
               gamma::AbstractVector{Fγ}, Phi::AbstractMatrix{Fβ},
               delta::AbstractVector{Fβ}, mu::AbstractVector{Fβ},
               A::AbstractVector{Fγ}, B::AbstractVector{Fγ},
               omega::AbstractVector{Fγ}, nu::AbstractVector{Fγ}
) where {Fl<:Real,Fβ<:Real,Fγ<:Real, Fl2<:Real, Fl3<:Real}
    # convert grad into Fγ
    grad_EWMA = convert(Vector{Fγ}, model.grad_EWMA)
    # Preserve gradient scaling state when rebuilding a base model from new
    # parameter arrays. Pass through `scale_grad`, `forget_factor`,
    # `grad_EWMA`, and `grad_EWMA_count` to the full-field constructor.
    return MSEDrivenBaseModel{Fl,Fβ,Fγ}(
        # use the *field constructor* with all fields in order:
        model.maturities, model.N, model.M, model.L,
        Z, beta, Phi, delta, mu, model.duplicator,
        gamma, A, B, omega, nu,
        model.transformations, model.untransformations,
        model.A_guesses, model.B_guesses,
        model.init_folder, model.results_folder, model.model_string,
        model.scale_grad, model.forget_factor, grad_EWMA, model.grad_EWMA_count
    )
end

function get_param_groups(model::AbstractMSEDrivenModel, param_groups::Vector{String})
    base = model.base
    length_total_params = length(get_params(model))
    if length(param_groups) == length_total_params
        return param_groups
    else
        println("Default param groups assigned.")
        return vcat(fill("1", length_total_params - base.M*(base.M +1)), fill("2", base.M*(base.M +1)))
    end
end


