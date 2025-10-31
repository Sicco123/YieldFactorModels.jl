### MSEDλModel - Lambda Parameter Based MSED Model

struct MSEDλModel{Fl <: Real, Fβ <: Real, Fγ <: Real} <: AbstractλMSEDrivenModel
    # Base
    base::MSEDrivenBaseModel{Fl, Fβ, Fγ}

    # λ parameter
    lambda::Fl

    # Constructor
    function MSEDλModel{T}(maturities::Vector{T}, N::Int, M::Int, random_walk::Bool;
                           model_string::String = "MSEDλ",
                           results_location::String = "results/",
                           scale_grad::Bool = false,
                           forget_factor::T = T(0.98)) where T<:Real

        specific_transformations = Function[from_R_to_pos]
        specific_untransformations = Function[from_pos_to_R]

        if !random_walk
            n = length(specific_transformations)
            append!(specific_transformations, fill(from_R_to_01, n))
            append!(specific_untransformations, fill(from_01_to_R, n))
        end

        A_guesses = T[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        B_guesses = random_walk ? T[] : T[0.9, 0.95, 0.98, 0.99, 0.999]

        L = 1  # Lambda model has 1 observation driven parameter
        duplicator = collect(1:L)  # No sharing for lambda model
        
        # append identity transformations for the lambda intercept 
        append!(specific_transformations, fill(identity, 1))
        append!(specific_untransformations, fill(identity, 1))

        # Create base model, passing optional gradient scaling parameters
        base = MSEDrivenBaseModel{T}(maturities, N, M, L, duplicator, random_walk,
                                     specific_transformations, specific_untransformations,
                                     A_guesses, B_guesses, model_string;
                                     scale_grad=scale_grad, forget_factor=forget_factor, results_location=results_location)

        lambda = T(0.0)

        new{T, T, T}(base, lambda)
    end

    MSEDλModel{T}(base::MSEDrivenBaseModel{T, Fβ, Fγ}, lambda::T) where {T<:Real, Fβ<:Real, Fγ<:Real} = new{T, Fβ, Fγ}(base, lambda)

end

function build(model::AbstractλMSEDrivenModel, Z::Matrix{Fγ}, beta::Vector{Fβ}, gamma::Vector{Fγ}, Phi::Matrix{Fβ}, delta::Vector{Fβ}, mu::Vector{Fβ}, A::Vector{Fγ}, B::Vector{Fγ}, omega::Vector{Fγ}, nu::Vector{Fγ}) where {Fβ<:Real, Fγ<:Real}
    base = build(model.base, Z, beta, gamma, Phi, delta, mu, A, B, omega, nu)
    return MSEDλModel{typeof(base.maturities[1])}(
       base, model.lambda
    )
end

function get_static_model_type(model::AbstractλMSEDrivenModel)
    return "NS"
end


function update_factor_loadings!(model::AbstractλMSEDrivenModel, gamma, Z)
    # Extract lambda parameter
    R = eltype(gamma)
    
    λ = R(1e-2) .+ exp.(gamma)
    tau_maturities = λ .* model.base.maturities
    z_i = exp.(-tau_maturities)

    Z[:, 1] .= R(1.0)
    Z[:, 2] .= (1 .- z_i) ./ tau_maturities
    Z[:, 3] .= Z[:, 2] .- z_i

    return 
end