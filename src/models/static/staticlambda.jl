### MSEDλModel - Lambda Parameter Based MSED Model

struct StaticλModel{Fl <: Real, Fβ <: Real, Fγ <: Real} <: AbstractStaticλModel
    # Base
    base::StaticBaseModel{Fl, Fβ, Fγ}

    # λ parameter
    lambda::Fl

    # Constructor
    function StaticλModel{T}(maturities::Vector{T}, N::Int, M::Int;  model_string::String = "NS") where T<:Real

        specific_transformations = Function[]
        specific_untransformations = Function[]

        # append identity transformations for the lambda intercept
        append!(specific_transformations, fill(identity, 1))
        append!(specific_untransformations, fill(identity, 1))

        L = 1  # Lambda model has 1 factor loading parameter

        # Create base model
        base = StaticBaseModel{T}(maturities, N, M, L, specific_transformations, specific_untransformations, model_string)

        lambda = T(0.0)

        new{T, T, T}(base, lambda)
    end

    StaticλModel{T}(base::StaticBaseModel{T, Fβ, Fγ}, lambda::T) where {T<:Real, Fβ<:Real, Fγ<:Real} = new{T, Fβ, Fγ}(base, lambda)

end

function build(model::AbstractStaticλModel, Z::Matrix{Fγ}, beta::Vector{Fβ}, gamma::Vector{Fγ}, Phi::Matrix{Fβ}, delta::Vector{Fβ}, mu::Vector{Fβ}) where {Fβ<:Real, Fγ<:Real}
    base = build(model.base, Z, beta, gamma, Phi, delta, mu)
    return StaticλModel{typeof(base.maturities[1])}(
       base, model.lambda
    )
end

function get_static_model_type(model::AbstractStaticλModel)
    return "NS"
end


function update_factor_loadings!(model::AbstractStaticλModel, gamma, Z)
    # Extract lambda parameter
    R = eltype(gamma)

    λ = R(1e-2) .+ exp.(gamma)
    tau_maturities = λ .* model.base.maturities
    z_i = exp.(-tau_maturities)

    Z[:, 1] .= R(1.0)
    Z[:, 2] .= (1 .- z_i) ./ tau_maturities
    Z[:, 3] .= Z[:, 2] .- z_i

    return nothing
end