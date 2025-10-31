### MSEDλModel - Lambda Parameter Based MSED Model

struct RandomWalkModel{Fl <: Real, Fβ <: Real, Fγ <: Real} <: AbstractRandomWalkModel
    # Base
    base::StaticBaseModel{Fl, Fβ, Fγ}

    # λ parameter
    last_y::Vector{Fl}

    # Constructor
    function RandomWalkModel{T}(maturities::Vector{T}, N::Int, M::Int;  model_string::String = "NS", results_location::String = "results/") where T<:Real

        specific_transformations = Function[]
        specific_untransformations = Function[]

        # append identity transformations for the lambda intercept
        append!(specific_transformations, fill(identity, 1))
        append!(specific_untransformations, fill(identity, 1))

        L = 1  # Lambda model has 1 factor loading parameter

        # Create base model
        base = StaticBaseModel{T}(maturities, N, M, L, specific_transformations, specific_untransformations, model_string; results_location=results_location)

        last_y = Vector{T}(undef, N)

        new{T, T, T}(base, last_y)
    end

    RandomWalkModel{T}(base::StaticBaseModel{T, Fβ, Fγ}, last_y::Vector{T}) where {T<:Real, Fβ<:Real, Fγ<:Real} = new{T, Fβ, Fγ}(base, last_y)

end

function build(model::AbstractRandomWalkModel, Z::Matrix{Fγ}, beta::Vector{Fβ}, gamma::Vector{Fγ}, Phi::Matrix{Fβ}, delta::Vector{Fβ}, mu::Vector{Fβ}) where {Fβ<:Real, Fγ<:Real}
    base = build(model.base, Z, beta, gamma, Phi, delta, mu)
    return RandomWalkModel{typeof(base.maturities[1])}(
       base, model.last_y
    )
end

function get_static_model_type(model::AbstractRandomWalkModel)
    return ""
end


function update_factor_loadings!(model::AbstractRandomWalkModel, gamma, Z)
   
    return nothing
end