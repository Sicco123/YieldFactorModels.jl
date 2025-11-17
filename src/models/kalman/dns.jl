### DNSModel - Lambda Parameter Based MSED Model

struct DNSModel{Fl <: Real, Fβ <: Real} <: AbstractDNSModel
    # Base
    base::KalmanBaseModel{Fl, Fβ}

    # λ parameter
    lambda::Fl

    # Constructor
    function DNSModel{T}(maturities::Vector{T}, N::Int, M::Int;
                           model_string::String = "DNS",
                           results_location::String = "results/") where T<:Real

        specific_transformations = Function[]
        specific_untransformations = Function[]

        L = 1  # Lambda model has 1 observation driven parameter
        
        # append identity transformations for the lambda intercept 
        append!(specific_transformations, fill(identity, 1))
        append!(specific_untransformations, fill(identity, 1))

        # Create base model, passing optional gradient scaling parameters
        base = KalmanBaseModel{T}(maturities, N, M, L, 
                                     specific_transformations, specific_untransformations,
                                      model_string;
                                     results_location=results_location)

        lambda = T(0.0)

        new{T, T}(base, lambda)
    end

    DNSModel{T}(base::KalmanBaseModel{T, Fβ}, lambda::T) where {T<:Real, Fβ<:Real} = new{T, Fβ}(base, lambda)

end

function build(model::AbstractDNSModel, Z::Matrix{Fβ}, beta::Vector{Fβ}, gamma::Vector{Fβ}, Phi::Matrix{Fβ}, delta::Vector{Fβ}, mu::Vector{Fβ}, A::Vector{Fβ}, B::Vector{Fβ}, omega::Vector{Fβ}, nu::Vector{Fβ}) where {Fβ<:Real}
    base = build(model.base, Z, beta, gamma, Phi, delta, mu, A, B, omega, nu)
    return DNSModel{typeof(base.maturities[1])}(
       base, model.lambda
    )
end

function get_static_model_type(model::AbstractDNSModel)
    return "DNS"
end


function update_factor_loadings!(model::AbstractDNSModel, gamma, Z)
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