### TVλDNSModel - Lambda Parameter Based MSED Model

struct TVλDNSModel{Fl <: Real, Fβ <: Real} <: AbstractTVλDNSModel
    # Base
    base::KalmanBaseModel{Fl, Fβ}

    # λ parameter
    lambda::Vector{Fl}
    z_i::Vector{Fl}
    tau_maturities::Vector{Fl}
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

        lambda = fill(T(0.0), 1)
        z_i = fill(T(0.0), N)
        tau_maturities = fill(T(0.0), N)


        new{T, T}(base, lambda, z_i, tau_maturities)
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


function update_factor_loadings!(model::AbstractTVλDNSModel, gamma::AbstractVector{R}, Z::AbstractMatrix{R}) where {R<:Real}
    # Extract lambda parameter
    
    model.lambda .= R(1e-2) .+ exp.(gamma)

    model.tau_maturities .= model.lambda .* model.base.maturities
    model.z_i .= exp.(-model.tau_maturities)

    Z[:, 2] .= (1 .- model.z_i) ./ model.tau_maturities
    Z[:, 3] .= Z[:, 2] .- model.z_i
    return 
end