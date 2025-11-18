### YieldFactorModels Base
abstract type AbstractNeuralNetworkModel <: AbstractYieldFactorModel end


struct NeuralNetworkModel{Fl<:Real,Fn, Fs, Fp} <: AbstractYieldFactorModel
    maturities::Vector{Fl}
    N::Int
    
    net 
    states 
    params

    transformations::Vector{Function}
    untransformations::Vector{Function}

    init_folder::String
    results_folder::String
    model_string::String

    # inner constructor (OK to use `new` here)
    function NeuralNetworkModel{Fl}(
        maturities::Vector{Fl}, N::Int,  hidden_layer_sizes::Vector{Int},
        model_string;
        results_location::String = "results/"
    ) where {Fl<:Real}

        
        if length(hidden_layer_sizes) == 0
            error("hidden_layer_sizes must contain at least one hidden layer size.")
        end
        
        if length(hidden_layer_sizes) == 1
            net = Chain(
                Dense(N, hidden_layer_sizes[1], relu), BatchNorm(hidden_layer_sizes[1]),
                Dense(hidden_layer_sizes[1], N)
            )
        else 
            net = Chain(
            Dense(N, hidden_layer_sizes[1], relu), [
                Dense(hidden_layer_sizes[i], hidden_layer_sizes[i+1], relu) for i in 1:length(hidden_layer_sizes)-1
            ]...,
            BatchNorm(hidden_layer_sizes[end]),
            Dense(hidden_layer_sizes[end], N)
        ) 
        end
        
        transformations = [
            
        ]
        untransformations = [
        ]

        init_folder    = "YieldFactorModels.jl/initializations/$(model_string)/"
        results_folder = results_location

        params, states = Lux.setup(Random.default_rng(), net)

        Fn = typeof(net)
        Fs = typeof(states)
        Fp = typeof(params)
        return new{Fl,Fn,Fs,Fp}(maturities, N, net, states, params,
                              transformations, untransformations,
                              init_folder, results_folder, model_string)
    end

    # Full-fields inner constructor (lets you call with fields directly)
    NeuralNetworkModel{Fl,Fβ,Fγ}(
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
function build(model::NeuralNetworkModel{Fl,Fl2,Fl3},
               Z::AbstractMatrix{Fβ}, beta::AbstractVector{Fβ},
               gamma::AbstractVector{Fγ}, Phi::AbstractMatrix{Fβ},
               delta::AbstractVector{Fβ}, mu::AbstractVector{Fβ},
) where {Fl<:Real,Fβ<:Real,Fγ<:Real, Fl2<:Real, Fl3<:Real}

    return NeuralNetworkModel{Fl,Fβ,Fγ}(
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