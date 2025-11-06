### StaticNeuralModel - Neural Network Based Static Model

mutable struct StaticNeuralModel{Fl <: Real, Fβ <: Real, Fγ <: Real} <: AbstractStaticNeuralModel
    # Base
    base::StaticBaseModel{Fl, Fβ, Fγ} 
    
    # Neural Networks    
    net1::Chain
    net2::Chain

    net1_dual::Chain
    net2_dual::Chain

    # Net input 
    net_input::Matrix{Fl}

    transform_bool::Bool
    


    # Constructor
    function StaticNeuralModel{T}(maturities::Vector{T}, N::Int, M::Int; net_size = 3, activation_func = tanh, bias_bool = false, model_string::String = "NNS", results_location::String = "results/", transform_bool::Bool = true) where T<:Real
        
        specific_transformations = Function[]
        specific_untransformations = Function[]

        # Calculate L 
        L = 3 * net_size * 2
        
        # Initialize neural networks
        net1 = Chain(Dense(1, net_size, activation_func), Dense(net_size, 1; bias = bias_bool))
        net2 = Chain(Dense(1, net_size, activation_func), Dense(net_size, 1; bias = bias_bool))
        if T == Float32
            net1 = f32(net1)
            net2 = f32(net2)
        else 
            net1 = f64(net1)
            net2 = f64(net2)
        end

        _, build_net1 = Flux.destructure(net1)
        _, build_net2 = Flux.destructure(net2)

        dual_params = ForwardDiff.Dual{Nothing, T, 18}.(randn(T, 18))
        net1_dual = build_net1(dual_params[1:9])
        net2_dual = build_net2(dual_params[10:18])
        
        # append identity transformations for the net params 
        append!(specific_transformations, fill(identity, net_size*3*2))
        append!(specific_untransformations, fill(identity, net_size*3*2))
        
        # Create base model
        base = StaticBaseModel{T}(maturities, N, M, L, specific_transformations, specific_untransformations, model_string; results_location=results_location)

        # Pre-allocate net input
        net_input = Matrix{T}(undef, 1, N)
        for i in 1:N
            net_input[1, i] = maturities[i]
        end

        # transform_bool
        

        new{T, T, T}(base, net1, net2, net1_dual, net2_dual, net_input, transform_bool)
    end

    # Full-fields inner constructor (lets you call with fields directly)
    StaticNeuralModel{Fl}(base::StaticBaseModel{Fl,Fβ,Fγ}, net1::Chain, net2::Chain,
               net1_dual::Chain, net2_dual::Chain, net_input::Matrix{Fl}, transform_bool::Bool) where {Fl<:Real, Fβ<:Real, Fγ<:Real} =
    new{Fl, Fβ, Fγ}(base, net1, net2, net1_dual, net2_dual, net_input, transform_bool)
end

function build(model::StaticNeuralModel, Z::Matrix{Fγ}, beta::Vector{Fβ}, gamma::Vector{Fγ}, Phi::Matrix{Fβ}, delta::Vector{Fβ}, mu::Vector{Fβ}) where {Fβ<:Real, Fγ<:Real}
    base = build(model.base, Z, beta, gamma, Phi, delta, mu)
    return StaticNeuralModel{typeof(base.maturities[1])}(
       base, model.net1, model.net2, model.net1_dual, model.net2_dual, model.net_input, model.transform_bool
    )
end

function get_static_model_type(model::AbstractStaticNeuralModel)
    if model.transform_bool
        return "NNS"
    else 
        return "NNS-Anchored"
    end
end


# Simpler, more idiomatic approach
@inline function _get_networks(model::AbstractStaticNeuralModel, ::Type{T}) where T
    if T <: AbstractFloat
        return model.net1, model.net2
    else  # Dual
        return model.net1_dual, model.net2_dual
    end
end

@inline function update_factor_loadings!(
    model::AbstractStaticNeuralModel, 
    gamma::AbstractVector{T}, 
    Z::AbstractMatrix{R}
) where {T <: Real, R <: Real}
    
    # Direct type dispatch - simpler and equally efficient
    net1, net2 = _get_networks(model, T)
    
    # Pre-compute views to avoid repeated indexing
    gamma1 = @view gamma[1:9]
    gamma2 = @view gamma[10:18]
    z2 = @view Z[:, 2]
    z3 = @view Z[:, 3]
    
    # Load parameters
    loadθ!(gamma1, net1)
    loadθ!(gamma2, net2)
    
    # Set first column to ones if needed (more robust check)
    if Z[1, 1] != one(T)
        Z[:, 1] .= one(T)
    end
    
    # Transform
    transform_net_1!(z2, net1, model.net_input, Val(model.transform_bool))
    transform_net_2!(z3, net2, model.net_input, Val(model.transform_bool))

    return nothing
end