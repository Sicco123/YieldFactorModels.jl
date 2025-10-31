### StaticNeuralModel - Neural Network Based Static Model

mutable struct StaticNeuralModel{Fl <: Real, Fβ <: Real, Fγ <: Real} <: AbstractStaticNeuralModel
    # Base
    base::StaticBaseModel{Fl, Fβ, Fγ} 
    
    # Neural Networks    
    net1::Chain
    net2::Chain
    build_net1::Any
    build_net2::Any
    
    # Net input 
    net_input::Matrix{Fl}

    # layout 
    layout::NamedTuple

    # dual net cache 
    dual_net_cache::Base.RefValue{Dict{DataType,NamedTuple}}

    # Constructor
    function StaticNeuralModel{T}(maturities::Vector{T}, N::Int, M::Int; net_size = 3, activation_func = tanh, bias_bool = false, model_string::String = "NNS", results_location::String = "results/") where T<:Real
        
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
        
        # append identity transformations for the net params 

        append!(specific_transformations, fill(identity, net_size*3*2))
        append!(specific_untransformations, fill(identity, net_size*3*2))
        
        # Create base model with duplicator
        base = StaticBaseModel{T}(maturities, N, M, L,  specific_transformations, specific_untransformations, model_string; results_location=results_location)



        # Pre-allocate net input
        net_input = Matrix{T}(undef, 1, N)
        for i in 1:N
            net_input[1, i] = maturities[i]
        end

        layout = param_layout(net1)  # assuming both nets have same layout

        dual_cache = Ref(Dict{DataType,NamedTuple}())
        new{T, T, T}(base, net1, net2, build_net1, build_net2, net_input, layout, dual_cache)
    end

    # Full-fields inner constructor (lets you call with fields directly)
    StaticNeuralModel{Fl}(base::StaticBaseModel{Fl,Fβ,Fγ}, net1::Chain, net2::Chain,
               build_net1, build_net2, net_input::Matrix{Fl}, layout::NamedTuple,
               dual_net_cache::Base.RefValue{Dict{DataType,NamedTuple}}) where {Fl<:Real, Fβ<:Real, Fγ<:Real} =
    new{Fl, Fβ, Fγ}(base, net1, net2, build_net1, build_net2, net_input, layout, dual_net_cache)
end

function build(model::StaticNeuralModel, Z::Matrix{Fγ}, beta::Vector{Fβ}, gamma::Vector{Fγ}, Phi::Matrix{Fβ}, delta::Vector{Fβ}, mu::Vector{Fβ}) where {Fβ<:Real, Fγ<:Real}
    base = build(model.base, Z, beta, gamma, Phi, delta, mu)
    return StaticNeuralModel{typeof(base.maturities[1])}(
       base, model.net1, model.net2, model.build_net1, model.build_net2, model.net_input, model.layout, model.dual_net_cache 
    )
end


function get_static_model_type(model::AbstractStaticNeuralModel)
    return "NNS"
end


@inline function get_nets(model::AbstractStaticNeuralModel, gamma::AbstractVector{T}) where T

    if T <: AbstractFloat
        return (net1=model.net1, net2=model.net2, layout=model.layout)
    else # Dual path: lazily build and cache once per Dual type
        cache = model.dual_net_cache[]
        if !haskey(cache, T)
            net1d   = model.build_net1(view(gamma, 1:9))
            net2d   = model.build_net2(view(gamma, 10:18))
            layoutd = param_layout(net1d)
            cache[T] = (net1=net1d, net2=net2d, layout=layoutd)
        end

        return cache[T]
    end
end


"""
    transform_net_1(net1, maturities)

    update_factor_loadings!(model::AbstractStaticNeuralModel, gamma, Z)

Update factor loadings matrix Z using neural network transformations.
"""
function update_factor_loadings!(model::AbstractStaticNeuralModel, gamma, Z)
    R = eltype(gamma)
    N = size(Z, 1)

    nets = get_nets(model, gamma)

    @views begin
        loadθ!(nets.net1, gamma[1:9], nets.layout)
        loadθ!(nets.net2, gamma[10:18], nets.layout)

        Z[:, 1] .= one(R)
        transform_net_1!(Z[:, 2], nets.net1, model.net_input)
        transform_net_2!(Z[:, 3], nets.net2, model.net_input)
    end

    
    return nothing
end

