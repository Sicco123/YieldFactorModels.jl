### MSEDNeural - Neural Network Based MSED Model

struct MSEDNeuralModel{Fl <: Real, Fβ <: Real, Fγ <: Real} <: AbstractNeuralMSEDrivenModel
    # Base
    base::MSEDrivenBaseModel{Fl, Fβ, Fγ} 
    
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
    function MSEDNeuralModel{T}(maturities::Vector{T}, N::Int, M::Int, dynamics::String, random_walk::Bool;
                               net_size = 3,
                               activation_func = tanh,
                               bias_bool = false,
                               model_string::String = "MSEDNeural",
                               results_location::String = "results/",
                               scale_grad::Bool = false,
                               forget_factor::T = T(0.9)) where T<:Real
        
        specific_transformations = Function[]
        specific_untransformations = Function[]

        # Calculate L 
        L = 3 * net_size * 2
        
        # Create duplicator based on dynamics
        if dynamics == "scalar"
            # All parameters share: [1, 1, 1, ..., 1]
            duplicator = vcat([fill(i, net_size*3) for i in 1:2]...)
        
            append!(specific_transformations, fill(from_R_to_pos, 2))
            append!(specific_untransformations, fill(from_pos_to_R, 2))
        elseif dynamics == "block_diag"
            # Two blocks: [1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6]
            duplicator = vcat([fill(i, net_size) for i in 1:6]...)
            append!(specific_transformations, fill(from_R_to_pos, 6))
            append!(specific_untransformations, fill(from_pos_to_R, 6))
        elseif dynamics == "diag"
            # No sharing: [1, 2, 3, 4, ..., L]
            duplicator = collect(1:L)
            append!(specific_transformations, fill(from_R_to_pos, L))
            append!(specific_untransformations, fill(from_pos_to_R, L))
        else
            error("dynamics must be 'scalar', 'block_diag' or 'diag'")
        end

        if !random_walk
            n = length(specific_transformations)
            append!(specific_transformations, fill(from_R_to_01, n))
            append!(specific_untransformations, fill(from_01_to_R, n))
        end

        A_guesses = T[1e-6, 1e-5, 1e-4, 1e-3]
        B_guesses = random_walk ? T[] : T[0.97, 0.98, 0.99, 0.999]

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
        
        # Create base model with duplicator, passing optional gradient scaling parameters
        base = MSEDrivenBaseModel{T}(maturities, N, M, L, duplicator, random_walk,
                                     specific_transformations, specific_untransformations,
                                     A_guesses, B_guesses, model_string;
                                     scale_grad=scale_grad, forget_factor=forget_factor, results_location=results_location)



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
    MSEDNeuralModel{Fl}(base::MSEDrivenBaseModel{Fl,Fβ,Fγ}, net1::Chain, net2::Chain,
               build_net1, build_net2, net_input::Matrix{Fl}, layout::NamedTuple,
               dual_net_cache::Base.RefValue{Dict{DataType,NamedTuple}}) where {Fl<:Real, Fβ<:Real, Fγ<:Real} =
    new{Fl, Fβ, Fγ}(base, net1, net2, build_net1, build_net2, net_input, layout, dual_net_cache)
end

function build(model::MSEDNeuralModel, Z::Matrix{Fγ}, beta::Vector{Fβ}, gamma::Vector{Fγ}, Phi::Matrix{Fβ}, delta::Vector{Fβ}, mu::Vector{Fβ}, A::Vector{Fγ}, B::Vector{Fγ}, omega::Vector{Fγ}, nu::Vector{Fγ}) where {Fβ<:Real, Fγ<:Real}
    base = build(model.base, Z, beta, gamma, Phi, delta, mu, A, B, omega, nu)
    return MSEDNeuralModel{typeof(base.maturities[1])}(
       base, model.net1, model.net2, model.build_net1, model.build_net2, model.net_input, model.layout, model.dual_net_cache
    )
end


function get_static_model_type(model::AbstractNeuralMSEDrivenModel)
    return "NNS"
end


function param_layout(m)
    ps = Flux.params(m)
    lengths = map(length, ps) |> collect
    shapes  = map(size, ps)   |> collect
    return (lengths = lengths, shapes = shapes)
end

function loadθ!(m, θ::AbstractVector, layout)
    i = 1
    for (p, n, shp) in zip(Flux.params(m), layout.lengths, layout.shapes)
        @views copyto!(p, reshape(θ[i:i+n-1], shp))
        i += n
    end
    return #m
end


@inline function get_nets(model::AbstractNeuralMSEDrivenModel, gamma::AbstractVector{T}) where T
    
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

    update_factor_loadings!(model::AbstractNeuralMSEDrivenModel, gamma, Z)

Update factor loadings matrix Z using neural network transformations.
"""
function update_factor_loadings!(model::AbstractNeuralMSEDrivenModel, gamma, Z)
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

