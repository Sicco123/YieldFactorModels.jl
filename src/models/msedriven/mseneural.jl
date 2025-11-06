### MSEDNeural - Neural Network Based MSED Model

struct MSEDNeuralModel{Fl <: Real, Fβ <: Real, Fγ <: Real, F1, F2} <: AbstractNeuralMSEDrivenModel
    # Base
    base::MSEDrivenBaseModel{Fl, Fβ, Fγ} 
    
    # Neural Networks    
    net1::F1
    net2::F2

    net_input::Matrix{Fl}

    transform_bool::Bool
    

    # Constructor
    function MSEDNeuralModel{T}(maturities::Vector{T}, N::Int, M::Int, dynamics::String, random_walk::Bool;
                               net_size = 3,
                               activation_func = tanh,
                               bias_bool = false,
                               model_string::String = "MSEDNeural",
                               results_location::String = "results/",
                               scale_grad::Bool = false,
                               forget_factor::T = T(0.9), transform_bool::Bool = true) where T<:Real
        
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
        net1 = Chain(Dense(1 => net_size, activation_func), Dense(net_size, 1; use_bias = bias_bool))
        net2 = Chain(Dense(1 => net_size, activation_func), Dense(net_size, 1; use_bias = bias_bool))


        rng = Random.default_rng()
        _, states = Lux.setup(rng, net1)
        state_dummy = states
        
        
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

        fnet1 = (p) -> net1(net_input, shapeγ(p), state_dummy)[1]
        fnet2 = (p) -> net2(net_input, shapeγ(p), state_dummy)[1]

        new{T, T, T, typeof(fnet1), typeof(fnet2)}(base, fnet1, fnet2, net_input, transform_bool)
    end

    # Full-fields inner constructor (lets you call with fields directly)
    MSEDNeuralModel{Fl}(base::MSEDrivenBaseModel{Fl,Fβ,Fγ}, net1::F1, net2::F2,
               net_input::Matrix{Fl}, transform_bool::Bool) where {Fl<:Real, Fβ<:Real, Fγ<:Real, F1, F2} =
    new{Fl, Fβ, Fγ, F1, F2}(base, net1, net2, net_input, transform_bool)
end

function build(model::MSEDNeuralModel, Z::Matrix{Fγ}, beta::Vector{Fβ}, gamma::Vector{Fγ}, Phi::Matrix{Fβ}, delta::Vector{Fβ}, mu::Vector{Fβ}, A::Vector{Fγ}, B::Vector{Fγ}, omega::Vector{Fγ}, nu::Vector{Fγ}) where {Fβ<:Real, Fγ<:Real}
    base = build(model.base, Z, beta, gamma, Phi, delta, mu, A, B, omega, nu)
    return MSEDNeuralModel{typeof(base.maturities[1])}(
       base, model.net1, model.net2,  model.net_input, model.transform_bool
    )
end


function get_static_model_type(model::AbstractNeuralMSEDrivenModel)
    if model.transform_bool
        return "NNS"
    else 
        return "NNS-Anchored"
    end
end

@inline function shapeγ(γ::StridedVector{T}) where T

    params = (
        layer_1 = (
            weight = reshape(view(γ, 1:3), 3, 1),
            bias = view(γ, 4:6)
        ),
        layer_2 = (
            weight = reshape(view(γ, 7:9), 1, 3),
        ),
    )

    return params
end



@inline function update_factor_loadings!(
    model::AbstractNeuralMSEDrivenModel, 
    gamma::AbstractVector{T}, 
    Z::AbstractMatrix{R}
) where {T <: Real, R <: Real}
    

    # Pre-compute views to avoid repeated indexing
    z2 = @view Z[:, 2]
    z3 = @view Z[:, 3]
    
    # Set first column to ones if needed (more robust check)
    if Z[1, 1] != one(T)
        Z[:, 1] .= one(T)
    end
    
    z2 .= model.net1(@view gamma[1:9])'
    z3 .= model.net2(@view gamma[10:18])'
    #println(typeof(gamma))
    # Transform
    transform_net_1!(z2, model.net_input, Val(model.transform_bool))
    transform_net_2!(z3, model.net_input, Val(model.transform_bool))


    
    return nothing
end