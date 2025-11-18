### YieldFactorModels Base
abstract type AbstractKalmanModel <: AbstractYieldFactorModel end
abstract type AbstractDNSModel <: AbstractKalmanModel end
abstract type AbstractTVλDNSModel <: AbstractKalmanModel end

struct KalmanBaseModel{Fl<:Real,Fβ<:Real} <: AbstractYieldFactorModel
    maturities::Vector{Fl}
    N::Int
    M::Int
    L::Int

    Z::Matrix{Fβ}
    beta::Vector{Fβ}

    Phi::Matrix{Fβ}
    delta::Vector{Fβ}

    gamma::Vector{Fβ}

    Omega_state::Matrix{Fβ}
    Omega_obs::Matrix{Fβ}

    P::Matrix{Fβ}
    In::Matrix{Fβ}
    y_pred::Vector{Fβ}
    v::Vector{Fβ}
    F::Matrix{Fβ}
    F_inv::Matrix{Fβ}
    temp_NxM::Matrix{Fβ}
    temp_MxN::Matrix{Fβ}


    transformations::Vector{Function}
    untransformations::Vector{Function}

    flat_params::Vector{Fβ}

    init_folder::String
    results_folder::String
    model_string::String


    # inner constructor (OK to use `new` here). Accepts optional
    # `scale_grad` and `forget_factor` keyword arguments for gradient
    # scaling behaviour. Defaults disable scaling.
    function KalmanBaseModel{Fl}(
        maturities::Vector{Fl}, N::Int, M::Int, L::Int,
        specific_transformations::Vector{Function},
        specific_untransformations::Vector{Function},
         model_string; results_location::String = "results/"
    ) where {Fl<:Real}

        Z     = ones(Fl, N, M)
        beta  = zeros(Fl, M)
        Phi   = zeros(Fl, M, M)
        delta = zeros(Fl, M)

        gamma = zeros(Fl, L)
    
        Omega_state = Matrix{Fl}(I, M, M)
        Omega_obs   = Matrix{Fl}(I, N, N)
        P = Matrix{Fl}(I, M, M)
        In = Matrix{Fl}(I, M, M)
        y_pred = zeros(Fl, N)
        v = zeros(Fl, N)
        F = zeros(Fl, N, N)
        F_inv = zeros(Fl, N, N)
        temp_NxM = zeros(Fl, N, M)
        temp_MxN = zeros(Fl, M, N)

         # Define common transformations


        state_covariance_transformations = Function[]
        state_covariance_untransformations = Function[]
        for i in 1:M 
            for j in 1:M
                if i == j
                    push!(state_covariance_transformations, from_R_to_pos)
                    push!(state_covariance_untransformations, from_pos_to_R)
                elseif j < i

                    push!(state_covariance_transformations, identity)
                    push!(state_covariance_untransformations, identity)
                else
                    continue
                end
            end
        end


        phi_transforms = Function[]
        phi_untransforms = Function[]
        for i in 1:M
            for j in 1:M
                if i == j
                    push!(phi_transforms, from_R_to_11)
                    push!(phi_untransforms, from_11_to_R)
                else
                    push!(phi_transforms, identity)
                    push!(phi_untransforms, identity)
                end
            end
        end
       
        transformations = [
            specific_transformations...,
            from_R_to_pos, 
            state_covariance_transformations...,
            fill(identity, M)...,
            phi_transforms...,
        ]
       
        untransformations = [
            specific_untransformations...,
            from_pos_to_R,
            state_covariance_untransformations...,
            fill(identity, M)...,
            phi_untransforms...
        ]

        init_folder    = "YieldFactorModels.jl/initializations/$(model_string)/"
        results_folder = results_location #"YieldFactorModels.jl/results/$(model_string)/"
        flat_params = zeros(Fl, length(transformations))
        
        return new{Fl,Fl}(maturities, N, M, L, Z, beta, Phi, delta, gamma,
                              Omega_state, Omega_obs, P, In, y_pred, v, F, F_inv, temp_NxM, temp_MxN,
                              transformations, untransformations, flat_params, 
                              init_folder, results_folder, model_string )
    end

    # Full-fields inner constructor (lets you call with fields directly)
    KalmanBaseModel{Fl,Fβ}(
        maturities::Vector{Fl}, N::Int, M::Int, L::Int,
        Z::Matrix{Fβ}, beta::Vector{Fβ},
        Phi::Matrix{Fβ}, delta::Vector{Fβ}, gamma::Vector{Fβ},Omega_state::Matrix{Fβ},
        Omega_obs::Matrix{Fβ}, P::Matrix{Fβ}, In::Matrix{Fβ},
        y_pred::Vector{Fβ}, v::Vector{Fβ}, F::Matrix{Fβ}, F_inv::Matrix{Fβ},
        transformations::Vector{Function}, untransformations::Vector{Function},
        init_folder::String, results_folder::String, model_string::String 
    ) where {Fl<:Real,Fβ<:Real} =
    new{Fl,Fβ}(maturities, N, M, L, Z, beta, Phi, delta, gamma,
                      Omega_state, Omega_obs, P, In, y_pred, v, F, F_inv,
                      transformations, untransformations,
                      init_folder, results_folder, model_string)
end



function get_param_groups(model::AbstractKalmanModel, param_groups::Vector{String})
    base = model.base
    length_total_params = length(get_params(model))
    if length(param_groups) == length_total_params
        return param_groups
    else
        println("Default param groups assigned.")
        return vcat(fill("1", length_total_params))
    end
end


