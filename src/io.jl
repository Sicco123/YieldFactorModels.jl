

# Function to filter states and save results
function save_results(model::AbstractYieldFactorModel, results, loss, thread_Id::String, data_type::String)
       
        # create result subfolders if they do not exist
        if !isdir(model.base.results_folder)
                mkpath(model.base.results_folder)
        end

        model_string = model.base.model_string

        params = natural_params(model)

        # Save filtered and smoothed states
        writedlm(string(model.base.results_folder, model_string, "__thread_id__", thread_Id, "__factors_filtered_", data_type, ".csv"), 
                vcat(results.factors, results.states)', ',')
        writedlm(string(model.base.results_folder, model_string, "__thread_id__", thread_Id, "__fit_filtered_", data_type, ".csv"), 
                results.preds', ',')
        writedlm(string(model.base.results_folder, model_string, "__thread_id__", thread_Id, "__factor_loadings_1_filtered_", data_type, ".csv"), 
                results.factor_loadings_1', ',')
        writedlm(string(model.base.results_folder, model_string, "__thread_id__", thread_Id, "__factor_loadings_2_filtered_", data_type, ".csv"), 
                results.factor_loadings_2', ',')

        writedlm(string(model.base.results_folder, model_string, "__thread_id__", thread_Id, "__loss.csv"), 
                        [loss], ',')

        println("Print io params: ")
        println(string(model.base.results_folder, model_string, "__thread_id__", thread_Id, "__out_params.csv"))
        writedlm(string(model.base.results_folder, model_string, "__thread_id__", thread_Id, "__out_params.csv"),                        params, ',')

        save_transformed_params(model, string(model.base.results_folder, model_string, "__thread_id__", thread_Id, "__out_params_transformed.csv"))
        end


# out_params.csv holds the model's own quantities (natural_params), which for TVλ means
# Γ and Φ rather than the stored Ansley-Kohn primitives chol(Γ) and A — the same layout
# the init files are written in, so a finished run can be copied into initializations/
# and reconcile_init_params converts it back on load. The name,value dump below is the
# readable companion: it also carries Σ, δ and the spectral radius, which no init needs.
save_transformed_params(::AbstractYieldFactorModel, ::String) = nothing

function save_transformed_params(model::AbstractKalmanModel, path::String)
    b = model.base
    names = String[]
    vals  = Float64[]
    push_pair!(n, v) = (push!(names, n); push!(vals, Float64(v)))

    # TVλ never fills gamma — λ is a state there — so a zero gamma means "unused"
    all(iszero, b.gamma) || push_pair!("gamma", b.gamma[1])
    for i in 1:b.N
        push_pair!("Omega_obs[$i,$i]", b.Omega_obs[i, i])
    end
    for i in 1:b.M
        push_pair!("delta[$i]", b.delta[i])
    end
    # long-run mean μ solving δ = (I - Φ)μ; NaN if I - Φ is singular
    mu = try
        (I - b.Phi) \ b.delta
    catch
        fill(NaN, b.M)
    end
    for i in 1:b.M
        push_pair!("mu[$i]", mu[i])
    end
    for j in 1:b.M, i in 1:b.M
        push_pair!("Phi[$i,$j]", b.Phi[i, j])
    end
    for j in 1:b.M, i in 1:b.M
        push_pair!("Sigma[$i,$j]", b.Omega_state[i, j])
    end
    push_pair!("spectral_radius_Phi", maximum(abs, eigvals(b.Phi)))

    println("Print transformed params: ")
    println(path)
    writedlm(path, [names vals], ',')
end

