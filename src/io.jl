

# Function to filter states and save results
function save_results(model::AbstractYieldFactorModel, results, loss, thread_Id::String, data_type::String)
       
        # create result subfolders if they do not exist
        if !isdir(model.base.results_folder)
                mkpath(model.base.results_folder)
        end

        model_string = model.base.model_string

        params = get_params(model)
   
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
        writedlm(string(model.base.results_folder, model_string, "__thread_id__", thread_Id, "__out_params.csv"),                        params, ',')
        end

