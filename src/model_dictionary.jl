"""
    create_model(model_type, maturities, N, M, float_type)

Initialize the appropriate model based on model_type specification.
Returns the initialized model and the standardized model_type string.
"""
function create_model(model_type::String, maturities::Vector, N::Int, M::Int, float_type::Type, results_location::String)
    # covert maturities 
    maturities = convert(Vector{float_type}, maturities)
    # Model type mapping and initialization
    if model_type == "1C" || model_type == "0"
        model = DNSModel{float_type}(maturities, N, M; model_string=model_type, results_location=results_location)
        model_type = "1C"  
    elseif model_type == "TVλ" || model_type == "1"
        model = nothing
    elseif model_type == "NS" || model_type == "2"
        model = StaticλModel{float_type}(maturities, N, M; model_string=model_type, results_location=results_location)
        model_type = "NS"
    elseif model_type == "NNS" || model_type == "3"
        model = StaticNeuralModel{float_type}(maturities, N, M; model_string=model_type, results_location=results_location)
        model_type = "NNS"

    elseif model_type == "SD-NS" || model_type == "4"
        model = MSEDλModel{float_type}(maturities, N, M, false; model_string=model_type, results_location=results_location)
        model_type = "SD-NS"
    elseif model_type == "RWSD-NS" || model_type == "5"
        model = MSEDλModel{float_type}(maturities, N, M, true; model_string=model_type, results_location=results_location)
        model_type = "RWSD-NS"
    elseif model_type == "SSD-NS" || model_type == "6"
        model = MSEDλModel{float_type}(maturities, N, M, false; model_string=model_type, scale_grad=true, results_location=results_location)
        model_type = "SSD-NS"
    elseif model_type == "SRWSD-NS" || model_type == "7"
        model = MSEDλModel{float_type}(maturities, N, M, true; model_string=model_type, scale_grad=true, results_location=results_location)
        model_type = "SRWSD-NS"
    elseif model_type == "1SD-NNS" || model_type == "8"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", false; model_string=model_type, results_location=results_location)

        model_type = "1SD-NNS"
    elseif model_type == "1RWSD-NNS" || model_type == "9"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", true; model_string=model_type, results_location=results_location)
        model_type = "1RWSD-NNS"
    elseif model_type == "2SD-NNS" || model_type == "10"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", false; model_string=model_type, results_location=results_location)
        model_type = "2SD-NNS"
    elseif model_type == "2RWSD-NNS" || model_type == "11"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", true; model_string=model_type, results_location=results_location)
        model_type = "2RWSD-NNS"
    elseif model_type == "3SD-NNS" || model_type == "12"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", false; model_string=model_type, results_location=results_location)
        model_type = "3SD-NNS"
    elseif model_type == "3RWSD-NNS" || model_type == "13"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", true; model_string=model_type, results_location=results_location)
        model_type = "3RWSD-NNS"
    elseif model_type == "1SSD-NNS" || model_type == "14"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", false; model_string=model_type, scale_grad=true, results_location=results_location)
        model_type = "1SSD-NNS"
    elseif model_type == "1SRWSD-NNS" || model_type == "15"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", true; model_string=model_type, scale_grad=true, results_location=results_location)
        model_type = "1SRWSD-NNS"
    elseif model_type == "2SSD-NNS" || model_type == "16"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", false; model_string=model_type, scale_grad=true, results_location=results_location)
        model_type = "2SSD-NNS"
    elseif model_type == "2SRWSD-NNS" || model_type == "17"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", true; model_string=model_type, scale_grad=true, results_location=results_location)
        model_type = "2SRWSD-NNS"
    elseif model_type == "3SSD-NNS" || model_type == "18"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", false; model_string=model_type, scale_grad=true, results_location=results_location)
        model_type = "3SSD-NNS"
    elseif model_type == "3SRWSD-NNS" || model_type == "19"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", true; model_string=model_type, scale_grad=true, results_location=results_location)
        model_type = "3SRWSD-NNS"
    
    elseif model_type == "NNS-Anchored" || model_type == "20"
        model = StaticNeuralModel{float_type}(maturities, N, M; model_string=model_type, results_location=results_location, transform_bool=false)
        model_type = "NNS-Anchored"
    elseif model_type == "1SD-NNS-Anchored" || model_type == "21"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", false; model_string=model_type, results_location=results_location, transform_bool=false)
        model_type = "1SD-NNS-Anchored"
    elseif model_type == "1RWSD-NNS-Anchored" || model_type == "22"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", true; model_string=model_type, results_location=results_location, transform_bool=false)
        model_type = "1RWSD-NNS-Anchored"
    elseif model_type == "2SD-NNS-Anchored" || model_type == "23"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", false; model_string=model_type, results_location=results_location, transform_bool=false)
        model_type = "2SD-NNS-Anchored"
    elseif model_type == "2RWSD-NNS-Anchored" || model_type == "24"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", true; model_string=model_type, results_location=results_location, transform_bool=false)
        model_type = "2RWSD-NNS-Anchored"
    elseif model_type == "3SD-NNS-Anchored" || model_type == "25"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", false; model_string=model_type, results_location=results_location, transform_bool=false)
        model_type = "3SD-NNS-Anchored"
    elseif model_type == "3RWSD-NNS-Anchored" || model_type == "26"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", true; model_string=model_type, results_location=results_location, transform_bool=false)
        model_type = "3RWSD-NNS-Anchored"
    elseif model_type == "1SSD-NNS-Anchored" || model_type == "27"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", false; model_string=model_type, scale_grad=true, results_location=results_location, transform_bool=false)
        model_type = "1SSD-NNS-Anchored"
    elseif model_type == "1SRWSD-NNS-Anchored" || model_type == "28"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", true; model_string=model_type, scale_grad=true, results_location=results_location, transform_bool=false)
        model_type = "1SRWSD-NNS-Anchored"
    elseif model_type == "2SSD-NNS-Anchored" || model_type == "29"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", false; model_string=model_type, scale_grad=true, results_location=results_location, transform_bool=false)
        model_type = "2SSD-NNS-Anchored"
    elseif model_type == "2SRWSD-NNS-Anchored" || model_type == "30"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", true; model_string=model_type, scale_grad=true, results_location=results_location, transform_bool=false)
        model_type = "2SRWSD-NNS-Anchored"
    elseif model_type == "3SSD-NNS-Anchored" || model_type == "31"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", false; model_string=model_type, scale_grad=true, results_location=results_location, transform_bool=false)
        model_type = "3SSD-NNS-Anchored"
    elseif model_type == "3SRWSD-NNS-Anchored" || model_type == "32"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", true; model_string=model_type, scale_grad=true, results_location=results_location, transform_bool=false)
        model_type = "3SRWSD-NNS-Anchored"

    elseif model_type == "pC" || model_type == "1100" 
        model = nothing
        model_type = "pC"
    elseif model_type == "vanillaNN" || model_type == "a"
        model = nothing
        model_type = "vanillaNN"        
    elseif model_type == "RW" || model_type == "-1"
        model = RandomWalkModel{float_type}(maturities, N, M; model_string=model_type, results_location=results_location)
        model_type = "RW"
    else
        error("Invalid model type: $model_type")
    end
    
    return model, model_type
end