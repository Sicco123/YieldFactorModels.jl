# Canonical model-type names keyed by every accepted alias (numeric or string).
# Single source of truth for alias → canonical-name resolution. Used both to
# build result/init folder paths and to dispatch model construction, so a model
# invoked by its numeric alias (e.g. "-1") still resolves to its canonical name
# (e.g. "RW") everywhere.
const MODEL_TYPE_ALIASES = Dict{String,String}(
    "1C" => "1C",                 "0"  => "1C",
    "TVλ" => "TVλ",               "1"  => "TVλ",
    "NS" => "NS",                 "2"  => "NS",
    "NNS" => "NNS",               "3"  => "NNS",
    "SD-NS" => "SD-NS",           "4"  => "SD-NS",
    "RWSD-NS" => "RWSD-NS",       "5"  => "RWSD-NS",
    "SSD-NS" => "SSD-NS",         "6"  => "SSD-NS",
    "SRWSD-NS" => "SRWSD-NS",     "7"  => "SRWSD-NS",
    "1SD-NNS" => "1SD-NNS",       "8"  => "1SD-NNS",
    "1RWSD-NNS" => "1RWSD-NNS",   "9"  => "1RWSD-NNS",
    "2SD-NNS" => "2SD-NNS",       "10" => "2SD-NNS",
    "2RWSD-NNS" => "2RWSD-NNS",   "11" => "2RWSD-NNS",
    "3SD-NNS" => "3SD-NNS",       "12" => "3SD-NNS",
    "3RWSD-NNS" => "3RWSD-NNS",   "13" => "3RWSD-NNS",
    "1SSD-NNS" => "1SSD-NNS",     "14" => "1SSD-NNS",
    "1SRWSD-NNS" => "1SRWSD-NNS", "15" => "1SRWSD-NNS",
    "2SSD-NNS" => "2SSD-NNS",     "16" => "2SSD-NNS",
    "2SRWSD-NNS" => "2SRWSD-NNS", "17" => "2SRWSD-NNS",
    "3SSD-NNS" => "3SSD-NNS",     "18" => "3SSD-NNS",
    "3SRWSD-NNS" => "3SRWSD-NNS", "19" => "3SRWSD-NNS",
    "NNS-Not-Anchored" => "NNS-Not-Anchored",                 "20" => "NNS-Not-Anchored",
    "1SD-NNS-Not-Anchored" => "1SD-NNS-Not-Anchored",         "21" => "1SD-NNS-Not-Anchored",
    "1RWSD-NNS-Not-Anchored" => "1RWSD-NNS-Not-Anchored",     "22" => "1RWSD-NNS-Not-Anchored",
    "2SD-NNS-Not-Anchored" => "2SD-NNS-Not-Anchored",         "23" => "2SD-NNS-Not-Anchored",
    "2RWSD-NNS-Not-Anchored" => "2RWSD-NNS-Not-Anchored",     "24" => "2RWSD-NNS-Not-Anchored",
    "3SD-NNS-Not-Anchored" => "3SD-NNS-Not-Anchored",         "25" => "3SD-NNS-Not-Anchored",
    "3RWSD-NNS-Not-Anchored" => "3RWSD-NNS-Not-Anchored",     "26" => "3RWSD-NNS-Not-Anchored",
    "1SSD-NNS-Not-Anchored" => "1SSD-NNS-Not-Anchored",       "27" => "1SSD-NNS-Not-Anchored",
    "1SRWSD-NNS-Not-Anchored" => "1SRWSD-NNS-Not-Anchored",   "28" => "1SRWSD-NNS-Not-Anchored",
    "2SSD-NNS-Not-Anchored" => "2SSD-NNS-Not-Anchored",       "29" => "2SSD-NNS-Not-Anchored",
    "2SRWSD-NNS-Not-Anchored" => "2SRWSD-NNS-Not-Anchored",   "30" => "2SRWSD-NNS-Not-Anchored",
    "3SSD-NNS-Not-Anchored" => "3SSD-NNS-Not-Anchored",       "31" => "3SSD-NNS-Not-Anchored",
    "3SRWSD-NNS-Not-Anchored" => "3SRWSD-NNS-Not-Anchored",   "32" => "3SRWSD-NNS-Not-Anchored",
    "pC" => "pC",                 "1100" => "pC",
    "vanillaNN" => "vanillaNN",   "a"  => "vanillaNN",
    "RW" => "RW",                 "-1" => "RW",
)

"""
    canonical_model_type(model_type) -> String

Resolve a model-type token (canonical name or numeric alias) to its canonical
name. Throws on an unrecognised token. Resolve a model type with this before
building result/init folder paths so that numeric aliases and canonical names
map to the same location.
"""
function canonical_model_type(model_type::String)
    haskey(MODEL_TYPE_ALIASES, model_type) || error("Invalid model type: $model_type")
    return MODEL_TYPE_ALIASES[model_type]
end

"""
    create_model(model_type, maturities, N, M, float_type, results_location)

Initialize the appropriate model based on model_type specification.
Returns the initialized model and the standardized model_type string.
"""
function create_model(model_type::String, maturities::Vector, N::Int, M::Int, float_type::Type, results_location::String)
    model_type = canonical_model_type(model_type)
    # covert maturities
    maturities = convert(Vector{float_type}, maturities)
    # Model type mapping and initialization
    if model_type == "1C"
        model = DNSModel{float_type}(maturities, N, M; model_string=model_type, results_location=results_location)
    elseif model_type == "TVλ"
        model = TVλDNSModel{float_type}(maturities, N, M; model_string=model_type, results_location=results_location)
    elseif model_type == "NS"
        model = StaticλModel{float_type}(maturities, N, M; model_string=model_type, results_location=results_location)
    elseif model_type == "NNS"
        model = StaticNeuralModel{float_type}(maturities, N, M; model_string=model_type, results_location=results_location)
    elseif model_type == "SD-NS"
        model = MSEDλModel{float_type}(maturities, N, M, false; model_string=model_type, results_location=results_location)
    elseif model_type == "RWSD-NS"
        model = MSEDλModel{float_type}(maturities, N, M, true; model_string=model_type, results_location=results_location)
    elseif model_type == "SSD-NS"
        model = MSEDλModel{float_type}(maturities, N, M, false; model_string=model_type, scale_grad=true, results_location=results_location)
    elseif model_type == "SRWSD-NS"
        model = MSEDλModel{float_type}(maturities, N, M, true; model_string=model_type, scale_grad=true, results_location=results_location)
    elseif model_type == "1SD-NNS"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", false; model_string=model_type, results_location=results_location)
    elseif model_type == "1RWSD-NNS"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", true; model_string=model_type, results_location=results_location)
    elseif model_type == "2SD-NNS"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", false; model_string=model_type, results_location=results_location)
    elseif model_type == "2RWSD-NNS"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", true; model_string=model_type, results_location=results_location)
    elseif model_type == "3SD-NNS"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", false; model_string=model_type, results_location=results_location)
    elseif model_type == "3RWSD-NNS"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", true; model_string=model_type, results_location=results_location)
    elseif model_type == "1SSD-NNS"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", false; model_string=model_type, scale_grad=true, results_location=results_location)
    elseif model_type == "1SRWSD-NNS"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", true; model_string=model_type, scale_grad=true, results_location=results_location)
    elseif model_type == "2SSD-NNS"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", false; model_string=model_type, scale_grad=true, results_location=results_location)
    elseif model_type == "2SRWSD-NNS"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", true; model_string=model_type, scale_grad=true, results_location=results_location)
    elseif model_type == "3SSD-NNS"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", false; model_string=model_type, scale_grad=true, results_location=results_location)
    elseif model_type == "3SRWSD-NNS"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", true; model_string=model_type, scale_grad=true, results_location=results_location)

    elseif model_type == "NNS-Not-Anchored"
        model = StaticNeuralModel{float_type}(maturities, N, M; model_string=model_type, results_location=results_location, transform_bool=false)
    elseif model_type == "1SD-NNS-Not-Anchored"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", false; model_string=model_type, results_location=results_location, transform_bool=false)
    elseif model_type == "1RWSD-NNS-Not-Anchored"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", true; model_string=model_type, results_location=results_location, transform_bool=false)
    elseif model_type == "2SD-NNS-Not-Anchored"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", false; model_string=model_type, results_location=results_location, transform_bool=false)
    elseif model_type == "2RWSD-NNS-Not-Anchored"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", true; model_string=model_type, results_location=results_location, transform_bool=false)
    elseif model_type == "3SD-NNS-Not-Anchored"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", false; model_string=model_type, results_location=results_location, transform_bool=false)
    elseif model_type == "3RWSD-NNS-Not-Anchored"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", true; model_string=model_type, results_location=results_location, transform_bool=false)
    elseif model_type == "1SSD-NNS-Not-Anchored"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", false; model_string=model_type, scale_grad=true, results_location=results_location, transform_bool=false)
    elseif model_type == "1SRWSD-NNS-Not-Anchored"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "scalar", true; model_string=model_type, scale_grad=true, results_location=results_location, transform_bool=false)
    elseif model_type == "2SSD-NNS-Not-Anchored"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", false; model_string=model_type, scale_grad=true, results_location=results_location, transform_bool=false)
    elseif model_type == "2SRWSD-NNS-Not-Anchored"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "block_diag", true; model_string=model_type, scale_grad=true, results_location=results_location, transform_bool=false)
    elseif model_type == "3SSD-NNS-Not-Anchored"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", false; model_string=model_type, scale_grad=true, results_location=results_location, transform_bool=false)
    elseif model_type == "3SRWSD-NNS-Not-Anchored"
        model = MSEDNeuralModel{float_type}(maturities, N, M, "diag", true; model_string=model_type, scale_grad=true, results_location=results_location, transform_bool=false)

    elseif model_type == "pC"
        model = nothing
    elseif model_type == "vanillaNN"
        model = nothing
    elseif model_type == "RW"
        model = RandomWalkModel{float_type}(maturities, N, M; model_string=model_type, results_location=results_location)
    else
        error("Invalid model type: $model_type")
    end

    return model, model_type
end
