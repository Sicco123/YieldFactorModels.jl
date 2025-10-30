### Parameter Operations for MSED Models

# Helper function to expand unique parameters based on duplicator
function expand_params(unique_params::AbstractVector{T}, duplicator::AbstractVector{Int}) where T<:Real
    return [unique_params[i] for i in duplicator]
end

# Extract unique parameters from full array using duplicator
function get_unique_params(full_params::AbstractVector{T}, duplicator::AbstractVector{Int}) where T<:Real
    n_unique = maximum(duplicator)
    unique_params = Vector{T}(undef, n_unique)
    for i in 1:n_unique
        # Take first occurrence of each unique index
        idx = findfirst(==(i), duplicator)
        unique_params[i] = full_params[idx]
    end
    return unique_params
end



function transform_params(model::AbstractYieldFactorModel, params::Vector{T}) where T<:Real
    if hasfield(typeof(model.base), :transformations) && !isnothing(model.base.transformations)
        transformed_params = Vector{T}(undef, length(params))
        for i in 1:length(params)
            transformed_params[i] = model.base.transformations[i](params[i])
        end
        return transformed_params
    else
        return params
    end
end

function untransform_params(model::AbstractYieldFactorModel, params::Matrix{T}) where T<:Real
    if hasfield(typeof(model.base), :untransformations) && !isnothing(model.base.untransformations)

        # Check if the model has the field `transformations`
        if !hasfield(typeof(model.base), :transformations)
            error("Model does not have transformations defined")
        elseif length(model.base.transformations) != length(params)
            error("Length of transformations $(length(model.base.transformations)) does not match the number of parameters $(length(params))")
        end

        # check if length transformation is equal to the number of parameters and display the length in an error message
        if length(model.base.untransformations) != length(params)
            error("Length of transformations $(length(model.base.untransformations)) does not match the number of parameters $(length(params))")
        end

        untransformed_params = Matrix{T}(undef, size(params, 1), size(params, 2))
        
        for j in 1:size(params, 2)
            for i in 1:size(params, 1)
                untransformed_params[i, j] = model.base.untransformations[i](params[i, j])
            end
        end 
        return untransformed_params
    else
        return params
    end
end