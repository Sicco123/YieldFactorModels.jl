### Parameter Operations for MSED Models

# Reconcile a stored init-parameter matrix with the model's current parameter
# count. Default: leave unchanged (models with a fixed layout). AFNS overrides
# this to resize its per-maturity measurement-variance block to N.
reconcile_init_params(::AbstractYieldFactorModel, all_params::AbstractMatrix) = all_params

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



"""
    natural_params(model)

The parameter vector as it should be written to `out_params.csv`: the quantities the
model is actually stated in, which for most models is the stored vector itself. Only
models whose stored slots hold a reparametrization (TVλ) need a method here.
"""
natural_params(model::AbstractYieldFactorModel) = get_params(model)

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

# ---------------------------------------------------------------------------
# diagonal factor dynamics
# ---------------------------------------------------------------------------
"""
    is_diagonal(model)

True when the model was built with `diagonal = true`: the factor transition Φ is
restricted to its diagonal, so the states evolve as independent AR(1)s.

The flag rides on `model_string`'s `-diag` suffix rather than on a field of its own.
That suffix is also what names the model's init and result files, so the restriction
and the files it reads can never disagree about which model this is. Works on a model
or on a bare base model.
"""
is_diagonal(m) = endswith(model_name(m), "-diag")

model_name(m) = (hasproperty(m, :base) ? m.base : m).model_string

"""Strip the `-diag` suffix: the folder an init file lives in is shared with the
unrestricted model, only the file name carries the suffix."""
base_model_name(s::AbstractString) = replace(s, "-diag" => "")

"""
Slots of the parameter vector that `diagonal = true` pins to zero: Φ's off-diagonal
entries. Every family stores Φ as the last `M^2` slots (row-major for the Kalman
models, column-major for the others — the off-diagonal index set is the same either
way). AFNS is the exception and needs none of this: its `K^P` is diagonal already.
"""
phi_offdiag_slots(M::Int, n::Int) = [n - M * M + (j - 1) * M + i for j in 1:M for i in 1:M if i != j]

"""
chol(Γ)'s off-diagonal slots in the Kalman layout, in the order `set_params!` reads
them: slot 1 is the observation variance, then (1,1), (1,2), (2,2), (1,3), ...
Only TVλ needs these — see its `set_params!`.
"""
function chol_offdiag_slots(M::Int)
    slots = Int[]
    k = 2
    for j in 1:M, i in 1:j
        i < j && push!(slots, k)
        k += 1
    end
    return slots
end

pin_slots!(p::AbstractVector, slots) = (for i in slots; p[i] = zero(eltype(p)); end; p)

"""
Mark the pinned slots as group `"-1"`, the placeholder `estimate_steps!` skips, so no
optimizer spends a block on a parameter `set_params!` zeroes anyway. No-op unless the
model is diagonal. Not used by AFNS, whose layout has no Φ block.
"""
function pin_diagonal_groups!(model, groups::Vector{String})
    is_diagonal(model) || return groups
    M = model.base.M
    groups[phi_offdiag_slots(M, length(groups))] .= "-1"
    model isa AbstractTVλDNSModel && (groups[chol_offdiag_slots(M)] .= "-1")
    return groups
end

"""
    static_counterpart(model)

The static model a time-varying model is seeded from, under the same restriction as
the model itself: a diagonal score-driven model reads the diagonal static fit's
parameters, not the unrestricted one's.
"""
static_counterpart(model) = get_static_model_type(model) * (is_diagonal(model) ? "-diag" : "")
