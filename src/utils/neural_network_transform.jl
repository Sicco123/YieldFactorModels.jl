
"""
    transform_net_1(net1, maturities)

Original allocating version for compatibility.
"""
function transform_net_1(net1, maturities)
    n = length(maturities)
    # # 1) grab the raw first output
    last_raw = net1([maturities[end-1]])[1]
    Fl = typeof(last_raw)
    first_raw = net1([maturities[1]])[1]-last_raw + + Fl(1e-7)

    # # 2) allocate result and accumulate sum of squares of the tail
    values = Vector{Fl}(undef, n)
    sum = Fl(0.0)

    @inbounds @simd for i in 2:n-1
        # compute ratio to the first
        values[i] = ((net1([maturities[i]])[1] - last_raw ) / (first_raw ))^2
        sum += values[i]
    end

    values[1] = Fl(1.0)
    values[n-1] = Fl(0.0)
    values[n] = Fl(0.0)
    
    return values
end


"""
    transform_net_2(net2, maturities; scale = 0.9610)

Original allocating version for compatibility.
"""
function transform_net_2(net2, maturities; scale = 0.9610)
    n = length(maturities)
    first_maturity = net2([maturities[1]])[1]
    last_maturity = net2([maturities[end]])[1]
    Fl = typeof(first_maturity)
    values = Vector{Fl}(undef, n)
    
    # Line through first and last maturity
    slope = (last_maturity - first_maturity ) / (maturities[end] - maturities[1])

    intercept = first_maturity - slope * maturities[1]

    @inbounds for i in 1:n
        line_val = slope * maturities[i] + intercept
        values[i] = (net2([maturities[i]])[1] - line_val)^2  # Ensure positivity
    end

    # transform l2 norm to 0.9610 
    scale_factor = sqrt(sum(values.^2)) / (Fl(scale)) + Fl(1e-7)

    @inbounds for i in 1:n
        values[i] /= scale_factor
    end

    return values
end

# @inline function transform_net_1!(
#     dest, net, inputs
# ) 
#     n = size(inputs, 2)
#     dest .= vec(net(inputs))                    # one forward over all maturities
#     T = eltype(dest[1])

#     dest[1] = dest[1] - dest[end-1] + T(1e-7)
#     dest_inv = T(1) / dest[1]
#     @inbounds for i in 2:n-2
#         dest[i] = ((dest[i] - dest[end-1]) * dest_inv)^2
#     end
#     dest[1]   = one(T)
#     dest[n-1] = zero(T)
#     dest[n]   = zero(T)
#     return dest
# end


# Generic fallback (works for views, OffsetArrays)
@inline function transform_net_1!(dest::AbstractVector, net, inputs)
    n = size(inputs, 2)
    # Write the raw forward directly into dest (no extra temp copy)
    dest .= vec(net(inputs))  # vec is a reshape when the output is a vector -> no alloc by itself

    T = eltype(dest)
    @inbounds begin
        #raw_first = dest[begin]          # cache raw values before we overwrite
        raw_last  = dest[end-1]
        inv_first = inv(dest[begin]  - raw_last + T(1e-7))

        # do not touch dest[1] until the very end
        for i in firstindex(dest)+1 : lastindex(dest)-2
            t = (dest[i] - raw_last) * inv_first
            dest[i] = t * t              # cheaper than ^2
        end
        dest[firstindex(dest)]         = one(T)
        dest[lastindex(dest)-1]        = zero(T)
        dest[lastindex(dest)]          = zero(T)
    end
    return dest
end

# Fast path for contiguous vectors – allows SIMD
@inline function transform_net_1!(dest::StridedVector{T}, net, inputs) where {T}
    n = size(inputs, 2)
    dest .= vec(net(inputs))

    @inbounds begin
        raw_first = dest[1]
        raw_last  = dest[end-1]
        inv_first = inv(raw_first - raw_last + T(1e-7))

        @simd for i in 2:n-2
            t = (dest[i] - raw_last) * inv_first
            dest[i] = t * t
        end
        dest[1]   = one(T)
        dest[n-1] = zero(T)
        dest[n]   = zero(T)
    end
    return dest
end

# @inline function transform_net_2!(
#     dest, net, inputs; scale = 0.9610
# ) 
#     n = size(inputs, 2)
#     dest .= vec(net(inputs))                     # one forward over all maturities
#     T = eltype(dest[1])

#     slope      = (dest[end] - dest[1]) / (inputs[1,end] - inputs[1,1])
#     intercept  = dest[1] - slope*inputs[1,1]

#     sum_sq = zero(T)
#     @inbounds for i in 2:n-1
#         dest[i]  = (dest[i] - slope*inputs[1,i] + intercept)^2      # ensure positivity
#         sum_sq  += dest[i]^2
#     end
#     dest[1]   = zero(T)
#     dest[n]   = zero(T)

#     scale_factor = sqrt(sum_sq) / T(scale) + T(1e-7)
#     @inbounds for i in 1:n
#         dest[i] /= scale_factor
#     end
#     return dest
# end

# Generic
@inline function transform_net_2!(dest::AbstractVector, net, inputs; scale = 0.9610)
    n = size(inputs, 2)
    dest .= vec(net(inputs))

    T = eltype(dest)
    x1 = @inbounds inputs[1, 1]
    xN = @inbounds inputs[1, n]

    @inbounds begin
        raw1 = dest[begin]
        rawN = dest[end]

        slope     = (rawN - raw1) / (xN - x1)
        intercept = raw1 - slope * x1

        sum_sq = zero(T)
        for i in firstindex(dest)+1 : lastindex(dest)-1
            # residual (before square)
            r = dest[i] - (slope * inputs[1, i] - intercept)
            r2 = r * r
            dest[i] = r2                 # ensure positivity
            sum_sq  = muladd(r2, r2, sum_sq)  # sum of squares of r^2  (as in your code)
        end
        dest[firstindex(dest)]  = zero(T)
        dest[lastindex(dest)]   = zero(T)

        # normalize with a multiply instead of n divides
        denom = sqrt(sum_sq) / T(scale) + T(1e-7)
        inv   = inv(denom)
        @simd for i in firstindex(dest):lastindex(dest)
            dest[i] *= inv
        end
    end
    return dest
end

# Fast path for StridedVector enables SIMD in the main loop too
@inline function transform_net_2!(dest::StridedVector{T}, net, inputs; scale = 0.9610) where {T}
    n = size(inputs, 2)
    dest .= vec(net(inputs))

    @inbounds begin
        x1  = inputs[1, 1]
        xN  = inputs[1, n]
        raw1 = dest[1]
        rawN = dest[end]

        slope     = (rawN - raw1) / (xN - x1)
        intercept = raw1 - slope * x1

        sum_sq = zero(T)
        @simd for i in 2:n-1
            r  = dest[i] - (slope * inputs[1, i] - intercept)
            r2 = r * r
            dest[i] = r2
            sum_sq  = muladd(r2, r2, sum_sq)
        end
        dest[1] = zero(T); dest[n] = zero(T)

        denom = sqrt(sum_sq) / T(scale) + T(1e-7)
        inv   = 1/(denom)
        @simd for i in 1:n
            dest[i] *= inv
        end
    end
    return dest
end