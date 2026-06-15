
function from_R_to_pos(x)
    return exp(x)
end

function from_pos_to_R(x)
    return log(x)
end

function from_11_to_R(x)
    # Clamp just inside (-1, 1) so log1p never sees an argument <= -1. The
    # optimiser's transformed params are always strictly inside, so this is a
    # no-op there; it only guards against (e.g.) a concentrated least-squares Phi
    # that lands a hair past +/-1, which would otherwise throw a DomainError.
    b = oftype(x, 1) - oftype(x, 1e-10)
    x = clamp(x, -b, b)
    return log1p(x) - log1p(-x)
end
# function from_R_to_11(x)
#     T = typeof(x)  # Read the type of x
#     println("hi")
#     println(T)
#     println(typeof(T(2) * exp(x) / (T(1) + exp(x)) - T(1)))
#     return T(2) * exp(x) / (T(1) + exp(x)) - T(1)
# end

function from_R_to_11(x)
    o = one(x)              # Dual “1” matching x’s type/tag
    t = o + o               # Dual “2”
    y = exp(x)
    return t * y / (o + y) - o
end

function from_R_to_01(x)
    # Transform R to [0, 1]
    T = typeof(x)  # Read the type of x
    return T(1) / (T(1) + exp(-x))
end

function from_01_to_R(x)
    # Transform [0, 1] to R (clamped just inside (0,1) to avoid log(<=0)).
    T = typeof(x)
    x = clamp(x, T(1e-10), T(1) - T(1e-10))
    return log(x / (T(1) - x))
end
