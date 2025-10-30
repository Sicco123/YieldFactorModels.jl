
function from_R_to_pos(x)
    return exp(x)
end

function from_pos_to_R(x)
    return log(x)
end

function from_11_to_R(x)
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
    # Transform [0, 1] to R
    T = typeof(x)  # Read the type of x
    return log(x / (T(1) - x))
end
