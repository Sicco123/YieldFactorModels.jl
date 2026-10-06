
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
    # clamp: a stored parameter sitting exactly on a bound would otherwise untransform to ±Inf
    y = min(max(x, T(1e-6)), T(1) - T(1e-6))
    return log(y / (T(1) - y))
end

# Transform R to (0, A_MAX) — used for the score loading A.
# Unbounded exp let the optimizer pick A ~ 285 (breaks/T250 run 429) and A ~ 111
# (smooth1/T500 run 52), which drives λ far outside its plausible range and blows the
# forecasts up. A_MAX = 10 sits above the healthy spread of the unscaled models
# (SD-NS: median 1.77, q95 5.14) so the cap stays slack on ordinary runs.
const A_MAX = 10

from_R_to_A(x) = A_MAX * from_R_to_01(x)

function from_A_to_R(x)
    T = typeof(x)
    # clamp: keep a parameter on a bound from untransforming to ±Inf. The lower margin has
    # to sit below the smallest A_guess (1e-6), so it cannot reuse from_01_to_R's 1e-6.
    y = min(max(x, T(1e-12)), T(A_MAX) - T(1e-6))
    return log(y / (T(A_MAX) - y))
end

# Largest starting λ an estimation run may imply, where the starting λ is the one the
# filter is initialized at: 1e-2 + exp(ω) for the score-driven λ models (γ₁ = ω), and
# 1e-2 + exp(((I-Φ)\δ)[4]) for TVλ — same link, same floor. Above ~0.5 the slope and
# curvature loadings are ≈ 1/(λτ) at every maturity past the short end, so the model
# degenerates to level-only and the forecasts blow up (smooth1/T250 run 140: ω = 0.204,
# b = 1, λ₁ = 1.24, RRMSE 271). Estimated λ sits at ~0.06 across all DGPs, so 0.5 is
# slack by ~8x. Neither link caps λ, so this gate binds for every λ model.
# Assumes maturities in months, as everywhere in this repo — in years, raise it.
const LAMBDA_START_MAX = 0.5

# Transform R to (0.5, 1) — used for the persistence B of the score-driven recursion.
# B near 0 degenerates the filter: the driver is pinned at its intercept and the
# in-sample optimum can sit at an implausible level (see breaks/T500 run 429, b = 0).
# `one(x)` arithmetic keeps this AD-safe, as in from_R_to_11.
function from_R_to_051(x)
    o = one(x)
    h = o / (o + o)
    return h + h / (o + exp(-x))
end

function from_051_to_R(x)
    T = typeof(x)
    # clamp: a stored parameter sitting exactly on a bound would otherwise untransform to ±Inf
    y = min(max(x, T(0.5) + T(1e-6)), T(1) - T(1e-6))
    return log((y - T(0.5)) / (T(1) - y))
end
