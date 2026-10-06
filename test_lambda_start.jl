# The starting-λ gate: an estimation run whose parameters imply λ₁ > LAMBDA_START_MAX
# is rejected before it can win a multi-start comparison.
# Run: julia --project=<env> YieldFactorModels.jl/test_lambda_start.jl
using YieldFactorModels
using YieldFactorModels: compute_loss, lambda_start, set_params!, untransform_params,
                         LAMBDA_START_MAX
using LinearAlgebra, Random

const M = YieldFactorModels

mats = Float64[3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120]
N, T = length(mats), 60
Random.seed!(1)
# level ~4%, mild slope, small noise — only has to keep the filter finite
data = [4.0 + 0.01 * τ + 0.05 * randn() for τ in mats, _ in 1:T]

unconstrain(model, v) = untransform_params(model, reshape(v, :, 1))[:, 1]
# the gate returns 1e10 * λ₁; a real negative log-likelihood here is O(1e2)-O(1e5)
rejected(loss) = loss >= 1e9

# ---- score-driven λ models: γ₁ = ω, so λ₁ = 1e-2 + exp(ω) ---------------------
# params: [a, b, ω, δ(3), vec(Φ)(9)]
sdns_params(ω) = [1.5, 0.95, ω, 4.3, -0.3, 2.0, vec(0.95 * I(3))...]

for (name, mdl) in ("SD-NS"  => M.MSEDλModel{Float64}(mats, N, 3, false),
                    "SSD-NS" => M.MSEDλModel{Float64}(mats, N, 3, false; scale_grad = true))
    for ω in (-4.6, -2.9, -0.72)          # λ₁ = 0.020, 0.065, 0.497 — all admissible
        p = unconstrain(mdl, sdns_params(ω))
        loss = compute_loss(mdl, data, p)
        @assert isapprox(lambda_start(mdl), 1e-2 + exp(ω); rtol = 1e-6) "$name: λ₁ wrong at ω=$ω"
        @assert lambda_start(mdl) < LAMBDA_START_MAX
        @assert isfinite(loss) && !rejected(loss) "$name: admissible ω=$ω was penalized ($loss)"
    end

    # smooth1/T250 run 140: ω = 0.204 with b = 1 → λ₁ = 1.24, RRMSE 271. Must be rejected.
    penalized = compute_loss(mdl, data, unconstrain(mdl, [1.5, 1.0, 0.204, 4.3, -0.3, 2.0, vec(0.95 * I(3))...]))
    @assert rejected(penalized) "$name: λ₁ = 1.24 not rejected ($penalized)"

    # penalty must increase in λ₁ so finite differences point back toward smaller ω
    worse = compute_loss(mdl, data, unconstrain(mdl, sdns_params(1.5)))
    @assert worse > penalized "$name: penalty is flat in λ₁, gradient would be zero"

    # either side of the cap: λ₁ = 1e-2 + exp(ω), so ω = log(0.48) → 0.49, log(0.50) → 0.51
    @assert !rejected(compute_loss(mdl, data, unconstrain(mdl, sdns_params(log(0.48))))) "$name: λ₁ = 0.49 rejected"
    @assert rejected(compute_loss(mdl, data, unconstrain(mdl, sdns_params(log(0.50))))) "$name: λ₁ = 0.51 accepted"
end

# ---- TVλ: state starts at (I - Φ)\δ = μ, λ is its 4th element ----------------
# params: [σ_obs, Γ upper (10), μ(4), A row-major (16)] — slots 12:15 are the long-run
# mean μ (set_params! builds δ = (I-Φ)μ) and 16:31 the Ansley-Kohn matrix A, from which
# Φ = L A' B⁻ᵀ L⁻¹. λ₁ = tvλ_link(μ₄) either way, since (I-Φ)⁻¹δ = μ for any Φ.
tvλ = M.TVλDNSModel{Float64}(mats, N, 3)
function tvλ_params(logλ_mean; a = 0.95)
    A = a * I(4)
    μ = [4.3, -0.3, 2.0, logλ_mean]
    Γ = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.1]  # upper triangle, column by column
    return [0.05, Γ..., μ..., vec(Matrix(A)')...]
end

# TVλ uses the same λ = 1e-2 + exp(γ) link as the score-driven models, so the gate
# is what bounds λ from above and γ∞ maps to λ₁ identically in both.
for γ∞ in (-4.6, -2.9, -0.75)          # λ₁ = 0.020, 0.065, 0.482 — all admissible
    loss = compute_loss(tvλ, data, unconstrain(tvλ, tvλ_params(γ∞)))
    @assert isapprox(lambda_start(tvλ), 1e-2 + exp(γ∞); rtol = 1e-4) "TVλ: λ₁ wrong at $γ∞ (got $(lambda_start(tvλ)))"
    @assert lambda_start(tvλ) < LAMBDA_START_MAX
    @assert !rejected(loss) "TVλ: admissible γ∞=$γ∞ was penalized ($loss)"
end

# γ∞ = 0.204 → λ₁ = 1.24, the level-only collapse. Nothing caps it now, so the gate
# must reject it, with a penalty rising in λ₁ so the finite differences point back.
penalized = compute_loss(tvλ, data, unconstrain(tvλ, tvλ_params(0.204)))
@assert rejected(penalized) "TVλ: λ₁ = 1.24 not rejected ($penalized)"
@assert compute_loss(tvλ, data, unconstrain(tvλ, tvλ_params(1.5))) > penalized "TVλ: penalty is flat in λ₁"

# Either side of the gate, and the floor holds in the limit. The EKF Jacobian is the
# derivative of this link: Z[:,4] = ∂y_pred/∂β₄ must match a finite difference of
# predict_observation!.
@assert !rejected(compute_loss(tvλ, data, unconstrain(tvλ, tvλ_params(log(0.48))))) "TVλ: λ₁ = 0.49 rejected"
@assert rejected(compute_loss(tvλ, data, unconstrain(tvλ, tvλ_params(log(0.50))))) "TVλ: λ₁ = 0.51 accepted"
@assert M.tvλ_link(-1e3) ≈ M.TVΛ_MIN "TVλ: link escapes its floor"

let m = M.TVλDNSModel{Float64}(mats, N, 3), h = 1e-6
    m.base.beta .= [4.3, -0.3, 2.0, -2.0]
    M.predict_observation!(m)
    M.update_measurement_jacobian!(m)
    analytic = copy(m.base.Z[:, 4])

    fd = similar(analytic)
    for (s, sgn) in ((1, +1), (2, -1))
        m.base.beta[4] = -2.0 + sgn * h
        M.predict_observation!(m)
        s == 1 ? (fd .= m.base.y_pred) : (fd .= (fd .- m.base.y_pred) ./ 2h)
    end
    @assert isapprox(analytic, fd; rtol = 1e-5) "TVλ: measurement Jacobian ≠ d(y_pred)/dβ₄\n$(analytic)\n$(fd)"
end

# Ansley-Kohn makes Φ stationary for any A, so a large A must simply be accepted
@assert !rejected(compute_loss(tvλ, data, unconstrain(tvλ, tvλ_params(-2.9; a = 12.0)))) "TVλ: large A (near-unit-root Φ) rejected"

println("✓ λ₁ gate OK: λ₁ = $(M.TVΛ_MIN) + exp(γ₁) for both families (γ₁ = ω score-driven, ((I-Φ)\\δ)[4] for TVλ), rejected above $LAMBDA_START_MAX, EKF Jacobian matches finite differences")
