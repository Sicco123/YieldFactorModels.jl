# Checks the stationarity gate that compute_loss applies to every Kalman model.
#
#   julia --project=<env> test_stationarity_gate.jl
#
# Complements test_equivalence.jl: that one pins the filter's numbers, this one
# pins what the optimizer is allowed to be handed.

using Pkg
Pkg.activate("/home/skooiker/TVNets/TVNets-1.11.6")

using YieldFactorModels
using LinearAlgebra
using DelimitedFiles
using Printf
using Random: MersenneTwister

const ROOT   = @__DIR__
const THREAD = "7"
const PENALTY = 1e10

function setup(mt)
    d, m = load_data(joinpath(ROOT, "data") * "/", THREAD)
    data = Matrix{Float64}(d); mats = Vector{Float64}(m)
    if mt == "AFNS"
        data ./= 100; mats ./= 12
    end
    model, canon = YieldFactorModels.create_model(mt, mats, length(mats), 3, Float64,
                        joinpath(ROOT, "results_gate_scratch", mt) * "/")
    raw = readdlm(joinpath(ROOT, "initializations", canon, "init_params_$(canon).csv"), ',')
    raw = YieldFactorModels.reconcile_init_params(model, raw)
    return model, data, Matrix{Float64}(raw)[:, 1]
end

# compute_loss takes UNCONSTRAINED params; the init files are constrained.
const compute_loss = YieldFactorModels.compute_loss

unconstrain(model, p) = vec(YieldFactorModels.untransform_params(model, reshape(p, :, 1)))

failures = String[]
note(c, msg) = c || push!(failures, msg)

for mt in ["1C", "AFNS", "TVλ"]
    model, data, init = setup(mt)
    insample = data[:, 1:231]

    # TVλ is skipped here: its Ansley-Kohn parametrization makes an explosive Phi
    # unreachable, so the "scale the transition block until it explodes" sweep below
    # can never produce one. That invariant is checked directly at the end instead.
    if mt != "TVλ"
        u = unconstrain(model, init)

        # 1) a stationary point must pass the gate and reproduce get_loss exactly
        gated = compute_loss(model, insample, u)
        set_params!(model, YieldFactorModels.transform_params(model, u))
        rho = YieldFactorModels.spectral_radius(model.base.Phi)
        direct = -get_loss(model, insample)
        note(rho < 1, "$mt: fixture is not stationary (rho=$rho), test is meaningless")
        note(gated == direct,
             "$mt: stationary point changed value ($gated vs $direct)")
        note(abs(gated) < PENALTY, "$mt: stationary point hit the penalty ($gated)")
        @printf("%-5s stationary  rho=%.6f  loss=%.6f  (matches get_loss: %s)\n",
                mt, rho, gated, gated == direct)

        # 2) every explosive Phi must be rejected, and the penalty must RISE with
        #    rho so the finite-difference gradient points back into the stable
        #    region. Scaling the transition block is not monotone in rho (for AFNS
        #    it scales K^P, and Phi = exp(-K^P dt)), so sweep and keep whichever
        #    points actually came out explosive.
        explosive = Tuple{Float64,Float64}[]   # (rho, loss)
        for scale in (1.5, 2.0, 3.0, 5.0, -2.0)
            bad = copy(u)
            bad[end-8:end] .*= scale        # transition block: Phi (DNS) / K^P (AFNS)
            v = compute_loss(model, insample, bad)
            set_params!(model, YieldFactorModels.transform_params(model, bad))

            # AFNS can produce a non-finite Phi (update_transition! NaNs it out when
            # the matrix exponential fails); that must be rejected too, and cannot
            # be handed to eigvals.
            if !all(isfinite, model.base.Phi)
                @printf("%-5s scale %+5.1f  Phi non-finite  loss=%11.4e  %s\n", mt, scale, v,
                        v >= PENALTY ? "REJECTED" : "!! ACCEPTED")
                note(v >= PENALTY, "$mt: non-finite Phi was not rejected (got $v)")
                continue
            end

            r = YieldFactorModels.spectral_radius(model.base.Phi)
            stable = r < 1
            stable || push!(explosive, (r, v))
            @printf("%-5s scale %+5.1f  rho=%.6f  loss=%11.4e  %s\n", mt, scale, r, v,
                    stable ? "stable, accepted" : (v >= PENALTY ? "explosive, REJECTED"
                                                               : "explosive, !! ACCEPTED"))
            note(stable || v >= PENALTY, "$mt: explosive Phi (rho=$r) was not rejected")
            note(!stable || v < PENALTY, "$mt: stationary Phi (rho=$r) was rejected")
        end
        # Only 1C can still reach an explosive Phi, so only there does the sweep owe
        # us points to check the slope on. AFNS holds K^P's diagonal positive
        # (Phi = exp(-K^P dt)) and TVλ uses Ansley-Kohn; for those the penalty is
        # unreachable by construction, which is a stronger guarantee than a slope.
        note(mt != "1C" || length(explosive) >= 2,
             "$mt: sweep produced $(length(explosive)) explosive points, need >= 2 " *
             "to check the penalty slope")
        if length(explosive) >= 2
            sort!(explosive, by = first)
            # Non-decreasing everywhere, and strictly increasing wherever rho
            # actually differs. Equal rho (e.g. scaling the block by +2 and -2
            # leaves the eigenvalue moduli unchanged) must give an equal penalty.
            ok = all(1:length(explosive)-1) do i
                ρ1, v1 = explosive[i]; ρ2, v2 = explosive[i+1]
                ρ1 < ρ2 ? v1 < v2 : v1 <= v2
            end
            note(ok, "$mt: penalty is not increasing in rho ($(explosive)); a flat " *
                     "penalty gives a zero finite-difference gradient")
        end
    end

    # 3) non-finite params must never reach the filter
    nf = fill(NaN, length(YieldFactorModels.get_params(model)))
    v = compute_loss(model, insample, nf)
    note(isfinite(v) && v >= PENALTY, "$mt: NaN params returned $v, expected the penalty")
    @printf("%-5s NaN params  loss=%.4e  %s\n", mt, v, v >= PENALTY ? "REJECTED" : "!! LEAKED")
end

# TVλ: Phi is stationary and Sigma positive definite BY CONSTRUCTION, for any
# parameter vector at all — that is the whole point of the Ansley-Kohn transform
# (Phi = L A' B⁻ᵀ L⁻¹, Sigma = Gamma - Phi Gamma Phi'). No penalty involved, so
# throw junk at it: the invariant either holds everywhere or the transform is wrong.
let (model, data, _) = setup("TVλ"), n = length(YieldFactorModels.get_params(model))
    worst_rho, min_eig, rejected = 0.0, Inf, 0
    for seed in 1:200
        p = 3 .* randn(MersenneTwister(seed), n)      # unconstrained, deliberately wild
        set_params!(model, YieldFactorModels.transform_params(model, p))
        # A numerically singular Gamma is NaN'd out on purpose; the gate rejects it.
        if !all(isfinite, model.base.Phi)
            rejected += 1
            note(compute_loss(model, data[:, 1:231], p) >= PENALTY,
                 "TVλ: draw $seed NaN'd Phi but was not rejected by the gate")
            continue
        end
        r = YieldFactorModels.spectral_radius(model.base.Phi)
        e = minimum(eigvals(Symmetric(model.base.Omega_state)))
        worst_rho = max(worst_rho, r); min_eig = min(min_eig, e)
        note(r < 1, "TVλ: draw $seed gave rho=$r, Ansley-Kohn must keep it below 1")
        note(e > 0, "TVλ: draw $seed gave a non-PD Sigma (min eigenvalue $e)")
    end
    @printf("TVλ   200 random draws: max rho=%.6f (<1), min Sigma eigenvalue=%.3e (>0), %d rejected\n",
            worst_rho, min_eig, rejected)
end

println()
if isempty(failures)
    println("STATIONARITY GATE: PASS")
    exit(0)
else
    println("STATIONARITY GATE: FAIL")
    foreach(f -> println("  ", f), failures)
    exit(1)
end
