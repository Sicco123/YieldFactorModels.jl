# Bounds on the score-driven parameters: A ∈ (0,1), B ∈ (0.5,1).
# Run: julia --project=<env> YieldFactorModels.jl/test_param_bounds.jl
using YieldFactorModels
using YieldFactorModels: from_R_to_A, from_A_to_R, from_R_to_051, from_051_to_R, A_MAX

const M = YieldFactorModels

# The caps hold for anything the optimizer can propose. At |x| beyond ~30 the logistic
# underflows to the bound itself in floating point (the old exp parameterization did the
# same), so the guarantee there is the closed interval.
for x in (-1e6, -50.0, -1.0, 0.0, 1.0, 50.0, 1e6)
    a = from_R_to_A(x)
    b = from_R_to_051(x)
    @assert 0 <= a <= A_MAX "A out of bounds at x=$x: $a"
    @assert 0.5 <= b <= 1 "B out of bounds at x=$x: $b"
end

# strictly interior over the range the optimizer realistically visits
for x in (-30.0, -10.0, -1.0, 0.0, 1.0, 10.0, 30.0)
    @assert 0 < from_R_to_A(x) < A_MAX "A not interior at x=$x"
    @assert 0.5 < from_R_to_051(x) < 1 "B not interior at x=$x"
end

# round trip on the values the optimizer actually starts from, plus the healthy spread of
# the unscaled models (SD-NS: median 1.77, q95 5.14) which must stay interior
for a in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1.77, 5.14, 9.0)
    @assert isapprox(from_R_to_A(from_A_to_R(a)), a; rtol=1e-6) "A round trip failed at $a"
    @assert 0 < from_R_to_A(from_A_to_R(a)) < A_MAX "A not interior at $a"
end
for b in (0.9, 0.95, 0.97, 0.98, 0.99, 0.999)
    @assert isapprox(from_R_to_051(from_051_to_R(b)), b; rtol=1e-6) "B round trip failed at $b"
end

# boundary values must untransform finitely rather than to ±Inf
for v in (0.0, float(A_MAX))
    @assert isfinite(from_A_to_R(v)) "from_A_to_R($v) not finite"
end
for v in (0.5, 1.0)
    @assert isfinite(from_051_to_R(v)) "from_051_to_R($v) not finite"
end

# the pathological estimates that motivated this (breaks/T500 run 429: a=285.75, b=0.0;
# smooth1/T500 run 52: a=111.19, b=0.0084) are no longer reachable
@assert from_R_to_A(from_A_to_R(285.75)) <= A_MAX
@assert from_R_to_A(from_A_to_R(111.19)) <= A_MAX
@assert from_R_to_051(from_051_to_R(0.0084)) > 0.5

# models still build and their transformation vectors line up with the parameter count
mats = Float64[3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120]
for (name, mdl) in ("SSD-NS"  => M.MSEDλModel{Float64}(mats, 17, 3, false; scale_grad=true),
                    "SD-NS"   => M.MSEDλModel{Float64}(mats, 17, 3, false),
                    "RWSD-NS" => M.MSEDλModel{Float64}(mats, 17, 3, true),
                    "1SSD-NNS"=> M.MSEDNeuralModel{Float64}(mats, 17, 3, "scalar", false; scale_grad=true))
    p = M.get_params(mdl)
    @assert length(mdl.base.transformations) == length(p) "$name: $(length(mdl.base.transformations)) transforms vs $(length(p)) params"
    @assert length(mdl.base.untransformations) == length(p) "$name: untransform length mismatch"
    # every A_guess/B_guess must survive the untransform used at the start of estimation
    for g in mdl.base.A_guesses
        @assert isfinite(mdl.base.untransformations[1](g)) "$name: A_guess $g untransforms to non-finite"
    end
    for g in mdl.base.B_guesses
        i = maximum(mdl.base.duplicator) + 1
        @assert isfinite(mdl.base.untransformations[i](g)) "$name: B_guess $g untransforms to non-finite"
    end
end

println("✓ param bounds OK: A ∈ (0,$A_MAX), B ∈ (0.5,1), round trips and model construction pass")
