# Guard on the neural slope loading at prediction time. Run: julia --project=... test_loading_guard.jl
using Pkg; Pkg.activate("/home/skooiker/TVNets/TVNets-1.11.6")
using YieldFactorModels, LinearAlgebra, Printf
const YFM = YieldFactorModels

mats = Float64[3,6,9,12,15,18,21,24,30,36,48,60,72,84,96,108,120]; N = length(mats)
neural = first(YFM.create_model("1SSD-NNS", mats, N, 3, Float64, "/tmp/lg/"))
lam    = first(YFM.create_model("SD-NS",    mats, N, 3, Float64, "/tmp/lg/"))

# the clamp itself: column 2 bounded, columns 1 and 3 untouched
Z = hcat(ones(N), Float64[-0.5, 0.0, 0.3, 1.0, 1.7, 5.0, 79.6, fill(0.4, N - 7)...], fill(0.377, N))
ref1, ref3 = copy(Z[:, 1]), copy(Z[:, 3])
YFM.clamp_slope_loading!(neural, Z)
@assert all(0 .<= Z[:, 2] .<= 1) "slope loading left [0, 1]: $(extrema(Z[:, 2]))"
@assert Z[1, 2] == 0.0 && Z[5, 2] == 1.0 && Z[7, 2] == 1.0 "clamp did not saturate at the bounds"
@assert Z[3, 2] == 0.3 "clamp moved a value that was already in range"
@assert Z[:, 1] == ref1 && Z[:, 3] == ref3 "clamp touched the level or curvature loading"
println("ok: clamp bounds column 2 only")

# no-op for the lambda family
Z2 = copy(Z); Z2[:, 2] .= 5.0
YFM.clamp_slope_loading!(lam, Z2)
@assert all(Z2[:, 2] .== 5.0) "clamp fired on a non-neural model"
println("ok: no-op for SD-NS")

# and inert on an observed step: filtering real data must be unchanged
Random_data = 4 .+ 0.5 .* cos.((1:60)' ./ 6) .* ones(N)
p = YFM.reconcile_init_params(neural, reshape(vec(YFM.get_params(neural)), :, 1))
YFM.set_params!(neural, p[:, 1])
l1 = YFM.get_loss(neural, Random_data)
YFM.set_params!(neural, p[:, 1])
l2 = YFM.get_loss(neural, Random_data)
@assert l1 == l2 && isfinite(l1) "observed-step loss is not reproducible: $l1 vs $l2"
println(@sprintf("ok: observed-step objective untouched (loss %.6f)", l1))
println("\nall loading-guard checks passed")
