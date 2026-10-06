# Check that __out_params_transformed.csv holds the matrices the filter runs on:
# Φ stationary, Σ symmetric, and μ recovered from δ = (I - Φ)μ equal to the raw μ
# slots of flat_params.
using Pkg; Pkg.activate(@__DIR__)
using YieldFactorModels, DelimitedFiles, LinearAlgebra, Test
const Y = YieldFactorModels

N, M = 24, 3
model = Y.TVλDNSModel{Float64}(collect(1.0:N), N, M; results_location = mktempdir() * "/")
# Driven from the init file rather than a saved out_params.csv: both are now written in
# the same readable layout (Γ and Φ, not chol(Γ) and A), and the init file is the one
# that is always present. reconcile_init_params turns it into the stored primitives.
init = readdlm(joinpath(@__DIR__, "initializations/TVλ/init_params_TVλ.csv"), ',')
raw = vec(Y.reconcile_init_params(model, init))
Y.set_params!(model, raw)

out = joinpath(mktempdir(), "transformed.csv")
Y.save_transformed_params(model, out)
tbl = readdlm(out, ',')
val(name) = tbl[findfirst(==(name), tbl[:, 1]), 2]

nstates = M + 1
mu_raw = raw[end-nstates^2-nstates+1:end-nstates^2]   # μ slots, before the A block
@test [val("mu[$i]") for i in 1:nstates] ≈ mu_raw
@test val("spectral_radius_Phi") < 1
@test [val("Sigma[$i,$j]") for i in 1:nstates, j in 1:nstates] ≈ model.base.Omega_state
@test [val("Phi[$i,$j]") for i in 1:nstates, j in 1:nstates] ≈ model.base.Phi

# reconcile_init_params turns Γ and Φ into the Ansley-Kohn slots, and set_params! must
# hand them back.
Γ_init = [init[1 + (j*(j-1))÷2 + i] for i in 1:nstates, j in 1:nstates]
Γ_init = Symmetric(triu(Γ_init))
@test model.Gamma ≈ Γ_init
@test model.base.Phi ≈ permutedims(reshape(init[end-nstates^2+1:end], nstates, nstates))

# natural_params is the other direction: what save_results now writes to out_params.csv
# must be the same readable layout the init file uses, i.e. reconcile back to `raw`.
nat = Y.natural_params(model)
@test nat ≈ vec(init)
@test vec(Y.reconcile_init_params(model, reshape(nat, :, 1))) ≈ raw
println("ok")
