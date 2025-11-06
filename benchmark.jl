# Add the package to the environment
using Pkg
Pkg.activate("TVNets-1.11.6")
Pkg.instantiate()

# Import the package and required dependencies
using YieldFactorModels
using LinearAlgebra
# using ForwardDiff
using Random
using BenchmarkTools

#Random.seed!(123)  # For reproducibility

#cd("..")

println(pwd())
# NOTE: `export VAR=...` is a shell command and is not valid Julia syntax in a code cell.
# For runtime settings that can be changed from within Julia use `ENV` or library APIs.
# Set BLAS / native libraries thread knobs where possible:
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["OMP_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"  # If MKL.jl is used, prefer MKL.set_num_threads(1)
# Also set BLAS threads from Julia (affects LinearAlgebra.BLAS):
LinearAlgebra.BLAS.set_num_threads(1)

function benchmark()
    data, maturities = load_data("YieldFactorModels.jl/data/", "6")
    Z_test = randn(24, 3)
    y_test = randn(24)
    beta_test = zeros(3)
    YieldFactorModels.get_β_OLS!(beta_test, Z_test, y_test)
    float_type = Float64
    model, model_type = YieldFactorModels.create_model("3SSD-NNS-Anchored", maturities,24, 3, float_type, "YieldFactorModels.jl/results/thread_id__6/")
    param_groups = YieldFactorModels.get_param_groups(model, String[])
    all_params = YieldFactorModels.load_initial_parameters!(model, model_type, float_type)
    YieldFactorModels.set_params!(model, all_params[:, 1])
    # Load static parameters if applicable
    all_params[:,1] = YieldFactorModels.load_static_parameters!(model, model_type, "YieldFactorModels.jl/results/", "6", all_params[:,1])
    # Convert parameters to appropriate float type
    all_params = convert(Matrix{float_type}, all_params)
    results = predict(model, data[:, 1:end])

    println("Benchmarking with BenchmarkTools...")
    println("(This may take a minute...)\n")

    benchmark_result = @benchmark YieldFactorModels.get_loss(
        $model, 
        $data, 
    ) samples=50 evals=3

    display(benchmark_result)

    println("\n" * "="^60)
    println("Summary:")
    println("  Minimum time: $(minimum(benchmark_result.times) / 1e9) seconds")
    println("  Median time:  $(median(benchmark_result.times) / 1e9) seconds")
    println("  Mean time:    $(mean(benchmark_result.times) / 1e9) seconds")
    println("  Allocations:  $(benchmark_result.allocs)")
    println("  Memory:       $(benchmark_result.memory / 1e6) MB")
    println("="^60)

end
benchmark()