# Add the package to the environment
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Import the package and required dependencies
using Revise
using YieldFactorModels
using LinearAlgebra
using ForwardDiff
using Random

#Random.seed!(123)  # For reproducibility

cd("..")

println(pwd())
# NOTE: `export VAR=...` is a shell command and is not valid Julia syntax in a code cell.
# For runtime settings that can be changed from within Julia use `ENV` or library APIs.
# Set BLAS / native libraries thread knobs where possible:
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["OMP_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"  # If MKL.jl is used, prefer MKL.set_num_threads(1)
# Also set BLAS threads from Julia (affects LinearAlgebra.BLAS):
LinearAlgebra.BLAS.set_num_threads(1)

#model_names = [ "1SSD-NNS", "1RWSD-NNS", "1SRWSD-NNS", "2SD-NNS", "2SSD-NNS", "2RWSD-NNS", "2SRWSD-NNS", "3SD-NNS", "3SSD-NNS", "3RWSD-NNS", "3SRWSD-NNS"] # "RWSD-NS", "SRWSD-NS",

model_names = ["RW"]

# shuffle model names 
Random.shuffle!(model_names)

for model_name in model_names
    try
        YieldFactorModels.run("6", 231, 12, true, model_name, Float64; window_type = "expanding",  max_group_iters=10, run_optimization=false, reestimate=false )
    catch e
        println("Error occurred while running model $model_name: $e")
    end
end

