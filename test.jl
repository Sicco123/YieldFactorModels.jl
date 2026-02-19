# Add the package to the environment
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Import the package and required dependencies
using YieldFactorModels
using Random

#Random.seed!(123)  # For reproducibility

cd("..")

println(pwd())
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["OMP_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"  
LinearAlgebra.BLAS.set_num_threads(1)

function main()

    model_names = [ "1SSD-NNS"] 
    for model_name in model_names
    
        YieldFactorModels.run("6", 231, 12, false, model_name, Float64; window_type = "expanding",  max_group_iters=10, run_optimization=true, reestimate=true, group_tol = 1e-6)
       
    end

end
main()
