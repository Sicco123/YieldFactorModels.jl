# Add the package to the environment
using Pkg
Pkg.activate("/home/skooiker/TVNets/TVNets-1.11.6")
Pkg.instantiate()

# Import the package and required dependencies
using YieldFactorModels
using Random
using LinearAlgebra

Random.seed!(123)  # For reproducibility

println(pwd())
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["OMP_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"  
LinearAlgebra.BLAS.set_num_threads(1)

function main()
    # DIAGONAL=1 restricts every model's Phi to its diagonal: separate init files
    # (init_params_<M>-diag__thread_id__7.csv), separate results folder <M>-diag/, so the
    # unrestricted runs already in results/thread_id__7/ are never touched.
    diagonal = get(ENV, "DIAGONAL", "0") == "1"

    # STATIC_SEED=0 stops the score-driven models inheriting NS/NNS per-window fits, so
    # they run on their own init_params_<M>__thread_id__7.csv instead.
    static_seed = get(ENV, "STATIC_SEED", "1") == "1"

    # The score-driven models seed from their static counterpart's forecast database,
    # so the static batch has to be finished before the second one is launched.
    #model_names = ["RW", "NS", "NNS", "1C"]
    # MODELS (comma-separated) overrides the list, so the static batch and the
    # score-driven batch can be queued together with a Slurm dependency instead of this
    # file being hand-edited between the two submissions.
    model_names = haskey(ENV, "MODELS") ? String.(split(ENV["MODELS"], ",")) :
                                          ["SD-NS", "SSD-NS", "1SD-NNS", "1SSD-NNS"]
    println("Models this process will run: ", model_names, "  static_seed=", static_seed)

    for model_name in model_names
    
        YieldFactorModels.run("7", 231, 12, true , model_name, Float64; window_type = "moving",  max_group_iters=10, run_optimization=false, reestimate=true, group_tol = 1e-6, diagonal=diagonal, static_seed=static_seed)
       
    end

end
main()
