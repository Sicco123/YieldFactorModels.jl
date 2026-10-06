# Full-sample in-sample fit for thread 8.
#
# test-thread_8.jl runs the moving-window forecast exercise with run_optimization=false,
# so results/thread_id__8/<M>/out_params.csv there is just the initialization, and the
# only fitted parameters that run produced are the per-window ones.
#
# This script estimates each model once on ALL 480 observations (in_sample_end = T), so
# out_params.csv is a genuine full-sample fit. It writes to thread id "8full" (data files
# are symlinks to thread 8's) so results/thread_id__8/ is never touched.
using Pkg
Pkg.activate("/home/skooiker/TVNets/TVNets-1.11.6")
Pkg.instantiate()

using YieldFactorModels
using Random
using LinearAlgebra

Random.seed!(123)

ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["OMP_NUM_THREADS"] = "1"
ENV["MKL_NUM_THREADS"] = "1"
LinearAlgebra.BLAS.set_num_threads(1)

const THREAD_ID = get(ENV, "THREAD_ID", "8full")
const T_FULL    = 480

function main()
    # Static models first: SSD-NS seeds from NS's out_params.csv and 3SSD-NNS from NNS's,
    # both under this same thread id, so they have to be written before the second pair.
    model_names = haskey(ENV, "MODELS") ? String.(split(ENV["MODELS"], ",")) :
                                          ["NS", "NNS", "SSD-NS", "3SSD-NNS"]
    println("Full-sample fits, thread_id=$(THREAD_ID), T=$(T_FULL): ", model_names)

    for model_name in model_names
        println("\n########## $model_name ##########")
        YieldFactorModels.run(THREAD_ID, T_FULL, 12, false, model_name, Float64;
                              run_optimization=true, max_group_iters=10, group_tol=1e-6)
    end
end
main()
