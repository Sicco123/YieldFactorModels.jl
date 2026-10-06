# Which init file `run` picks: this thread's own file when it exists, the shared
# default otherwise. Run: julia --project=/home/skooiker/TVNets/TVNets-1.11.6 \
#                              YieldFactorModels.jl/test_init_lookup.jl
using Pkg
Pkg.activate("/home/skooiker/TVNets/TVNets-1.11.6")
using YieldFactorModels
const P = YieldFactorModels.init_params_path

mktempdir() do d
    f = joinpath(d, "")                       # init_folder is a string ending in a separator
    touch(joinpath(d, "init_params_1C.csv"))
    @assert P(f, "1C", "7") == joinpath(d, "init_params_1C.csv")

    touch(joinpath(d, "init_params_1C__thread_id__7.csv"))
    @assert P(f, "1C", "7") == joinpath(d, "init_params_1C__thread_id__7.csv")
    @assert P(f, "1C", "8") == joinpath(d, "init_params_1C.csv")   # other threads unaffected

    # simulation prefers a _simulation file, but only over the same thread's plain one
    touch(joinpath(d, "init_params_1C_simulation.csv"))
    @assert P(f, "1C", "7"; simulation=true) == joinpath(d, "init_params_1C__thread_id__7.csv")
    @assert P(f, "1C", "8"; simulation=true) == joinpath(d, "init_params_1C_simulation.csv")

    # nothing on disk: the default path, which is what the caller then writes to
    @assert P(f, "NS", "7") == joinpath(d, "init_params_NS.csv")
end
println("init lookup ok")
