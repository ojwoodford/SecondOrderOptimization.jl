!in(@__DIR__, LOAD_PATH) && push!(LOAD_PATH, @__DIR__)

using Test

# Test groups, chosen by the arguments (`julia test/runtests.jl threaded`, or `Pkg.test(test_args = ["threaded"])`);
# every group when there are none. `threaded`: the tests whose behaviour depends on `Threads.nthreads()`;
# `serial`: everything else. Full check before a push: every group on 1 thread, then `threaded` on 4.
const TEST_GROUPS = (
    serial = ["publicapi.jl", "imports.jl", "stats.jl", "linearsolve.jl", "objectives.jl", "jacobianobjective.jl",
        "jacobianiterators.jl", "jacobiansystem.jl", "sparsejacobiansystem.jl", "staticjacobiansystem.jl", "staticrule.jl",
        "functional.jl", "convergence.jl", "nantrial.jl", "trialreuse.jl", "OptimizationWorkspace.jl",
        "general_objectives.jl", "dogleg_trustregion.jl", "illconditioned.jl", "extensions.jl", "lsmr.jl",
        "blocklinearsystems_ext.jl", "qrelimination.jl", "jacobiancombinations.jl", "edgepaths.jl", "precision.jl", "iteratorinterface.jl", "triallinearization.jl", "iteratorconformance.jl", "collectstats.jl"],
    threaded = ["qrelimination_threaded.jl"],
)
const SELECTED_GROUPS = isempty(ARGS) ? collect(keys(TEST_GROUPS)) : Symbol.(ARGS)
for g in SELECTED_GROUPS
    g in keys(TEST_GROUPS) || error("Unknown test group :$g; the groups are $(join(keys(TEST_GROUPS), ", "))")
end

# Includes a test file and logs its wall time and compile time. `compile_time` is reported by `@timed` from
# Julia 1.11; before that it is logged as NaN.
function timed_include(file)
    s = @timed include(file)
    compile = get(s, :compile_time, NaN)
    @info file time = round(s.time; digits = 1) compile = round(compile; digits = 1) gc = round(s.gctime; digits = 1)
    return nothing
end

@testset verbose = true "SecondOrderOptimization Tests" begin
    for g in SELECTED_GROUPS, file in TEST_GROUPS[g]
        timed_include(file)
    end
end
