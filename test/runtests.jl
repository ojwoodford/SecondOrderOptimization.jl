!in(@__DIR__, LOAD_PATH) && push!(LOAD_PATH, @__DIR__)

using Test

# Includes a test file and logs its wall time and compile time. `compile_time` is reported by `@timed` from
# Julia 1.11; before that it is logged as NaN.
function timed_include(file)
    s = @timed include(file)
    compile = get(s, :compile_time, NaN)
    @info file time = round(s.time; digits = 1) compile = round(compile; digits = 1) gc = round(s.gctime; digits = 1)
    return nothing
end

@testset verbose = true "SecondOrderOptimization Tests" begin
    timed_include("publicapi.jl")
    timed_include("imports.jl")
    timed_include("stats.jl")
    timed_include("linearsolve.jl")
    timed_include("objectives.jl")
    timed_include("jacobianobjective.jl")
    timed_include("jacobianiterators.jl")
    timed_include("tallsystem.jl")
    timed_include("sparsetallsystem.jl")
    timed_include("statictallsystem.jl")
    timed_include("staticrule.jl")
    timed_include("functional.jl")
    timed_include("convergence.jl")
    timed_include("nantrial.jl")
    timed_include("trialreuse.jl")
    timed_include("OptimizationWorkspace.jl")
    timed_include("general_objectives.jl")
    timed_include("dogleg_trustregion.jl")
    timed_include("illconditioned.jl")
    timed_include("extensions.jl")
    timed_include("lsmr.jl")
    timed_include("blocklinearsystems_ext.jl")
    timed_include("qrelimination.jl")
    timed_include("jacobiancombinations.jl")
end
