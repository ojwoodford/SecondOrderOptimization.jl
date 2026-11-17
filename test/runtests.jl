!in(@__DIR__, LOAD_PATH) && push!(LOAD_PATH, @__DIR__)

using Test

@testset "SecondOrderOptimization Tests" begin
    include("publicapi.jl")
    include("imports.jl")
    include("stats.jl")
    include("linearsolve.jl")
    include("objectives.jl")
    include("jacobianobjective.jl")
    include("jacobianiterators.jl")
    include("tallsystem.jl")
    include("sparsetallsystem.jl")
    include("statictallsystem.jl")
    include("staticrule.jl")
    include("functional.jl")
    include("convergence.jl")
    include("nantrial.jl")
    include("trialreuse.jl")
    include("OptimizationWorkspace.jl")
    include("general_objectives.jl")
    include("dogleg_trustregion.jl")
    include("extensions.jl")
    include("lsmr.jl")
    include("blocklinearsystems_ext.jl")
    include("qrelimination.jl")
    include("jacobiancombinations.jl")
end
