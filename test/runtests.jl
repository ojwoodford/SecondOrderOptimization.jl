!in(@__DIR__, LOAD_PATH) && push!(LOAD_PATH, @__DIR__)

using Test

@testset "SecondOrderOptimization Tests" begin
    include("utils.jl")
    include("stats.jl")
    include("linearsolve.jl")
    include("objectives.jl")
    include("conformance.jl")
    include("staticrule.jl")
    include("functional.jl")
    include("nantrial.jl")
    include("trialreuse.jl")
    include("OptimizationWorkspace.jl")
    include("general_objectives.jl")
    include("dogleg_trustregion.jl")
    include("extensions.jl")
    include("blocklinearsystems_ext.jl")
end
