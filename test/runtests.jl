using Test

@testset "SecondOrderOptimization Tests" begin
    include("utils.jl")
    include("stats.jl")
    include("linearsolve.jl")
    include("functional.jl")
    include("nantrial.jl")
    include("trialreuse.jl")
    include("OptimizationWorkspace.jl")
    include("general_objectives.jl")
    include("dogleg_trustregion.jl")
    include("extensions.jl")
end
