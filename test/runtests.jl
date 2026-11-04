using Test

@testset "SecondOrderOptimization Tests" begin
    include("utils.jl")
    include("linearsolve.jl")
    include("functional.jl")
    include("nantrial.jl")
    include("trialreuse.jl")
    include("OptimizationWorkspace.jl")
end
