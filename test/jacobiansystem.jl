using SecondOrderOptimization, Test

# The memory a workspace of the Jacobian system takes, which needs the optimizer's workspace (the
# solve-level tests of the Jacobian system are DampedLinearSolve's)
@testset "jacobiansystem.jl" begin
    @testset "Memory: J, one scratch copy, and one more J for the trial system (Q3)" begin
        m, n = 20_000, 50
        sz = 8 * m * (n + 1)
        obj = JacobianObjective((x) -> 0.0, (r, J, x) -> 0.0, m)
        for (ccs, expected) in ((true, 2), (false, 3))
            opts = OptimizationOptions(callbackchangesstate=ccs)
            OptimizationWorkspace(zeros(n), obj, opts)
            bytes = @allocated OptimizationWorkspace(zeros(n), obj, opts)
            @test (expected - 0.1) * sz < bytes < (expected + 0.3) * sz
        end
    end
end
