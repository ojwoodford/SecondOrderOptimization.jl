using SecondOrderOptimization, StaticArrays, LinearAlgebra, Test
import DampedLinearSolve

# The parts of the linear-solve tests that need the iterators: a failed solve ends a run with
# TERMINATION_LINSOLVER_FAILED, and the first shift of Newton's retry. (The solve-level tests are
# DampedLinearSolve's.)
@testset "linearsolve.jl" begin
    # 6. Linear solver failure termination signaling via LINSOLVER_FAILED
    unsolvable_cost(x) = x[1]^2 + x[2]^2
    function unsolvable_fdf!(g, H, x)
        g[1] = 2.0 * x[1]
        g[2] = 2.0 * x[2]
        H[1, 1] = 0.0
        H[2, 2] = 0.0
        H[1, 2] = 1e20
        H[2, 1] = 1e20
        return x[1]^2 + x[2]^2
    end

    for iter in (:newton, :levenbergmarquardt)
        opts = OptimizationOptions(iterator=iter, linsolver=:cholesky, maxiters=10)
        x0 = @SVector [1.0, 1.0]
        x_opt, res = optimize(x0, HessianObjective(unsolvable_cost, unsolvable_fdf!), opts)
        @test (res.termination & TERMINATION_LINSOLVER_FAILED) != 0
        @test !isconverged(res)
    end

    # B10: the first Newton shift is finite even when the diagonal contains NaN
    ls_nan = DampedLinearSolve.DenseLinearSystem{Float64}([NaN 0.0; 0.0 1.0], zeros(2), zeros(2))
    @test SecondOrderOptimization.initial_shift(SecondOrderOptimization.OptimizationInternal(ls_nan, zeros(2))) == 1e-6
    ls_big = DampedLinearSolve.DenseLinearSystem{Float64}([1e4 0.0; 0.0 1.0], zeros(2), zeros(2))
    @test SecondOrderOptimization.initial_shift(SecondOrderOptimization.OptimizationInternal(ls_big, zeros(2))) == 1.0
end
