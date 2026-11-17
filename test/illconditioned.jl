using SecondOrderOptimization, SparseArrays, LinearAlgebra, Test

# Iterators on problems whose Hessian spans more than the solver resolves

# The Bratu ODE boundary-value problem, -u'' = exp(u) on N interior points, as a sparse
# HessianObjective with H = JᵀJ. J is (N+1)²·tridiag(−1, 2, −1) − diag(exp(u)), so at N = 32768
# cond(JᵀJ) is past 1/eps.
function odebvp_objective(n)
    c = Float64(n + 1)^2
    r = zeros(n)
    function res!(r, u)
        r[1] = (2u[1] - u[2]) * c - exp(u[1])
        @inbounds for i in 2:n-1
            r[i] = (-u[i-1] + 2u[i] - u[i+1]) * c - exp(u[i])
        end
        r[n] = (-u[n-1] + 2u[n]) * c - exp(u[n])
        return r
    end
    jac(u) = spdiagm(-1 => fill(-c, n - 1), 0 => 2c .- exp.(u), 1 => fill(-c, n - 1))
    J0 = jac(zeros(n))
    f(u) = 0.5 * sum(abs2, res!(r, u))
    function fdf!(g, H, u)
        res!(r, u)
        J = jac(u)
        mul!(g, J', r)
        copyto!(nonzeros(H), nonzeros(sparse(J' * J)))   # the same pattern for every u
        return 0.5 * sum(abs2, r)
    end
    return HessianObjective(f, fdf!; H = sparse(J0' * J0))
end

@testset "illconditioned.jl" begin
    # IC1: at N = 32768 the undamped LDLᵀ of the stored JᵀJ is indefinite from rounding alone at
    # iteration 2 (the exact JᵀJ is positive definite, its smallest eigenvalue about 79). A first shift
    # of 1e-4·max_diag (6.9e14) gave a step of norm 3e-12, which ended the run as converged at a
    # cost of 4.7. A shift of eps·max_diag (about 1.5e3), which any positive-definite matrix solves
    # at, is tried first.
    @testset "IC1: a failure from rounding is retried at eps times the diagonal first" begin
        obj = odebvp_objective(32768)
        for iter in (:dogleg, :newton)
            opts = OptimizationOptions(iterator = iter, reldcost = 1e-8, dstep = 1e-8, gtol = 1e-8, maxiters = 500)
            _, r = optimize(zeros(32768), obj, opts)
            @test r.niterations > 3
            @test r.bestcost < 1e-5
        end
    end
end
