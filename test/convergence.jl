using SecondOrderOptimization, StaticArrays, LinearAlgebra, Test
const SOO = SecondOrderOptimization

# Q6: what counts as converged (isconverged), and the two-sided cost-change tests of optimizeloop!

# A termination value from reason symbols
function q6bits(reasons::Symbol...)
    t = 0
    for r in reasons
        i = findfirst(x -> x[2] === r, SOO.TERMINATION_REASONS)
        t |= SOO.TERMINATION_REASONS[i][1]
    end
    return t
end
q6conv(reasons::Symbol...) = SOO.isconverged_bits(q6bits(reasons...))

const Q6_CONVERGENCE = (:rel_dcost, :abs_dcost, :step_size, :grad_tol)
const Q6_LIMITS = (:max_fails, :max_iters, :max_time)
const Q6_NONFINITE = (:cost_inf, :cost_nan, :step_inf, :step_nan, :grad_not_finite, :start_not_finite)

@testset "convergence.jl" begin
    @testset "isconverged on hand-built termination values" begin
        # every bit is named, and q6bits builds what termination_reasons reads back
        @test sort([x[2] for x in SOO.TERMINATION_REASONS]) == sort([Q6_CONVERGENCE..., Q6_LIMITS..., Q6_NONFINITE..., :cost_discrepancy, :linsolver_failed])
        # a convergence bit alone is converged
        for c in Q6_CONVERGENCE
            @test q6conv(c)
        end
        # a convergence bit with a limit is converged; a limit alone is not
        for c in Q6_CONVERGENCE, l in Q6_LIMITS
            @test q6conv(c, l)
        end
        for l in Q6_LIMITS
            @test !q6conv(l)
        end
        @test q6conv(:rel_dcost, :abs_dcost, :max_fails, :max_iters, :max_time)
        # grad_tol with cost_discrepancy is converged; cost_discrepancy alone is not
        @test q6conv(:grad_tol, :cost_discrepancy)
        @test !q6conv(:cost_discrepancy)
        # anything non-finite is a failure whatever else is set
        for nf in Q6_NONFINITE
            @test !q6conv(nf)
            for c in Q6_CONVERGENCE
                @test !q6conv(c, nf)
                @test !q6conv(c, nf, :max_iters)
            end
        end
        @test !q6conv(:step_size, :cost_nan) && !q6conv(:step_size, :cost_inf)        # a small step to a non-finite cost
        @test !q6conv(:grad_tol, :start_not_finite, :cost_inf)                          # a small gradient at a start with no finite cost
        # linsolver_failed is a failure
        @test !q6conv(:linsolver_failed)
        for c in Q6_CONVERGENCE
            @test !q6conv(c, :linsolver_failed)
        end
        # nothing set, or only a callback's flags, is not converged
        @test !SOO.isconverged_bits(0)
        @test !SOO.isconverged_bits(1 << 16)
    end

    # The two-residual Rosenbrock, as a Jacobian objective
    function rosen_resjac!(r, J, x)
        r[1] = 1 - x[1]; r[2] = 10 * (x[2] - x[1]^2)
        J[1, 1] = -1.0; J[2, 1] = -20x[1]; J[2, 2] = 10.0
        return 0.5 * (r[1]^2 + r[2]^2)
    end
    rosen = JacobianObjective(x -> 0.5 * ((1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2), rosen_resjac!, 2)
    # An exponential fit with a nonzero residual at its minimum
    tq = collect(range(0, 2; length=25)); yq = @. 3exp(-1.3tq) + 0.05sin(7tq)
    q6resid(x) = @. x[1] * exp(x[2] * tq) - yq
    expfit = JacobianObjective(x -> 0.5 * sum(abs2, q6resid(x)),
                               (r, J, x) -> (r .= q6resid(x); J[:, 1] .= exp.(x[2] .* tq); J[:, 2] .= x[1] .* tq .* exp.(x[2] .* tq); 0.5 * sum(abs2, r)), 25)

    @testset "Constructed runs, one per rule" begin
        # a run that meets a convergence criterion on its last allowed iteration is converged.
        # (grad_tol is tested after a relinearization, which a max_iters stop never reaches, so the
        # criterion here is the change in cost.)
        opts(; kw...) = OptimizationOptions(; reldcost=1e-10, absdcost=0.0, dstep=0.0, kw...)
        _, res = optimize([1.0, 0.0], expfit, opts())
        @test isconverged(res) && termination_reasons(res) == [:rel_dcost]
        n = res.niterations
        _, res = optimize([1.0, 0.0], expfit, opts(maxiters=n - 1))
        @test termination_reasons(res) == [:max_iters] && !isconverged(res)              # max_iters alone
        _, res = optimize([1.0, 0.0], expfit, opts(maxiters=n))
        @test termination_reasons(res) == [:rel_dcost, :max_iters]
        @test isconverged(res)                                                              # a limit with a convergence bit
        # grad_tol alone
        _, res = optimize([-1.2, 1.0], rosen, OptimizationOptions(gtol=1e-6, reldcost=0.0, absdcost=0.0, dstep=0.0))
        @test termination_reasons(res) == [:grad_tol] && isconverged(res)
        # a small gradient with a cost discrepancy: fdf! reports a cost 1e-6 above f away from the start
        f(x) = 0.5 * (x[1] - 1)^2
        objd = HessianObjective(f, (g, H, x) -> (g[1] = x[1] - 1; H[1, 1] = 1.0; f(x) + (x[1] == 0.0 ? 0.0 : 1e-6)))
        # (callbackchangesstate = true: the plain Newton variant scores the step with f, the
        # relinearizing one with fdf!, which would hide the discrepancy)
        _, res = optimize([0.0], objd, OptimizationOptions(iterator=:newton, gtol=1e-10, callbackchangesstate=true))
        @test Set(termination_reasons(res)) == Set([:cost_discrepancy, :grad_tol])
        @test isconverged(res)
        # a small gradient at a start with no finite cost (converged before Q6)
        objinf = HessianObjective(x -> Inf, (g, H, x) -> (g .= 0.0; H .= I(1); Inf))
        _, res = optimize([0.0], objinf, OptimizationOptions(gtol=1e-3))
        @test :grad_tol in termination_reasons(res) && :start_not_finite in termination_reasons(res)
        @test !isconverged(res)
        # a small step to a point whose cost is not finite: the cost is NaN away from the start, and
        # every trial is rejected until the step is below dstep
        fn(x) = x[1] == 0.0 ? 0.5 * (x[1] - 1)^2 : NaN
        objn = HessianObjective(fn, (g, H, x) -> (g[1] = x[1] - 1; H[1, 1] = 1.0; fn(x)))
        _, res = optimize([0.0], objn, OptimizationOptions(iterator=:dogleg, dstep=1e-3))
        @test :cost_nan in termination_reasons(res) && !isconverged(res)
        # a linear-solver failure: an indefinite Hessian that :cholesky cannot solve, even shifted
        objf = HessianObjective(x -> -0.5 * x[1]^2 - x[1], (g, H, x) -> (g[1] = -x[1] - 1; H[1, 1] = -1e300; -0.5 * x[1]^2 - x[1]); H=zeros(1, 1))
        _, res = optimize([0.0], objf, OptimizationOptions(iterator=:levenbergmarquardt, linsolver=:cholesky))
        @test :linsolver_failed in termination_reasons(res) && !isconverged(res)
    end

    @testset "Gauss-Newton converges at the minimum (it ended on max_fails before Q6)" begin
        # a well-conditioned fit: Gauss-Newton accepts every step, and at the minimum the cost moves
        # up and down by rounding; a rise below the tolerance now ends the run as converged
        t = collect(range(0, 2; length=25)); y = @. 3exp(-1.3t) + 0.05sin(7t)
        resid(x) = @. x[1] * exp(x[2] * t) - y
        costs = Float64[]
        obj = JacobianObjective(x -> 0.5 * sum(abs2, resid(x)),
                                (r, J, x) -> (r .= resid(x); J[:, 1] .= exp.(x[2] .* t); J[:, 2] .= x[1] .* t .* exp.(x[2] .* t); 0.5 * sum(abs2, r)), 25)
        x, res = optimize([1.0, 0.0], obj, OptimizationOptions(iterator=:newton); callback=(c, args...) -> (push!(costs, c); (c, 0)))
        @test isconverged(res)
        @test !isempty(intersect(termination_reasons(res), (:rel_dcost, :abs_dcost)))
        # the point returned is the best one, also when the last step was a small rise
        @test res.bestcost == minimum([res.startcost; costs])
        @test 0.5 * sum(abs2, resid(x)) == res.bestcost
        # a rise above the tolerance is still a failure, and a run that keeps rising still ends on
        # max_fails, not converged: Newton on √(1 + x²) from |x| > 1 maps x to -x³
        fd(x) = sqrt(1 + x[1]^2)
        objdiv = HessianObjective(fd, (g, H, x) -> (s = fd(x); g[1] = x[1] / s; H[1, 1] = 1 / s^3; s))
        xd, res = optimize([1.5], objdiv, OptimizationOptions(iterator=:newton, maxiters=20))
        @test termination_reasons(res) == [:max_fails]
        @test !isconverged(res)
        @test res.niterations == 4 && xd == [1.5]            # maxfails = 3, and the start is the best point
    end
end
