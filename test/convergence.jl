using SecondOrderOptimization, StaticArrays, LinearAlgebra, Test
include("termbits.jl")
const SOO = SecondOrderOptimization

# Q6: what counts as converged (isconverged), and the two-sided cost-change tests of optimizeloop!

# A termination value from masks, and isconverged on it
q6conv(masks::Integer...) = SOO.isconverged_bits(reduce(|, masks; init=0))

const Q6_CONVERGENCE = (TERMINATION_REL_DCOST, TERMINATION_ABS_DCOST, TERMINATION_STEP_SIZE, TERMINATION_GRAD_TOL)
const Q6_LIMITS = (TERMINATION_MAX_FAILS, TERMINATION_MAX_ITERS, TERMINATION_MAX_TIME)
const Q6_NONFINITE = (TERMINATION_COST_INF, TERMINATION_COST_NAN, TERMINATION_STEP_INF, TERMINATION_STEP_NAN, TERMINATION_GRAD_NOT_FINITE, TERMINATION_START_NOT_FINITE)

@testset "convergence.jl" begin
    @testset "isconverged on hand-built termination values" begin
        # Q8: termination_reasons is gone; the masks and isconverged are the public interface
        @test !isdefined(SOO, :termination_reasons) && !isdefined(@__MODULE__, :termination_reasons)
        @test all(b -> isdefined(SOO, b) && b in names(SOO), (:TERMINATION_REL_DCOST, :TERMINATION_MAX_FAILS, :TERMINATION_GRAD_TOL, :isconverged))
        @test all(x -> length(x) == 2 && x[2] isa String, SOO.TERMINATION_REASONS)    # bit and message only
        # the rules below cover every bit that has a message
        @test Set(x[1] for x in SOO.TERMINATION_REASONS) == Set([Q6_CONVERGENCE..., Q6_LIMITS..., Q6_NONFINITE..., TERMINATION_COST_DISCREPANCY, TERMINATION_LINSOLVER_FAILED])
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
        @test q6conv(TERMINATION_REL_DCOST, TERMINATION_ABS_DCOST, TERMINATION_MAX_FAILS, TERMINATION_MAX_ITERS, TERMINATION_MAX_TIME)
        # grad_tol with cost_discrepancy is converged; cost_discrepancy alone is not
        @test q6conv(TERMINATION_GRAD_TOL, TERMINATION_COST_DISCREPANCY)
        @test !q6conv(TERMINATION_COST_DISCREPANCY)
        # anything non-finite is a failure whatever else is set
        for nf in Q6_NONFINITE
            @test !q6conv(nf)
            for c in Q6_CONVERGENCE
                @test !q6conv(c, nf)
                @test !q6conv(c, nf, TERMINATION_MAX_ITERS)
            end
        end
        @test !q6conv(TERMINATION_STEP_SIZE, TERMINATION_COST_NAN) && !q6conv(TERMINATION_STEP_SIZE, TERMINATION_COST_INF)        # a small step to a non-finite cost
        @test !q6conv(TERMINATION_GRAD_TOL, TERMINATION_START_NOT_FINITE, TERMINATION_COST_INF)                          # a small gradient at a start with no finite cost
        # linsolver_failed is a failure
        @test !q6conv(TERMINATION_LINSOLVER_FAILED)
        for c in Q6_CONVERGENCE
            @test !q6conv(c, TERMINATION_LINSOLVER_FAILED)
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
        @test isconverged(res) && res.termination == TERMINATION_REL_DCOST
        n = res.niterations
        _, res = optimize([1.0, 0.0], expfit, opts(maxiters=n - 1))
        @test res.termination == TERMINATION_MAX_ITERS && !isconverged(res)              # max_iters alone
        _, res = optimize([1.0, 0.0], expfit, opts(maxiters=n))
        @test res.termination == TERMINATION_REL_DCOST | TERMINATION_MAX_ITERS
        @test isconverged(res)                                                              # a limit with a convergence bit
        # grad_tol alone
        _, res = optimize([-1.2, 1.0], rosen, OptimizationOptions(gtol=1e-6, reldcost=0.0, absdcost=0.0, dstep=0.0))
        @test res.termination == TERMINATION_GRAD_TOL && isconverged(res)
        # a small gradient with a cost discrepancy: fdf! reports a cost 1e-6 above f away from the start
        f(x) = 0.5 * (x[1] - 1)^2
        objd = HessianObjective(f, (g, H, x) -> (g[1] = x[1] - 1; H[1, 1] = 1.0; f(x) + (x[1] == 0.0 ? 0.0 : 1e-6)))
        # (callbackchangesstate = true: the plain Newton variant scores the step with f, the
        # relinearizing one with fdf!, which would hide the discrepancy)
        _, res = optimize([0.0], objd, OptimizationOptions(iterator=:newton, gtol=1e-10, callbackchangesstate=true))
        @test res.termination == TERMINATION_COST_DISCREPANCY | TERMINATION_GRAD_TOL
        @test isconverged(res)
        # a small gradient at a start with no finite cost (converged before Q6)
        objinf = HessianObjective(x -> Inf, (g, H, x) -> (g .= 0.0; H .= I(1); Inf))
        _, res = optimize([0.0], objinf, OptimizationOptions(gtol=1e-3))
        @test hasbits(res, TERMINATION_GRAD_TOL) && hasbits(res, TERMINATION_START_NOT_FINITE)
        @test !isconverged(res)
        # a small step to a point whose cost is not finite: the cost is NaN away from the start, and
        # every trial is rejected until the step is below dstep
        fn(x) = x[1] == 0.0 ? 0.5 * (x[1] - 1)^2 : NaN
        objn = HessianObjective(fn, (g, H, x) -> (g[1] = x[1] - 1; H[1, 1] = 1.0; fn(x)))
        _, res = optimize([0.0], objn, OptimizationOptions(iterator=:dogleg, dstep=1e-3))
        @test hasbits(res, TERMINATION_COST_NAN) && !isconverged(res)
        # a linear-solver failure: an indefinite Hessian that :cholesky cannot solve, even shifted
        objf = HessianObjective(x -> -0.5 * x[1]^2 - x[1], (g, H, x) -> (g[1] = -x[1] - 1; H[1, 1] = -1e300; -0.5 * x[1]^2 - x[1]); H=zeros(1, 1))
        _, res = optimize([0.0], objf, OptimizationOptions(iterator=:levenbergmarquardt, linsolver=:cholesky))
        @test hasbits(res, TERMINATION_LINSOLVER_FAILED) && !isconverged(res)
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
        @test hasbits(res, TERMINATION_REL_DCOST | TERMINATION_ABS_DCOST)
        # the point returned is the best one, also when the last step was a small rise
        @test res.bestcost == minimum([res.startcost; costs])
        @test 0.5 * sum(abs2, resid(x)) == res.bestcost
        # a rise above the tolerance is still a failure, and a run that keeps rising still ends on
        # max_fails, not converged: Newton on √(1 + x²) from |x| > 1 maps x to -x³
        fd(x) = sqrt(1 + x[1]^2)
        objdiv = HessianObjective(fd, (g, H, x) -> (s = fd(x); g[1] = x[1] / s; H[1, 1] = 1 / s^3; s))
        xd, res = optimize([1.5], objdiv, OptimizationOptions(iterator=:newton, maxiters=20))
        @test res.termination == TERMINATION_MAX_FAILS
        @test !isconverged(res)
        @test res.niterations == 4 && xd == [1.5]            # maxfails = 3, and the start is the best point
    end
end

@testset "The default reldcost = 1.5e-8 (Q6)" begin
    @test OptimizationOptions().reldcost == 1.5e-8 && OptimizationOptions().absdcost == 1e-15 && OptimizationOptions().dstep == 1e-15
    @test 1.5e-8 ≈ sqrt(eps()) rtol=0.02
    # Gauss-Newton on the ill-conditioned fit: at the minimum the cost is computable to about 1e-8
    # relative, so with the old default the run bounced until max_fails (not converged); with the
    # default it ends on rel_dcost, at the minimum
    tt = collect(range(0, 1; length=60))
    V = 1e4 * hcat((tt .^ k for k in 0:13)...)
    yv = 1e4 * (@. sin(3tt) + 0.01 * cos(17tt))
    exactcost(x) = Float64(0.5 * sum(abs2, big.(V) * big.(x) - big.(yv)))
    costmin = exactcost(big.(V) \ big.(yv))
    obj = JacobianObjective(x -> 0.5 * sum(abs2, V * x - yv), (r, J, x) -> (r .= V * x - yv; J .= V; 0.5 * sum(abs2, r)), 60)
    for ccs in (false, true)
        x, res = optimize(zeros(14), obj, OptimizationOptions(iterator=:newton, callbackchangesstate=ccs, maxiters=500))
        @test isconverged(res) && hasbits(res, TERMINATION_REL_DCOST)
        @test (exactcost(x) - costmin) / costmin <= 1e-12
        # with the old default it ends on max_fails alone
        _, res = optimize(zeros(14), obj, OptimizationOptions(iterator=:newton, callbackchangesstate=ccs, maxiters=500, reldcost=1e-15))
        @test res.termination == TERMINATION_MAX_FAILS && !isconverged(res)
    end
    # a slowly converging run can stop on the cost test far from its minimum: uniform damping on
    # Rosenbrock with x[2] in units of 1e-6 (the case the documentation warns of). With reldcost
    # lowered it keeps going.
    d = [1.0, 1e-6]
    f(x) = 0.5 * ((1 - x[1] * d[1])^2 + 100 * (x[2] * d[2] - (x[1] * d[1])^2)^2)
    objs = JacobianObjective(f, (r, J, x) -> (r[1] = 1 - x[1] * d[1]; r[2] = 10 * (x[2] * d[2] - (x[1] * d[1])^2);
                                              J[1, 1] = -d[1]; J[2, 1] = -20 * x[1] * d[1]^2; J[2, 2] = 10 * d[2]; f(x)), 2)
    x0 = [-1.2, 1.0] ./ d
    _, res = optimize(x0, objs, OptimizationOptions(maxiters=500))
    @test res.termination == TERMINATION_REL_DCOST && res.bestcost > 0.5
    _, res2 = optimize(x0, objs, OptimizationOptions(maxiters=500, reldcost=1e-15))
    @test res2.termination == TERMINATION_MAX_ITERS && res2.bestcost < res.bestcost
    # the right iterator for it converges with the default
    _, res3 = optimize(x0, objs, OptimizationOptions(iterator=:levenbergmarquardt_dtd, maxiters=500))
    @test isconverged(res3) && res3.bestcost < 1e-20
end
